"""Demo / Inferencia en Tiempo Real v2 (Multi-persona, ByteTrack + ROI-Crop Pose + P3D).

Características sobre Plan 1:
  - Soporta múltiples personas simultáneamente (`track_id` independiente).
  - Visualiza silueta `64x44` + Skeleton Map `2x64x44` (huesos + articulaciones).
  - Filtra frames cortados/ruidosos con `Frame Quality Gate`.
  - Detecta marcha aunque la persona camine en diagonal o de frente (`is_actively_walking`).
  - Usa `VectorizedGaitMatcher` (<1 ms) con `TTA Horizontal Flip` + `AS-Norm`.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA")
V2_ROOT = ROOT / "MetodoAlternativo"
OUT_V2 = V2_ROOT / "outputs"
WEIGHTS_V2 = V2_ROOT / "weights"

sys.path.insert(0, str(V2_ROOT / "src"))
sys.path.insert(0, str(ROOT / "src"))

from heatmap_generator import flip_sil_and_maps, generate_skeleton_maps
from matcher_v2 import VectorizedGaitMatcher
from models_v2 import DeepGaitV2Inference, SkeletonGaitPPInference
from phase2_extract_and_visualize import draw_skeleton
from quality_and_motion import compute_frame_quality, is_actively_walking
from tracker_extractor import MultiPersonGaitExtractor


def _load_model_and_matcher(model_key: str, use_as_norm: bool = True):
    cache_path = OUT_V2 / f"cache_{model_key}.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No existe caché {cache_path}. Ejecuta primero MetodoAlternativo/eval/benchmark_ab.py."
        )
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)

    gal_entries = [
        {
            "subject": r["subject"],
            "angle": r["angle"],
            "emb": r["emb"],
            "emb_flip": r["emb_flip"],
        }
        for r in payload["records"]
        if r["split"] == "gallery" and r["emb"] is not None
    ]
    matcher = VectorizedGaitMatcher(
        gal_entries,
        top_k_parts=10,
        top_k_gallery=3,
        use_as_norm=use_as_norm,
    )

    if model_key == "deepgaitv2":
        model = DeepGaitV2Inference(WEIGHTS_V2 / "DeepGaitV2_GREW.pt")
    elif model_key == "skeletongait":
        model = DeepGaitV2Inference(WEIGHTS_V2 / "SkeletonGait_GREW.pt")
    elif model_key == "skeletongaitpp":
        model = SkeletonGaitPPInference(WEIGHTS_V2 / "SkeletonGaitPP_GREW.pt")
    else:
        raise ValueError(f"Modelo no soportado: {model_key}")
    return model, matcher


def run_realtime_v2(
    source: str,
    model_key: str = "skeletongaitpp",
    out_video: Path | None = None,
    show: bool = False,
    window_frames: int = 60,
    eval_interval: int = 15,
    min_as_score: float = 1.6,
    min_as_margin: float = 0.25,
):
    model, matcher = _load_model_and_matcher(model_key, use_as_norm=True)
    extractor = MultiPersonGaitExtractor()

    cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW) if source.isdigit() else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir fuente: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if out_video is not None:
        out_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

    sil_bufs: dict[int, deque] = defaultdict(lambda: deque(maxlen=window_frames))
    kpt_bufs: dict[int, deque] = defaultdict(lambda: deque(maxlen=window_frames))
    box_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
    valid_cnt: dict[int, int] = defaultdict(int)
    decisions: dict[int, tuple[str, float, float]] = {}
    votes_by_tid: dict[int, Counter] = defaultdict(Counter)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_sec = frame_idx / fps
        dets = extractor.process_frame(frame, use_tracker=True)
        vis = frame.copy()

        for d in dets:
            tid = d["track_id"]
            box = d["box"]
            sil = d["sil"]
            kpts = d["kpts"]
            cx = float(0.5 * (box[0] + box[2]))
            bh = float(box[3] - box[1])
            box_hist[tid].append((t_sec, cx, bh))

            ok_q, q_sc = compute_frame_quality(sil, kpts_17x3=kpts, box_xyxy=box, frame_shape=(fh, fw))
            walking, _ = is_actively_walking(list(kpt_bufs[tid]) + [kpts], list(box_hist[tid]))

            if ok_q and walking and sil is not None:
                sil_bufs[tid].append(sil)
                kpt_bufs[tid].append(kpts)
                valid_cnt[tid] += 1

                if len(sil_bufs[tid]) >= 18 and (valid_cnt[tid] % eval_interval == 0):
                    s_arr = np.stack(sil_bufs[tid], axis=0)
                    k_arr = np.stack(kpt_bufs[tid], axis=0)
                    if model_key == "deepgaitv2":
                        emb = model.extract_embedding(s_arr)
                        sf, _ = flip_sil_and_maps(s_arr, None)
                        emb_f = model.extract_embedding(sf)
                    elif model_key == "skeletongait":
                        hm = generate_skeleton_maps(k_arr)
                        emb = model.extract_embedding(hm)
                        _, hmf = flip_sil_and_maps(None, hm)
                        emb_f = model.extract_embedding(hmf)
                    else:
                        hm = generate_skeleton_maps(k_arr)
                        emb = model.extract_embedding(s_arr, hm)
                        sf, hmf = flip_sil_and_maps(s_arr, hm)
                        emb_f = model.extract_embedding(sf, hmf)

                    ranked = matcher.match_views({"live": (emb, emb_f)})
                    if ranked:
                        top1_name, top1_sc = ranked[0]
                        margin = top1_sc - (ranked[1][1] if len(ranked) > 1 else 0.0)
                        label = (
                            top1_name
                            if (top1_sc >= min_as_score and margin >= min_as_margin)
                            else "Desconocido"
                        )
                        decisions[tid] = (label, top1_sc, margin)
                        votes_by_tid[tid][label] += 1

            label, sc, mg = decisions.get(tid, ("Analizando...", 0.0, 0.0))
            color = (0, 230, 140) if label not in ("Analizando...", "Desconocido") else (0, 165, 255)
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            draw_skeleton(vis, kpts, thickness=2, radius=3)
            txt = f"ID:{tid} {label} (AS:{sc:.2f}, Q:{q_sc:.2f})"
            cv2.putText(vis, txt, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if writer is not None:
            writer.write(vis)
        if show:
            cv2.imshow(f"MetodoAlternativo v2 ({model_key})", vis)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q"), ord("Q")):
                break
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"\n=== RESUMEN REALTIME V2 ({model_key}) ===")
    for tid, cnt in votes_by_tid.items():
        print(f"  Track #{tid}: {dict(cnt)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default="skeletongaitpp", choices=["deepgaitv2", "skeletongait", "skeletongaitpp"])
    parser.add_argument("--out-video", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run_realtime_v2(
        args.source,
        model_key=args.model,
        out_video=Path(args.out_video) if args.out_video else None,
        show=args.show,
    )
