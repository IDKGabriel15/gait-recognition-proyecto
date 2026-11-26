"""Evaluación 'In-the-Wild' en los 5 videos no controlados reales (Plan 1 vs Plan 2).

Videos evaluados en la raíz del proyecto:
  - AlejandroNoControlado.mp4                    -> Alejandro
  - BernardoBejaranoNocontrolado.mp4             -> BernardoBejarano
  - GabrielRoman(Video escena no controlada).mp4 -> GabrielRoman
  - GabrielRomanNoControlado2.mp4                -> GabrielRoman
  - JoseLuisNocontrolado.mp4                     -> JoseLuis

Usa el nuevo extractor `MultiPersonGaitExtractor` (ByteTrack + ROI-Crop Pose)
+ `Frame Quality Gate` + `is_actively_walking` y evalúa cada modelo sobre las
mismas ventanas de caminata extraídas.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
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

from domain_adapter import PartWisePCAWhitener
from heatmap_generator import flip_sil_and_maps, generate_skeleton_maps
from matcher_v2 import VectorizedGaitMatcher
from models_v2 import DeepGaitV2Inference, SkeletonGaitPPInference
from phase3_gait_engine import GaitBaseInference
from quality_and_motion import compute_frame_quality, is_actively_walking
from tracker_extractor import MultiPersonGaitExtractor

WILD_VIDEOS = [
    ("AlejandroNoControlado.mp4", "Alejandro"),
    ("BernardoBejaranoNocontrolado.mp4", "BernardoBejarano"),
    ("GabrielRoman(Video escena no controlada).mp4", "GabrielRoman"),
    ("GabrielRomanNoControlado2.mp4", "GabrielRoman"),
    ("JoseLuisNocontrolado.mp4", "JoseLuis"),
]


def extract_wild_windows(
    video_path: Path,
    extractor: MultiPersonGaitExtractor,
    window_frames: int = 60,
    eval_interval: int = 15,
    max_frames: int = 900,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extrae ventanas limpias `(sil_seq, kpts_seq)` del sujeto principal en un video libre."""
    extractor.reset_tracker()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Buffers por track_id
    sil_bufs: dict[int, deque] = defaultdict(lambda: deque(maxlen=window_frames))
    kpt_bufs: dict[int, deque] = defaultdict(lambda: deque(maxlen=window_frames))
    box_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
    valid_cnt: dict[int, int] = defaultdict(int)
    windows_by_tid: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        t_sec = frame_idx / fps
        fh, fw = frame.shape[:2]

        dets = extractor.process_frame(frame, use_tracker=True)
        for d in dets:
            tid = d["track_id"]
            box = d["box"]
            sil = d["sil"]
            kpts = d["kpts"]
            cx = float(0.5 * (box[0] + box[2]))
            bh = float(box[3] - box[1])
            box_hist[tid].append((t_sec, cx, bh))

            ok_q, _ = compute_frame_quality(sil, kpts_17x3=kpts, box_xyxy=box, frame_shape=(fh, fw))
            walking, _ = is_actively_walking(list(kpt_bufs[tid]) + [kpts], list(box_hist[tid]))

            if ok_q and walking and sil is not None:
                sil_bufs[tid].append(sil)
                kpt_bufs[tid].append(kpts)
                valid_cnt[tid] += 1
                if len(sil_bufs[tid]) >= 18 and (valid_cnt[tid] % eval_interval == 0):
                    windows_by_tid[tid].append(
                        (np.stack(sil_bufs[tid], axis=0), np.stack(kpt_bufs[tid], axis=0))
                    )
        frame_idx += 1

    cap.release()

    if not windows_by_tid:
        # Fallback: si el video es muy corto/difícil, devolver el buffer más largo
        best_tid = max(sil_bufs.keys(), key=lambda t: len(sil_bufs[t]), default=None)
        if best_tid is not None and len(sil_bufs[best_tid]) >= 8:
            return [(np.stack(sil_bufs[best_tid], axis=0), np.stack(kpt_bufs[best_tid], axis=0))]
        return []

    # Elegir el track_id con más ventanas de marcha válidas (sujeto protagonista)
    main_tid = max(windows_by_tid.keys(), key=lambda t: len(windows_by_tid[t]))
    return windows_by_tid[main_tid]


def build_matcher_from_cache(
    model_key: str,
    top_k_parts: int = 10,
    top_k_gallery: int = 3,
    use_tta: bool = True,
    use_as_norm: bool = False,
    use_whitening: bool = True,
) -> VectorizedGaitMatcher:
    cache_path = OUT_V2 / f"cache_{model_key}.pkl"
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    gal_entries = [
        {
            "subject": r["subject"],
            "angle": r["angle"],
            "emb": r["emb"],
            "emb_flip": r["emb_flip"] if use_tta else None,
        }
        for r in payload["records"]
        if r["split"] == "gallery" and r["emb"] is not None
    ]
    adapter = None
    if use_whitening:
        adapter = PartWisePCAWhitener(16, 256, whiten_power=0.5).fit(
            np.stack([e["emb"] for e in gal_entries], axis=0)
        )
    return VectorizedGaitMatcher(
        gal_entries,
        top_k_parts=top_k_parts,
        top_k_gallery=top_k_gallery,
        use_as_norm=use_as_norm,
        adapter=adapter,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluación In-the-Wild (5 videos no controlados)")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gaitbase_p1", "deepgaitv2", "skeletongait", "skeletongaitpp"],
    )
    parser.add_argument("--overwrite-windows", action="store_true")
    args = parser.parse_args()

    win_cache = OUT_V2 / "wild_windows_cache.pkl"
    if win_cache.exists() and not args.overwrite_windows:
        print(f"[CACHE] Cargando ventanas wild desde {win_cache.name}...", flush=True)
        with open(win_cache, "rb") as f:
            wild_windows = pickle.load(f)
    else:
        extractor = MultiPersonGaitExtractor()
        wild_windows = {}
        for fname, gt in WILD_VIDEOS:
            vpath = ROOT / fname
            if not vpath.exists():
                print(f"[WARN] No existe: {vpath}", flush=True)
                continue
            t0 = time.perf_counter()
            wins = extract_wild_windows(vpath, extractor)
            print(f"[WILD-EXTRACT] {fname} ({gt}): {len(wins)} ventanas en {time.perf_counter()-t0:.1f}s", flush=True)
            wild_windows[fname] = {"gt": gt, "windows": wins}
        OUT_V2.mkdir(parents=True, exist_ok=True)
        with open(win_cache, "wb") as f:
            pickle.dump(wild_windows, f)

    models_obj = {}
    matchers = {}
    for mk in args.models:
        if mk == "gaitbase_p1":
            models_obj[mk] = GaitBaseInference(ROOT / "weights" / "GaitBase_DA-180000.pt")
            matchers[mk] = build_matcher_from_cache(mk, top_k_parts=8, top_k_gallery=1, use_tta=False, use_as_norm=False, use_whitening=False)
        elif mk == "deepgaitv2":
            models_obj[mk] = DeepGaitV2Inference(WEIGHTS_V2 / "DeepGaitV2_GREW.pt")
            matchers[mk] = build_matcher_from_cache(mk, top_k_parts=10, top_k_gallery=3, use_tta=True, use_as_norm=False, use_whitening=True)
        elif mk == "skeletongait":
            models_obj[mk] = DeepGaitV2Inference(WEIGHTS_V2 / "SkeletonGait_GREW.pt")
            matchers[mk] = build_matcher_from_cache(mk, top_k_parts=12, top_k_gallery=3, use_tta=True, use_as_norm=False, use_whitening=True)
        elif mk == "skeletongaitpp":
            models_obj[mk] = SkeletonGaitPPInference(WEIGHTS_V2 / "SkeletonGaitPP_GREW.pt")
            matchers[mk] = build_matcher_from_cache(mk, top_k_parts=10, top_k_gallery=3, use_tta=True, use_as_norm=False, use_whitening=True)

    report = {}
    for mk in args.models:
        print(f"\n================ WILD BENCHMARK: {mk} ================", flush=True)
        m = models_obj[mk]
        matcher = matchers[mk]
        correct_vids = 0
        total_vids = 0
        vid_res = {}

        for fname, info in wild_windows.items():
            gt = info["gt"]
            wins = info["windows"]
            if not wins:
                print(f"  {fname:44s} | GT={gt:16s} | SIN VENTANAS", flush=True)
                continue
            total_vids += 1
            votes = Counter()
            score_sums = defaultdict(float)
            for sil_seq, kpt_seq in wins:
                if mk == "gaitbase_p1":
                    emb = m.extract_embedding(sil_seq)
                    ranked = matcher.match_views({"wild": (emb, None)})
                elif mk == "deepgaitv2":
                    emb = m.extract_embedding(sil_seq)
                    sil_f, _ = flip_sil_and_maps(sil_seq, None)
                    emb_f = m.extract_embedding(sil_f)
                    ranked = matcher.match_views({"wild": (emb, emb_f)})
                elif mk == "skeletongait":
                    hm = generate_skeleton_maps(kpt_seq)
                    emb = m.extract_embedding(hm)
                    _, hm_f = flip_sil_and_maps(None, hm)
                    emb_f = m.extract_embedding(hm_f)
                    ranked = matcher.match_views({"wild": (emb, emb_f)})
                elif mk == "skeletongaitpp":
                    hm = generate_skeleton_maps(kpt_seq)
                    emb = m.extract_embedding(sil_seq, hm)
                    sil_f, hm_f = flip_sil_and_maps(sil_seq, hm)
                    emb_f = m.extract_embedding(sil_f, hm_f)
                    ranked = matcher.match_views({"wild": (emb, emb_f)})
                else:
                    ranked = []

                if ranked:
                    votes[ranked[0][0]] += 1
                    for subj, sc in ranked[:5]:
                        score_sums[subj] += sc

            # Decisión por consenso de votos (desempate por score acumulado)
            if votes:
                pred = max(votes.keys(), key=lambda s: (votes[s], score_sums[s]))
            else:
                pred = "Desconocido"

            ok = pred == gt
            if ok:
                correct_vids += 1
            gt_votes = votes.get(gt, 0)
            top3_v = votes.most_common(3)
            tag = "OK  " if ok else "FAIL"
            print(
                f"  [{tag}] {fname:42s} | GT={gt:16s} | Pred={pred:16s} | "
                f"GT_votos={gt_votes}/{len(wins)} | Top3={top3_v}",
                flush=True,
            )
            vid_res[fname] = {
                "gt": gt,
                "pred": pred,
                "ok": ok,
                "windows": len(wins),
                "gt_votes": gt_votes,
                "top3_votes": top3_v,
            }

        print(f"  >>> TOTAL {mk}: {correct_vids}/{total_vids} videos no controlados acertados", flush=True)
        report[mk] = {"correct": correct_vids, "total": total_vids, "videos": vid_res}

    out_p = OUT_V2 / "wild_benchmark_results.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReporte Wild guardado en: {out_p}", flush=True)


if __name__ == "__main__":
    main()
