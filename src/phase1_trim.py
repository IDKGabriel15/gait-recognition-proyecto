"""Fase 1: Recorte automático de tiempo muerto sincronizado entre cámaras.

Analiza la trayectoria del sujeto en la cámara ancla (normalmente `frente`,
30 fps) para detectar las pasadas activas de caminata (eliminando los tramos
donde la persona está parada o dando la vuelta en los extremos del pasillo).
Recorta de forma sincronizada todas las cámaras disponibles (1 a 3 ángulos)
para cada pasada hacia `dataset_trimmed/`.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MANIFEST_PATH = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\configs\dataset_manifest.json")
TRIMMED_ROOT = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\dataset_trimmed")


def load_timestamps_csv(csv_path: str | None) -> list[int]:
    if not csv_path or not Path(csv_path).exists():
        return []
    timestamps = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(int(row["master_timestamp_ms"]))
    return timestamps


def detect_walking_passes_ms(
    video_path: str,
    timestamps_ms: list[int],
    model: YOLO,
    sample_stride: int = 3,
    min_height_px: int = 64,
    min_conf: float = 0.45,
    min_speed_px_s: float = 85.0,
    min_pass_duration_s: float = 1.4,
    pad_ms: int = 150,
) -> tuple[list[dict], dict] | None:
    """Detecta los intervalos [t_start_ms, t_end_ms] de caminata activa."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = 1 if fps < 15 else sample_stride

    samples = []  # (t_ms, cx, cy, h)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            results = model.predict(
                frame, classes=[0], conf=min_conf, imgsz=416, verbose=False
            )
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                heights = boxes[:, 3] - boxes[:, 1]
                best_idx = int(np.argmax(heights))
                if heights[best_idx] >= min_height_px:
                    t_ms = (
                        timestamps_ms[frame_idx]
                        if frame_idx < len(timestamps_ms)
                        else int((frame_idx / fps) * 1000)
                    )
                    x1, y1, x2, y2 = boxes[best_idx]
                    samples.append(
                        (t_ms, float((x1 + x2) / 2.0), float((y1 + y2) / 2.0), float(heights[best_idx]))
                    )
        frame_idx += 1

    cap.release()

    if len(samples) < 5:
        return None

    t_arr = np.array([s[0] for s in samples], dtype=np.float64)
    cx_arr = np.array([s[1] for s in samples], dtype=np.float64)

    # Suavizado ligero de cx para evitar ruido cuadro a cuadro
    kernel = np.ones(3) / 3.0
    cx_smooth = np.convolve(cx_arr, kernel, mode="same")
    dt_s = np.maximum(np.gradient(t_arr) / 1000.0, 1e-3)
    vx = np.gradient(cx_smooth) / dt_s
    speed = np.abs(vx)

    is_walking = speed >= min_speed_px_s

    # Rellenar huecos breves (<= 2 muestras) dentro de una misma pasada
    for i in range(1, len(is_walking) - 1):
        if not is_walking[i] and is_walking[i - 1] and is_walking[i + 1]:
            is_walking[i] = True

    passes = []
    in_seg = False
    seg_start = 0

    for i, active in enumerate(is_walking):
        if active and not in_seg:
            in_seg = True
            seg_start = i
        elif not active and in_seg:
            in_seg = False
            seg_end = i - 1
            dur_s = (t_arr[seg_end] - t_arr[seg_start]) / 1000.0
            if dur_s >= min_pass_duration_s:
                dx = cx_arr[seg_end] - cx_arr[seg_start]
                passes.append(
                    {
                        "t_start_ms": int(t_arr[seg_start] - pad_ms),
                        "t_end_ms": int(t_arr[seg_end] + pad_ms),
                        "duration_sec": round(dur_s, 2),
                        "direction": "L2R" if dx >= 0 else "R2L",
                        "displacement_px": round(abs(float(dx)), 1),
                    }
                )
    if in_seg:
        seg_end = len(is_walking) - 1
        dur_s = (t_arr[seg_end] - t_arr[seg_start]) / 1000.0
        if dur_s >= min_pass_duration_s:
            dx = cx_arr[seg_end] - cx_arr[seg_start]
            passes.append(
                {
                    "t_start_ms": int(t_arr[seg_start] - pad_ms),
                    "t_end_ms": int(t_arr[seg_end] + pad_ms),
                    "duration_sec": round(dur_s, 2),
                    "direction": "L2R" if dx >= 0 else "R2L",
                    "displacement_px": round(abs(float(dx)), 1),
                }
            )

    if not passes:
        return None

    stats = {
        "anchor_total_frames": total_frames,
        "anchor_fps": round(fps, 2),
        "num_passes": len(passes),
    }
    return passes, stats


def trim_video_by_ms(
    src_video: str,
    dst_video: Path,
    timestamps_ms: list[int],
    t_start_ms: int,
    t_end_ms: int,
    anchor_t0_ms: int,
) -> dict | None:
    cap = cv2.VideoCapture(src_video)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    if timestamps_ms:
        valid_indices = [
            i for i, t in enumerate(timestamps_ms) if t_start_ms <= t <= t_end_ms
        ]
        if valid_indices:
            start_f, end_f = valid_indices[0], min(valid_indices[-1], total_frames - 1)
        else:
            cap.release()
            return None
    else:
        rel_start = max(0.0, (t_start_ms - anchor_t0_ms) / 1000.0)
        rel_end = max(rel_start + 0.5, (t_end_ms - anchor_t0_ms) / 1000.0)
        start_f = max(0, int(rel_start * fps))
        end_f = min(total_frames - 1, int(rel_end * fps))

    dst_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dst_video), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    written = 0
    for _ in range(start_f, end_f + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        written += 1

    cap.release()
    out.release()

    if written == 0:
        return None

    return {
        "video_path": str(dst_video),
        "orig_frames": total_frames,
        "trimmed_frames": written,
        "start_frame": start_f,
        "end_frame": end_f,
        "fps": round(fps, 2),
    }


def process_take(take: dict, out_dir: Path, model: YOLO, overwrite: bool = False) -> dict | None:
    meta_path = out_dir / "trim_metadata.json"
    if meta_path.exists() and not overwrite:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    videos = take["videos"]
    anchor_angle = "frente" if "frente" in videos else take["angles"][0]
    anchor_info = videos[anchor_angle]
    anchor_ts = load_timestamps_csv(anchor_info.get("timestamps_csv"))

    res = detect_walking_passes_ms(anchor_info["video_path"], anchor_ts, model)
    if res is None:
        return None
    passes, stats = res
    anchor_t0_ms = anchor_ts[0] if anchor_ts else 0

    passes_out = []
    for p_idx, p in enumerate(passes, start=1):
        pass_name = f"pass_{p_idx:02d}"
        pass_dir = out_dir / pass_name
        trimmed_videos = {}
        for angle, vinfo in videos.items():
            ts_ms = load_timestamps_csv(vinfo.get("timestamps_csv"))
            src_path = Path(vinfo["video_path"])
            dst_path = pass_dir / f"{angle}.mp4"
            t_res = trim_video_by_ms(
                str(src_path), dst_path, ts_ms, p["t_start_ms"], p["t_end_ms"], anchor_t0_ms
            )
            if t_res is not None:
                trimmed_videos[angle] = t_res

        if trimmed_videos:
            passes_out.append(
                {
                    "pass_name": pass_name,
                    **p,
                    "videos": trimmed_videos,
                }
            )

    if not passes_out:
        return None

    result = {
        "take_name": take["take_name"],
        "anchor_angle": anchor_angle,
        "stats": stats,
        "passes": passes_out,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def main():
    parser = argparse.ArgumentParser(description="Fase 1: Recorte de tiempo muerto")
    parser.add_argument("--subject", type=str, default=None, help="Procesar solo un sujeto (ej. Alejandro)")
    parser.add_argument("--external-only", action="store_true", help="Procesar solo carpetas *Prueba")
    parser.add_argument("--limit-takes", type=int, default=0, help="Límite de tomas por sujeto (0 = todas)")
    parser.add_argument("--overwrite", action="store_true", help="Sobreescribir recortes existentes")
    args = parser.parse_args()

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    model = YOLO("yolov8n.pt")
    processed = 0
    total_passes = 0
    skipped = 0

    if args.external_only:
        groups = [("external_test", manifest["external_test"])]
    else:
        groups = [
            ("subjects", manifest["subjects"]),
            ("external_test", manifest["external_test"]),
        ]

    for group_name, group_dict in groups:
        for subj, sdata in group_dict.items():
            if args.subject and subj != args.subject:
                continue
            takes = (
                sdata if group_name == "external_test" else sdata["gallery"] + sdata["probe"]
            )
            if args.limit_takes > 0:
                takes = takes[: args.limit_takes]
            for take in takes:
                out_dir = TRIMMED_ROOT / group_name / subj / take["take_name"]
                res = process_take(take, out_dir, model, overwrite=args.overwrite)
                if res:
                    processed += 1
                    n_p = len(res["passes"])
                    total_passes += n_p
                    durs = [f"{p['duration_sec']}s({p['direction']})" for p in res["passes"]]
                    print(f"[OK] {subj}/{take['take_name']} -> {n_p} pasadas: {', '.join(durs)}")
                else:
                    skipped += 1
                    print(f"[SKIP] {subj}/{take['take_name']} (sin pasadas activas)")

    print(f"\nResumen Fase 1: {processed} tomas -> {total_passes} pasadas limpias, {skipped} omitidas.")


if __name__ == "__main__":
    main()
