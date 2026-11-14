"""Fase 2: Extracción de Silueta + Puntos Clave (Esqueleto) y Visualización Gráfica.

Procesa los clips recortados de `dataset_trimmed/` usando YOLOv8-seg y YOLOv8-pose:
  1. Extrae siluetas binarias alineadas (64x44 estilo OpenGait) -> silhouettes.npy
  2. Extrae los 17 puntos clave COCO (x, y, conf) -> keypoints.npy
  3. Opcionalmente (--render-video) genera un video gráfico de 3 paneles:
     [Video + BBox + Esqueleto] | [Esqueleto + Estela de pasos + Ángulos] | [Silueta limpia + 64x44]
"""

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

TRIMMED_ROOT = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\dataset_trimmed")
FEATURES_ROOT = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\outputs\features")
VISUALS_ROOT = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\outputs\visuals")

# Conexiones COCO 17 keypoints
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # Cabeza
    (5, 6), (5, 11), (6, 12), (11, 12),    # Torso
    (5, 7), (7, 9),                        # Brazo izq
    (6, 8), (8, 10),                       # Brazo der
    (11, 13), (13, 15),                    # Pierna izq
    (12, 14), (14, 16),                    # Pierna der
]

LEFT_INDICES = {1, 3, 5, 7, 9, 11, 13, 15}
RIGHT_INDICES = {2, 4, 6, 8, 10, 12, 14, 16}


def align_silhouette_opengait(mask_bin: np.ndarray, out_h: int = 64, out_w: int = 44) -> np.ndarray | None:
    """Alinea una máscara binaria (0/255) al estándar OpenGait (64x44)."""
    ys, xs = np.where(mask_bin > 127)
    if len(ys) < 20:
        return None
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    h = y_max - y_min + 1
    w = x_max - x_min + 1
    if h < 16 or w < 4:
        return None

    crop = mask_bin[y_min : y_max + 1, x_min : x_max + 1]
    scale = out_h / float(h)
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(crop, (new_w, out_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((out_h, out_w), dtype=np.uint8)
    # Centrar horizontalmente por centro de masa de x
    fg_y, fg_x = np.where(resized > 127)
    cx = int(round(fg_x.mean())) if len(fg_x) > 0 else new_w // 2
    target_cx = out_w // 2
    x_offset = target_cx - cx

    src_x1 = max(0, -x_offset)
    src_x2 = min(new_w, out_w - x_offset)
    dst_x1 = max(0, x_offset)
    dst_x2 = dst_x1 + (src_x2 - src_x1)

    if src_x2 > src_x1 and dst_x2 > dst_x1:
        canvas[:, dst_x1:dst_x2] = resized[:, src_x1:src_x2]

    return canvas


def calc_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """Ángulo en grados en el vértice b formado por a-b-c."""
    if a[2] < 0.3 or b[2] < 0.3 or c[2] < 0.3:
        return None
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    norm = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cos_val = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def draw_skeleton(
    canvas: np.ndarray,
    kpts: np.ndarray,
    conf_thresh: float = 0.3,
    thickness: int = 3,
    radius: int = 5,
) -> None:
    """Dibuja el esqueleto COCO con colores diferenciados Izq/Der/Torso."""
    for i, j in COCO_SKELETON:
        if kpts[i, 2] >= conf_thresh and kpts[j, 2] >= conf_thresh:
            p1 = (int(kpts[i, 0]), int(kpts[i, 1]))
            p2 = (int(kpts[j, 0]), int(kpts[j, 1]))
            if i in LEFT_INDICES and j in LEFT_INDICES:
                color = (80, 230, 120)  # Verde (Izq)
            elif i in RIGHT_INDICES and j in RIGHT_INDICES:
                color = (40, 165, 255)  # Naranja (Der)
            else:
                color = (240, 220, 90)  # Cian (Torso/Cabeza)
            cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)

    for idx in range(len(kpts)):
        if kpts[idx, 2] >= conf_thresh:
            pt = (int(kpts[idx, 0]), int(kpts[idx, 1]))
            if idx in LEFT_INDICES:
                c = (120, 255, 160)
            elif idx in RIGHT_INDICES:
                c = (80, 195, 255)
            else:
                c = (255, 255, 255)
            cv2.circle(canvas, pt, radius, c, -1, cv2.LINE_AA)


def process_video(
    video_path: Path,
    feat_dir: Path,
    seg_model: YOLO,
    pose_model: YOLO,
    visual_path: Path | None = None,
    overwrite: bool = False,
) -> dict:
    sil_path = feat_dir / "silhouettes.npy"
    kpt_path = feat_dir / "keypoints.npy"

    if sil_path.exists() and kpt_path.exists() and not overwrite and visual_path is None:
        s = np.load(sil_path)
        return {"frames": int(s.shape[0]), "cached": True}

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    panel_w, panel_h = 480, 360
    if visual_path is not None:
        visual_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(visual_path), fourcc, fps, (panel_w * 3, panel_h))

    silhouettes = []
    keypoints_seq = []
    l_ankle_trail = deque(maxlen=18)
    r_ankle_trail = deque(maxlen=18)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        seg_res = seg_model.predict(frame, classes=[0], conf=0.45, imgsz=512, verbose=False)
        pose_res = pose_model.predict(frame, classes=[0], conf=0.45, imgsz=512, verbose=False)

        best_mask_full = np.zeros((h, w), dtype=np.uint8)
        best_box = None
        if seg_res and seg_res[0].masks is not None and len(seg_res[0].boxes) > 0:
            boxes = seg_res[0].boxes.xyxy.cpu().numpy()
            heights = boxes[:, 3] - boxes[:, 1]
            bi = int(np.argmax(heights))
            best_box = boxes[bi]
            mask_raw = seg_res[0].masks.data[bi].cpu().numpy()
            mask_resized = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)
            best_mask_full = (mask_resized > 0.5).astype(np.uint8) * 255

        aligned_sil = align_silhouette_opengait(best_mask_full)
        if aligned_sil is not None:
            silhouettes.append(aligned_sil)
        else:
            silhouettes.append(np.zeros((64, 44), dtype=np.uint8))

        kpts = np.zeros((17, 3), dtype=np.float32)
        if pose_res and pose_res[0].keypoints is not None and len(pose_res[0].boxes) > 0:
            pboxes = pose_res[0].boxes.xyxy.cpu().numpy()
            pheights = pboxes[:, 3] - pboxes[:, 1]
            pbi = int(np.argmax(pheights))
            kpts = pose_res[0].keypoints.data[pbi].cpu().numpy()
        keypoints_seq.append(kpts)

        if writer is not None:
            # Panel 1: Video original + BBox + Esqueleto
            p1 = frame.copy()
            if best_box is not None:
                x1, y1, x2, y2 = [int(v) for v in best_box]
                cv2.rectangle(p1, (x1, y1), (x2, y2), (0, 220, 255), 2)
            draw_skeleton(p1, kpts, thickness=2, radius=4)
            p1_r = cv2.resize(p1, (panel_w, panel_h))
            cv2.putText(p1_r, f"Original + Pose (F:{frame_idx})", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Panel 2: Esqueleto puro + estela de pasos + ángulos
            p2 = np.full((h, w, 3), 22, dtype=np.uint8)
            if kpts[15, 2] >= 0.3:
                l_ankle_trail.append((int(kpts[15, 0]), int(kpts[15, 1])))
            if kpts[16, 2] >= 0.3:
                r_ankle_trail.append((int(kpts[16, 0]), int(kpts[16, 1])))
            for t_i in range(1, len(l_ankle_trail)):
                cv2.line(p2, l_ankle_trail[t_i - 1], l_ankle_trail[t_i], (60, 190, 90), 2)
            for t_i in range(1, len(r_ankle_trail)):
                cv2.line(p2, r_ankle_trail[t_i - 1], r_ankle_trail[t_i], (40, 140, 230), 2)
            draw_skeleton(p2, kpts, thickness=4, radius=6)
            p2_r = cv2.resize(p2, (panel_w, panel_h))

            lk_ang = calc_angle_deg(kpts[11], kpts[13], kpts[15])
            rk_ang = calc_angle_deg(kpts[12], kpts[14], kpts[16])
            ang_txt = f"Rodilla I:{lk_ang:.0f}deg" if lk_ang else "Rodilla I:--"
            ang_txt += f" | D:{rk_ang:.0f}deg" if rk_ang else " | D:--"
            cv2.putText(p2_r, "Esqueleto & Estela de Marcha", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 220, 90), 2)
            cv2.putText(p2_r, ang_txt, (12, panel_h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2)

            # Panel 3: Silueta binaria + miniatura OpenGait 64x44
            p3 = cv2.cvtColor(best_mask_full, cv2.COLOR_GRAY2BGR)
            p3_r = cv2.resize(p3, (panel_w, panel_h))
            if aligned_sil is not None:
                sil_big = cv2.resize(aligned_sil, (88, 128), interpolation=cv2.INTER_NEAREST)
                sil_big_bgr = cv2.cvtColor(sil_big, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(p3_r, (panel_w - 102, 10), (panel_w - 10, 142), (0, 220, 255), 1)
                p3_r[12:140, panel_w - 100 : panel_w - 12] = sil_big_bgr
            cv2.putText(p3_r, "Silueta + OpenGait 64x44", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

            composite = np.hstack([p1_r, p2_r, p3_r])
            writer.write(composite)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    feat_dir.mkdir(parents=True, exist_ok=True)
    sil_arr = np.stack(silhouettes, axis=0) if silhouettes else np.zeros((0, 64, 44), dtype=np.uint8)
    kpt_arr = np.stack(keypoints_seq, axis=0) if keypoints_seq else np.zeros((0, 17, 3), dtype=np.float32)
    np.save(sil_path, sil_arr)
    np.save(kpt_path, kpt_arr)

    return {"frames": int(sil_arr.shape[0]), "cached": False}


def main():
    parser = argparse.ArgumentParser(description="Fase 2: Silueta + Esqueleto y Visualización")
    parser.add_argument("--group", type=str, default="external_test", choices=["external_test", "subjects", "all"])
    parser.add_argument("--subject", type=str, default=None, help="Uno o varios sujetos separados por coma")
    parser.add_argument("--limit-passes", type=int, default=0, help="Límite total de pasadas por sujeto")
    parser.add_argument("--max-passes-per-take", type=int, default=0, help="Máximo de pasadas por toma (0=todas)")
    parser.add_argument("--render-video", action="store_true", help="Renderizar video visual de 3 paneles")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seg_weights = "yolov8n-seg.pt"
    pose_weights = "yolov8n-pose.pt"
    print(f"Usando modelos: seg={seg_weights}, pose={pose_weights}", flush=True)
    seg_model = YOLO(seg_weights)
    pose_model = YOLO(pose_weights)

    subj_filter = set(args.subject.split(",")) if args.subject else None
    groups = ["external_test", "subjects"] if args.group == "all" else [args.group]
    count = 0

    for grp in groups:
        grp_dir = TRIMMED_ROOT / grp
        if not grp_dir.exists():
            continue
        for subj_dir in sorted(grp_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            if subj_filter and subj_dir.name not in subj_filter:
                continue
            pass_count = 0
            for take_dir in sorted(subj_dir.iterdir()):
                if not take_dir.is_dir():
                    continue
                take_pass_count = 0
                for pass_dir in sorted(take_dir.iterdir()):
                    if not pass_dir.is_dir() or not pass_dir.name.startswith("pass_"):
                        continue
                    if args.limit_passes > 0 and pass_count >= args.limit_passes:
                        break
                    if args.max_passes_per_take > 0 and take_pass_count >= args.max_passes_per_take:
                        break
                    pass_count += 1
                    take_pass_count += 1
                    for vid in sorted(pass_dir.glob("*.mp4")):
                        angle = vid.stem
                        feat_dir = FEATURES_ROOT / grp / subj_dir.name / take_dir.name / pass_dir.name / angle
                        vis_path = (
                            VISUALS_ROOT / f"{subj_dir.name}_{take_dir.name}_{pass_dir.name}_{angle}_visual.mp4"
                            if args.render_video
                            else None
                        )
                        res = process_video(
                            vid, feat_dir, seg_model, pose_model, visual_path=vis_path, overwrite=args.overwrite
                        )
                        count += 1
                        tag = "CACHED" if res.get("cached") else "OK"
                        print(f"[{tag}] {subj_dir.name}/{take_dir.name}/{pass_dir.name}/{angle}: {res['frames']} frames", flush=True)

    print(f"\nResumen Fase 2: {count} clips procesados.", flush=True)


if __name__ == "__main__":
    main()

