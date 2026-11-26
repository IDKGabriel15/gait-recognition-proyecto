"""Extractor rápido y multi-persona v2: Tracker Húngaro (IoU + Centroide) + ROI-Crop Pose.

Mejoras sobre Plan 1 (`phase2_extract_and_visualize.py` / `phase4_realtime_sim.py`):
  1. Reemplaza `argmax(heights)` por seguimiento multi-persona real basado en asignación
     óptima Húngara (`scipy.optimize.linear_sum_assignment`, cero dependencias externas
     como `lap`), manteniendo `track_id` estables incluso con varias personas en escena.
  2. Ejecuta `YOLOv8n-pose` **solo sobre el recorte ROI expandido** de cada sujeto en
     lugar de la imagen completa: mejora la nitidez de rodillas/tobillos en personas
     lejanas (videos de celular) y reduce el costo en CPU.
  3. Interpolación bilineal de máscaras antes del umbral (`> 0.5`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\outputs").resolve()),
)
from ultralytics import YOLO

ROOT = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA")
sys.path.insert(0, str(ROOT / "src"))
from phase2_extract_and_visualize import align_silhouette_opengait


def _box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matricial entre `a (M, 4)` y `b (N, 4)` en formato xyxy."""
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0][None, :], b[:, 1][None, :], b[:, 2][None, :], b[:, 3][None, :]
    ix1, iy1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


class SimpleHungarianTracker:
    """Tracker multi-persona ligero (IoU + predicción lineal de velocidad) sin dependencia `lap`."""

    def __init__(self, iou_thresh: float = 0.25, max_age: int = 20):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_id = 1
        self.tracks: dict[int, dict] = {}  # tid -> {"box": xyxy, "vel": dxyxy, "age": int}

    def reset(self) -> None:
        self.next_id = 1
        self.tracks.clear()

    def update(self, det_boxes: np.ndarray) -> np.ndarray:
        """Asigna un `track_id` a cada caja en `det_boxes (N, 4)`."""
        n_det = len(det_boxes)
        if n_det == 0:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
            return np.zeros((0,), dtype=np.int64)

        if not self.tracks:
            assigned = []
            for box in det_boxes:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"box": box.copy(), "vel": np.zeros(4, dtype=np.float32), "age": 0}
                assigned.append(tid)
            return np.array(assigned, dtype=np.int64)

        tids = list(self.tracks.keys())
        pred_boxes = np.stack([self.tracks[t]["box"] + 0.5 * self.tracks[t]["vel"] for t in tids], axis=0)
        iou = _box_iou_matrix(pred_boxes, det_boxes)
        row_ind, col_ind = linear_sum_assignment(1.0 - iou)

        assigned = np.full(n_det, -1, dtype=np.int64)
        matched_tids = set()
        for r, c in zip(row_ind, col_ind):
            if iou[r, c] >= self.iou_thresh:
                tid = tids[r]
                old_box = self.tracks[tid]["box"]
                new_box = det_boxes[c]
                self.tracks[tid]["vel"] = 0.6 * (new_box - old_box) + 0.4 * self.tracks[tid]["vel"]
                self.tracks[tid]["box"] = new_box.copy()
                self.tracks[tid]["age"] = 0
                assigned[c] = tid
                matched_tids.add(tid)

        # Envejecer tracks no emparejados
        for tid in tids:
            if tid not in matched_tids:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]

        # Crear nuevos tracks para detecciones no emparejadas
        for c in range(n_det):
            if assigned[c] == -1:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"box": det_boxes[c].copy(), "vel": np.zeros(4, dtype=np.float32), "age": 0}
                assigned[c] = tid

        return assigned


class MultiPersonGaitExtractor:
    """Extrae `(track_id, box, aligned_sil_64x44, kpts_17x3)` por cada persona en un frame."""

    def __init__(
        self,
        seg_weights: str | Path = ROOT / "yolov8n-seg.pt",
        pose_weights: str | Path = ROOT / "yolov8n-pose.pt",
        seg_imgsz: int = 512,
        pose_roi_imgsz: int = 256,
        roi_pad_ratio: float = 0.18,
        conf: float = 0.40,
    ):
        self.seg_model = YOLO(str(seg_weights))
        self.pose_model = YOLO(str(pose_weights))
        self.seg_imgsz = seg_imgsz
        self.pose_roi_imgsz = pose_roi_imgsz
        self.roi_pad_ratio = roi_pad_ratio
        self.conf = conf
        self.tracker = SimpleHungarianTracker()

    def reset_tracker(self) -> None:
        self.tracker.reset()

    def process_frame(self, frame: np.ndarray, use_tracker: bool = True) -> list[dict]:
        """Procesa un frame BGR y devuelve detecciones por persona rastreada."""
        fh, fw = frame.shape[:2]
        seg_res = self.seg_model.predict(
            frame,
            classes=[0],
            conf=self.conf,
            imgsz=self.seg_imgsz,
            verbose=False,
        )

        detections = []
        if not seg_res or seg_res[0].masks is None or len(seg_res[0].boxes) == 0:
            if use_tracker:
                self.tracker.update(np.zeros((0, 4), dtype=np.float32))
            return detections

        boxes = seg_res[0].boxes.xyxy.cpu().numpy().astype(np.float32)
        masks_raw = seg_res[0].masks.data.cpu().numpy()

        keep = [
            i for i in range(len(boxes))
            if (boxes[i, 3] - boxes[i, 1]) >= 48 and (boxes[i, 2] - boxes[i, 0]) >= 16
        ]
        if not keep:
            if use_tracker:
                self.tracker.update(np.zeros((0, 4), dtype=np.float32))
            return detections

        boxes = boxes[keep]
        masks_raw = masks_raw[keep]
        ids = self.tracker.update(boxes) if use_tracker else np.arange(len(boxes))

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            bh = float(y2 - y1)
            bw = float(x2 - x1)

            # 1. Silueta binaria con interpolación bilineal antes del umbral
            m_float = cv2.resize(masks_raw[i], (fw, fh), interpolation=cv2.INTER_LINEAR)
            mask_full = (m_float > 0.5).astype(np.uint8) * 255
            px, py = int(bw * 0.08), int(bh * 0.05)
            bx1, by1 = max(0, int(x1) - px), max(0, int(y1) - py)
            bx2, by2 = min(fw, int(x2) + px), min(fh, int(y2) + py)
            mask_clean = np.zeros_like(mask_full)
            mask_clean[by1:by2, bx1:bx2] = mask_full[by1:by2, bx1:bx2]
            aligned_sil = align_silhouette_opengait(mask_clean)

            # 2. ROI-Crop Pose: correr pose solo sobre el recorte expandido de esta persona
            pad_x, pad_y = int(bw * self.roi_pad_ratio), int(bh * self.roi_pad_ratio)
            rx1, ry1 = max(0, int(x1) - pad_x), max(0, int(y1) - pad_y)
            rx2, ry2 = min(fw, int(x2) + pad_x), min(fh, int(y2) + pad_y)
            roi = frame[ry1:ry2, rx1:rx2]

            kpts = np.zeros((17, 3), dtype=np.float32)
            if roi.size > 0:
                pose_res = self.pose_model.predict(
                    roi,
                    classes=[0],
                    conf=0.25,
                    imgsz=self.pose_roi_imgsz,
                    verbose=False,
                )
                if pose_res and pose_res[0].keypoints is not None and len(pose_res[0].boxes) > 0:
                    pboxes = pose_res[0].boxes.xyxy.cpu().numpy()
                    pbi = int(np.argmax(pboxes[:, 3] - pboxes[:, 1]))
                    kp_roi = pose_res[0].keypoints.data[pbi].cpu().numpy().copy()
                    valid_kp = kp_roi[:, 2] > 0.01
                    kp_roi[valid_kp, 0] += rx1
                    kp_roi[valid_kp, 1] += ry1
                    kpts = kp_roi

            detections.append(
                {
                    "track_id": int(ids[i]),
                    "box": boxes[i],
                    "sil": aligned_sil,
                    "kpts": kpts,
                    "mask_bbox": (bx1, by1, bx2, by2),
                }
            )

        detections.sort(key=lambda d: float(d["box"][3] - d["box"][1]), reverse=True)
        return detections
