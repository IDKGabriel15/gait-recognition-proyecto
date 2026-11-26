"""Generador rápido y exacto de Skeleton Maps (2x64x44) compatible con OpenGait SkeletonGait / SkeletonGait++.

Replica fielmente `third_party/OpenGait/datasets/pretreatment_heatmap.py`:
  - Canal 0: Mapa de extremidades/huesos (17 conexiones COCO-17 con decaimiento Gaussiano `sigma=8` en 128x128).
  - Canal 1: Mapa de articulaciones (17 keypoints COCO-17 con decaimiento Gaussiano `sigma=8` en 128x128).
  - Alineado a 64x64 y recortado centrado a 64x44 (`[10:54]`), exactamente igual que `BaseSilCuttingTransform`.
  - Incluye helper de espejado horizontal (`flip_sequence_for_tta`) para Test-Time Augmentation.
"""

from __future__ import annotations

import cv2
import numpy as np

# Conexiones oficiales COCO-17 de OpenGait `GeneratePoseTarget`
COCO17_LIMBS = (
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),
    (7, 9), (0, 6), (6, 8), (8, 10), (5, 11), (11, 13),
    (13, 15), (6, 12), (12, 14), (14, 16), (11, 12),
)

# Pares Izquierda <-> Derecha para Test-Time Horizontal Flip
COCO17_FLIP_PERM = np.array(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15], dtype=np.int64
)


def _center_and_scale_coco17(
    kpts_seq: np.ndarray, canvas_size: int = 128, min_conf: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Centra en el punto medio de caderas y normaliza escala vertical al estándar OpenGait."""
    t_len = len(kpts_seq)
    xy = kpts_seq[:, :, :2].astype(np.float32).copy()
    scores = kpts_seq[:, :, 2].astype(np.float32).copy()

    # Reemplazar puntos con confianza nula por la mediana válida del propio frame para evitar rangos extremos
    norm_xy = np.zeros_like(xy)
    valid_frame = np.zeros(t_len, dtype=bool)
    target_h = float(canvas_size // 1.5)  # 85.0 para canvas=128
    half = canvas_size / 2.0

    for t in range(t_len):
        sc = scores[t]
        pts = xy[t]
        good = (sc >= min_conf) & ((pts[:, 0] > 0) | (pts[:, 1] > 0))
        if int(good.sum()) < 4:
            continue

        # Centro de cadera; si falta alguna cadera usar centroide de puntos válidos
        if sc[11] >= min_conf and sc[12] >= min_conf:
            hip = 0.5 * (pts[11] + pts[12])
        else:
            hip = pts[good].mean(axis=0)

        rel = pts - hip[None, :]
        y_vals = rel[good, 1]
        y_span = float(y_vals.max() - y_vals.min())
        if y_span < 8.0:
            continue

        scale = target_h / y_span
        rel = rel * scale + half
        norm_xy[t] = rel
        valid_frame[t] = True

    return norm_xy, scores * valid_frame[:, None]


def generate_skeleton_maps(
    kpts_seq: np.ndarray,
    canvas_size: int = 128,
    sigma: float = 8.0,
    out_h: int = 64,
    out_w: int = 44,
    min_conf: float = 0.05,
) -> np.ndarray:
    """Convierte `(T, 17, 3)` COCO keypoints -> Skeleton Maps `(T, 2, 64, 44)` en uint8 `[0, 255]`.

    Canal 0 = extremidades (bones), Canal 1 = articulaciones (joints).
    """
    t_len = len(kpts_seq)
    if t_len == 0:
        return np.zeros((0, 2, out_h, out_w), dtype=np.uint8)

    norm_xy, scores = _center_and_scale_coco17(kpts_seq, canvas_size=canvas_size, min_conf=min_conf)
    out = np.zeros((t_len, 2, out_h, out_w), dtype=np.uint8)

    inv_2sig2 = 1.0 / (2.0 * sigma * sigma)
    rad = int(np.ceil(3.0 * sigma))
    cut = (out_h - out_w) // 2  # (64 - 44)//2 = 10

    for t in range(t_len):
        sc = scores[t]
        if float(sc.max()) < min_conf:
            continue
        pts = norm_xy[t]

        bone_map = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        joint_map = np.zeros((canvas_size, canvas_size), dtype=np.float32)

        # Canal 1: Articulaciones (Gaussianas 2D)
        for j in range(17):
            conf = float(sc[j])
            if conf < min_conf:
                continue
            mx, my = float(pts[j, 0]), float(pts[j, 1])
            x1 = max(0, int(mx - rad))
            x2 = min(canvas_size, int(mx + rad) + 1)
            y1 = max(0, int(my - rad))
            y2 = min(canvas_size, int(my + rad) + 1)
            if x1 >= x2 or y1 >= y2:
                continue
            xs = np.arange(x1, x2, dtype=np.float32)[None, :]
            ys = np.arange(y1, y2, dtype=np.float32)[:, None]
            patch = np.exp(-((xs - mx) ** 2 + (ys - my) ** 2) * inv_2sig2) * conf
            joint_map[y1:y2, x1:x2] = np.maximum(joint_map[y1:y2, x1:x2], patch)

        # Canal 0: Extremidades / Huesos (distancia al segmento)
        for a_idx, b_idx in COCO17_LIMBS:
            conf = float(min(sc[a_idx], sc[b_idx]))
            if conf < min_conf:
                continue
            ax, ay = float(pts[a_idx, 0]), float(pts[a_idx, 1])
            bx, by = float(pts[b_idx, 0]), float(pts[b_idx, 1])
            x1 = max(0, int(min(ax, bx) - rad))
            x2 = min(canvas_size, int(max(ax, bx) + rad) + 1)
            y1 = max(0, int(min(ay, by) - rad))
            y2 = min(canvas_size, int(max(ay, by) + rad) + 1)
            if x1 >= x2 or y1 >= y2:
                continue
            xs = np.arange(x1, x2, dtype=np.float32)[None, :]
            ys = np.arange(y1, y2, dtype=np.float32)[:, None]
            dx, dy = bx - ax, by - ay
            d2_ab = dx * dx + dy * dy
            if d2_ab < 1.0:
                d2_seg = (xs - ax) ** 2 + (ys - ay) ** 2
            else:
                proj = np.clip(((xs - ax) * dx + (ys - ay) * dy) / d2_ab, 0.0, 1.0)
                px = ax + proj * dx
                py = ay + proj * dy
                d2_seg = (xs - px) ** 2 + (ys - py) ** 2
            patch = np.exp(-d2_seg * inv_2sig2) * conf
            bone_map[y1:y2, x1:x2] = np.maximum(bone_map[y1:y2, x1:x2], patch)

        # Alineación vertical + recorte cuadrado centrado -> 64x64 -> corte central 64x44
        for ch_idx, raw in enumerate((bone_map, joint_map)):
            raw_u8 = np.clip(raw * 255.0, 0, 255).astype(np.uint8)
            y_sum = raw_u8.sum(axis=1)
            nz = np.where(y_sum > 0)[0]
            if len(nz) > 2:
                y_top, y_btm = int(nz[0]), int(nz[-1])
                h = y_btm - y_top + 1
                half_h = h // 2
                cx = canvas_size // 2
                cx1 = max(0, cx - half_h)
                cx2 = min(canvas_size, cx + half_h + 1)
                crop = raw_u8[y_top : y_btm + 1, cx1:cx2]
                if crop.size > 0:
                    sq64 = cv2.resize(crop, (out_h, out_h), interpolation=cv2.INTER_AREA)
                else:
                    sq64 = cv2.resize(raw_u8, (out_h, out_h), interpolation=cv2.INTER_AREA)
            else:
                sq64 = cv2.resize(raw_u8, (out_h, out_h), interpolation=cv2.INTER_AREA)

            if cut > 0:
                out[t, ch_idx] = sq64[:, cut : cut + out_w]
            else:
                out[t, ch_idx] = cv2.resize(sq64, (out_w, out_h), interpolation=cv2.INTER_AREA)

    return out


def flip_sil_and_maps(
    sil_seq: np.ndarray | None,
    map_seq: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Espejado horizontal de silueta `(T, H, W)` y skeleton maps `(T, 2, H, W)` para TTA."""
    sil_f = sil_seq[:, :, ::-1].copy() if sil_seq is not None else None
    map_f = map_seq[:, :, :, ::-1].copy() if map_seq is not None else None
    return sil_f, map_f
