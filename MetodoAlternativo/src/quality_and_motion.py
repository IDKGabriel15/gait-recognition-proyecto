"""Control de calidad por frame (Frame Quality Gate) y detector cinemático de marcha.

Mejoras sobre Plan 1:
  1. Filtra frames tóxicos (pies cortados en bordes, silueta mínima/anómala,
     baja confianza en piernas) ANTES de que contaminen el pooling temporal.
  2. Detecta marcha activa usando oscilación relativa de tobillos/rodillas +
     velocidad 2D/escala, funcionando tanto lateral (90°) como diagonal/frontal
     (videos verticales de celular).
"""

from __future__ import annotations

import numpy as np


def compute_frame_quality(
    sil_64x44: np.ndarray,
    kpts_17x3: np.ndarray | None = None,
    box_xyxy: np.ndarray | None = None,
    frame_shape: tuple[int, int] | None = None,
    min_sil_area: float = 0.04,
    max_sil_area: float = 0.65,
    min_leg_conf: float = 0.25,
    border_margin_px: int = 3,
) -> tuple[bool, float]:
    """Evalúa la calidad de un frame (silueta + keypoints).

    Retorna `(is_valid, quality_score_0_1)`.
    """
    if sil_64x44 is None or sil_64x44.size == 0:
        return False, 0.0

    area_ratio = float((sil_64x44 > 127).mean())
    if area_ratio < min_sil_area or area_ratio > max_sil_area:
        return False, 0.0

    score = 1.0

    # Penalizar/rechazar si la persona está cortada en el borde inferior/superior
    if box_xyxy is not None and frame_shape is not None:
        fh, fw = frame_shape
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        bh = max(y2 - y1, 1.0)
        bw = max(x2 - x1, 1.0)
        aspect = bh / bw
        if aspect < 1.15:  # Persona agachada, sentada o muy cortada
            return False, 0.0
        if y2 >= fh - border_margin_px or y1 <= border_margin_px:
            # Pies o cabeza tocando el borde del frame
            score *= 0.45
        if x1 <= border_margin_px or x2 >= fw - border_margin_px:
            score *= 0.65

    if kpts_17x3 is not None and kpts_17x3.shape == (17, 3):
        # Caderas (11,12), rodillas (13,14), tobillos (15,16)
        leg_confs = kpts_17x3[[11, 12, 13, 14, 15, 16], 2]
        mean_leg_conf = float(leg_confs.mean())
        visible_legs = int((leg_confs >= min_leg_conf).sum())
        if visible_legs < 3:
            return False, 0.0
        score *= float(np.clip(mean_leg_conf, 0.2, 1.0))

    return score >= 0.28, float(np.clip(score, 0.0, 1.0))


def filter_sequence_quality(
    sil_seq: np.ndarray,
    kpts_seq: np.ndarray | None = None,
    min_valid_frames: int = 10,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Filtra una secuencia completa descartando frames de baja calidad.

    Si el filtro deja menos de `min_valid_frames`, conserva los frames con mejor
    calidad (>0) para no perder secuencias cortas (ej. cámaras de 9 fps).
    """
    n = len(sil_seq)
    if n == 0:
        return sil_seq, kpts_seq, np.zeros((0,), dtype=bool)

    valid_mask = np.zeros(n, dtype=bool)
    scores = np.zeros(n, dtype=np.float32)

    for i in range(n):
        k = kpts_seq[i] if (kpts_seq is not None and i < len(kpts_seq)) else None
        ok, sc = compute_frame_quality(sil_seq[i], kpts_17x3=k)
        valid_mask[i] = ok
        scores[i] = sc

    if int(valid_mask.sum()) < min_valid_frames:
        # Fallback ordenado por mejor puntaje (pero siempre exigiendo silueta no vacía)
        non_empty = sil_seq.sum(axis=(1, 2)) > 0
        if int(non_empty.sum()) <= min_valid_frames:
            valid_mask = non_empty
        else:
            order = np.argsort(scores)[::-1]
            valid_mask[:] = False
            picked = 0
            for idx in order:
                if non_empty[idx]:
                    valid_mask[idx] = True
                    picked += 1
                    if picked >= min_valid_frames:
                        break

    sil_out = sil_seq[valid_mask]
    kpts_out = kpts_seq[valid_mask] if kpts_seq is not None else None
    return sil_out, kpts_out, valid_mask


def is_actively_walking(
    kpts_window: list[np.ndarray] | np.ndarray,
    box_history: list[tuple[float, float, float]],
    conf_thresh: float = 0.25,
) -> tuple[bool, dict]:
    """Detector de marcha invariante a ángulo (lateral, diagonal o frontal).

    Combina:
      1. Oscilación cinemática de tobillos/rodillas normalizada por longitud de torso.
      2. Desplazamiento horizontal (`|vx|`) o cambio de escala (`|vh|`, acercándose/alejándose).
    """
    metrics = {"vx": 0.0, "vh": 0.0, "ankle_osc": 0.0, "knee_osc": 0.0}

    # 1. Velocidad de traslación / escala (BBox)
    if len(box_history) >= 5:
        dt = max(box_history[-1][0] - box_history[0][0], 1e-3)
        metrics["vx"] = abs(box_history[-1][1] - box_history[0][1]) / dt
        metrics["vh"] = abs(box_history[-1][2] - box_history[0][2]) / dt

    # 2. Oscilación relativa de piernas (funciona incluso caminando de frente o en cinta)
    if len(kpts_window) >= 6:
        k_arr = np.asarray(kpts_window[-12:], dtype=np.float32)
        valid = (
            (k_arr[:, 5, 2] >= conf_thresh)
            & (k_arr[:, 6, 2] >= conf_thresh)
            & (k_arr[:, 11, 2] >= conf_thresh)
            & (k_arr[:, 12, 2] >= conf_thresh)
            & (k_arr[:, 15, 2] >= conf_thresh)
            & (k_arr[:, 16, 2] >= conf_thresh)
        )
        if int(valid.sum()) >= 5:
            kv = k_arr[valid]
            mid_sh = 0.5 * (kv[:, 5, :2] + kv[:, 6, :2])
            mid_hp = 0.5 * (kv[:, 11, :2] + kv[:, 12, :2])
            torso = np.maximum(np.median(np.linalg.norm(mid_sh - mid_hp, axis=-1)), 12.0)

            # Variación temporal de la distancia entre tobillos (en X y en Y)
            ankle_dx = (kv[:, 15, 0] - kv[:, 16, 0]) / torso
            ankle_dy = (kv[:, 15, 1] - kv[:, 16, 1]) / torso
            metrics["ankle_osc"] = float(np.std(ankle_dx) + np.std(ankle_dy))

            knee_dx = (kv[:, 13, 0] - kv[:, 14, 0]) / torso
            knee_dy = (kv[:, 13, 1] - kv[:, 14, 1]) / torso
            metrics["knee_osc"] = float(np.std(knee_dx) + np.std(knee_dy))

    walking = (
        (metrics["vx"] >= 55.0)
        or (metrics["vh"] >= 16.0)
        or (metrics["ankle_osc"] >= 0.14)
        or (metrics["knee_osc"] >= 0.10)
    )
    return walking, metrics
