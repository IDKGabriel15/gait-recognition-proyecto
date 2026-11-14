"""Fase 3: Motor de Reconocimiento de Marcha (Enfoque 1 Preentrenado + Fusión Silueta & Esqueleto).

Carga `GaitBase_DA-180000.pt` (OpenGait preentrenado en GREW) para extraer
embeddings de silueta, y los combina con un descriptor biomecánico basado en
los 17 puntos clave COCO (proporciones, ángulos articulares, zancada y cadencia).
Soporta evaluación flexible con 1, 2 o 3 ángulos disponibles sin depender de ninguno.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OPENGAIT_ROOT = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\third_party\OpenGait")
for p in (OPENGAIT_ROOT, OPENGAIT_ROOT / "opengait"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from opengait.modeling.backbones.resnet import ResNet9
from opengait.modeling.modules import (
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
    SeparateBNNecks,
    SeparateFCs,
    SetBlockWrapper,
)


class GaitBaseInference(nn.Module):
    """Arquitectura exacta de GaitBase_DA (GREW) para inferencia en CPU sin DDP."""

    def __init__(self, ckpt_path: str | Path):
        super().__init__()
        backbone_cfg = {
            "type": "ResNet9",
            "block": "BasicBlock",
            "channels": [64, 128, 256, 512],
            "layers": [1, 1, 1, 1],
            "strides": [1, 2, 2, 1],
            "maxpool": False,
        }
        # Construir ResNet9 sin pasar por BaseModel
        resnet = ResNet9(
            block="BasicBlock",
            channels=backbone_cfg["channels"],
            layers=backbone_cfg["layers"],
            strides=backbone_cfg["strides"],
            maxpool=backbone_cfg["maxpool"],
        )
        self.Backbone = SetBlockWrapper(resnet)
        self.FCs = SeparateFCs(in_channels=512, out_channels=256, parts_num=16)
        self.BNNecks = SeparateBNNecks(class_num=20000, in_channels=256, parts_num=16)
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt)
        self.load_state_dict(sd, strict=True)
        self.eval()

    @torch.no_grad()
    def extract_embedding(self, sil_seq: np.ndarray) -> np.ndarray | None:
        """Extrae embedding L2-normalizado (4096-D) desde siluetas (T, 64, 44)."""
        # Filtrar frames vacíos
        valid_mask = sil_seq.sum(axis=(1, 2)) > 0
        sil_valid = sil_seq[valid_mask]
        if len(sil_valid) < 5:
            return None

        # [1, 1, S, 64, 44] normalizado a [0, 1]
        x = torch.from_numpy(sil_valid).float().unsqueeze(0).unsqueeze(0) / 255.0
        outs = self.Backbone(x)  # [1, 512, S, H', W']
        outs = self.TP(outs, seqL=None, options={"dim": 2})[0]  # [1, 512, H', W']
        feat = self.HPP(outs)  # [1, 512, 16]
        embed_1 = self.FCs(feat)  # [1, 256, 16]
        vec = embed_1.reshape(-1).cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return None
        return vec / norm


def _angle_series(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Calcula serie temporal de ángulos (radianes) en el vértice b."""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)) + 1e-6
    cos_v = np.clip(np.sum(ba * bc, axis=-1) / denom, -1.0, 1.0)
    return np.arccos(cos_v)


def extract_skeleton_embedding(kpts_seq: np.ndarray, conf_thresh: float = 0.25) -> np.ndarray | None:
    """Extrae descriptor biomecánico L2-normalizado desde keypoints (T, 17, 3)."""
    if len(kpts_seq) < 5:
        return None

    # Filtrar frames con caderas y hombros visibles
    valid = (
        (kpts_seq[:, 5, 2] >= conf_thresh)
        & (kpts_seq[:, 6, 2] >= conf_thresh)
        & (kpts_seq[:, 11, 2] >= conf_thresh)
        & (kpts_seq[:, 12, 2] >= conf_thresh)
    )
    k = kpts_seq[valid]
    if len(k) < 5:
        return None

    xy = k[:, :, :2]  # (T, 17, 2)
    mid_hip = (xy[:, 11, :] + xy[:, 12, :]) / 2.0
    mid_shoulder = (xy[:, 5, :] + xy[:, 6, :]) / 2.0
    torso_len = np.linalg.norm(mid_shoulder - mid_hip, axis=-1, keepdims=True)
    scale = np.maximum(np.median(torso_len), 10.0)

    # Coordenadas normalizadas invariantes a traslación y escala
    norm_xy = (xy - mid_hip[:, None, :]) / scale  # (T, 17, 2)

    # 1. Estadísticas espaciales de postura (34 + 34 = 68 dims)
    mean_pose = norm_xy.mean(axis=0).reshape(-1)
    std_pose = norm_xy.std(axis=0).reshape(-1)

    # 2. Ángulos articulares dinámicos (rodillas, caderas, codos)
    lk_ang = _angle_series(xy[:, 11], xy[:, 13], xy[:, 15])
    rk_ang = _angle_series(xy[:, 12], xy[:, 14], xy[:, 16])
    lh_ang = _angle_series(xy[:, 5], xy[:, 11], xy[:, 13])
    rh_ang = _angle_series(xy[:, 6], xy[:, 12], xy[:, 14])
    le_ang = _angle_series(xy[:, 5], xy[:, 7], xy[:, 9])
    re_ang = _angle_series(xy[:, 6], xy[:, 8], xy[:, 10])

    angles = np.stack([lk_ang, rk_ang, lh_ang, rh_ang, le_ang, re_ang], axis=1)  # (T, 6)
    ang_stats = np.concatenate(
        [
            angles.mean(axis=0),
            angles.std(axis=0),
            np.percentile(angles, 10, axis=0),
            np.percentile(angles, 90, axis=0),
        ]
    )  # 24 dims

    # 3. Proporciones y zancada (ancho de hombros, longitud de piernas, apertura de tobillos)
    shoulder_w = np.linalg.norm(norm_xy[:, 5] - norm_xy[:, 6], axis=-1)
    hip_w = np.linalg.norm(norm_xy[:, 11] - norm_xy[:, 12], axis=-1)
    ankle_dist = np.linalg.norm(norm_xy[:, 15] - norm_xy[:, 16], axis=-1)
    l_leg = np.linalg.norm(norm_xy[:, 11] - norm_xy[:, 13], axis=-1) + np.linalg.norm(
        norm_xy[:, 13] - norm_xy[:, 15], axis=-1
    )
    r_leg = np.linalg.norm(norm_xy[:, 12] - norm_xy[:, 14], axis=-1) + np.linalg.norm(
        norm_xy[:, 14] - norm_xy[:, 16], axis=-1
    )
    geom_stats = np.array(
        [
            shoulder_w.mean(),
            hip_w.mean(),
            ankle_dist.mean(),
            ankle_dist.std(),
            np.percentile(ankle_dist, 95),
            l_leg.mean(),
            r_leg.mean(),
        ],
        dtype=np.float32,
    )

    feat = np.concatenate([mean_pose, std_pose, ang_stats, geom_stats]).astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm < 1e-8:
        return None
    return feat / norm


def match_probe_against_gallery(
    probe_views: dict[str, tuple[np.ndarray | None, np.ndarray | None]],
    gallery_db: dict[str, list[dict]],
    alpha_sil: float = 0.85,
    exclude_same_angle: bool = False,
) -> list[tuple[str, float]]:
    """Compara un probe (1 a 3 ángulos) contra toda la galería sin depender de ningún ángulo.

    `probe_views`: {angle: (sil_emb, skel_emb)}
    `gallery_db`: {subject_name: [{"angle": str, "sil": np.ndarray, "skel": np.ndarray}, ...]}
    Retorna lista ordenada [(subject, score), ...] de mayor a menor similitud.
    """
    # Calcular media global de skel_emb en la galería para centrar (evitar similitud 0.999)
    all_skel = [g["skel"] for entries in gallery_db.values() for g in entries if g.get("skel") is not None]
    skel_mean = np.mean(all_skel, axis=0) if all_skel else None

    def _norm_centered(v: np.ndarray | None) -> np.ndarray | None:
        if v is None:
            return None
        vc = v - skel_mean if skel_mean is not None else v
        n = np.linalg.norm(vc)
        return vc / n if n > 1e-8 else None

    scores = []
    for subj, entries in gallery_db.items():
        view_best_scores = []
        for p_angle, (p_sil, p_skel) in probe_views.items():
            p_skel_c = _norm_centered(p_skel)
            best_for_view = -1.0
            for g in entries:
                if exclude_same_angle and g["angle"] == p_angle:
                    continue
                s_sil = float(np.dot(p_sil, g["sil"])) if (p_sil is not None and g["sil"] is not None) else None
                g_skel_c = _norm_centered(g.get("skel"))
                s_skel = float(np.dot(p_skel_c, g_skel_c)) if (p_skel_c is not None and g_skel_c is not None) else None
                if s_sil is not None and s_skel is not None:
                    sim = alpha_sil * s_sil + (1.0 - alpha_sil) * s_skel
                elif s_sil is not None:
                    sim = s_sil
                elif s_skel is not None:
                    sim = s_skel
                else:
                    continue
                if g["angle"] == p_angle:
                    sim += 0.02
                if sim > best_for_view:
                    best_for_view = sim
            if best_for_view > -1.0:
                view_best_scores.append(best_for_view)

        if view_best_scores:
            scores.append((subj, float(np.mean(view_best_scores))))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


if __name__ == "__main__":
    ckpt = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\weights\GaitBase_DA-180000.pt")
    model = GaitBaseInference(ckpt)
    sample_dir = Path(
        r"c:\Users\angel.roman\Downloads\ProyectoIA\outputs\features\external_test\AlejandroPrueba\toma_01\pass_01\frente"
    )
    sil = np.load(sample_dir / "silhouettes.npy")
    kpt = np.load(sample_dir / "keypoints.npy")
    s_emb = model.extract_embedding(sil)
    k_emb = extract_skeleton_embedding(kpt)
    print("GaitBase cargado OK!")
    print("  sil_emb shape:", None if s_emb is None else s_emb.shape)
    print("  skel_emb shape:", None if k_emb is None else k_emb.shape)
