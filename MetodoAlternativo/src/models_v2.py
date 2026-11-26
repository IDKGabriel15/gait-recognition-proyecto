"""Loaders limpios de inferencia en CPU (sin DDP) para arquitecturas SOTA de OpenGait:

  1. `DeepGaitV2Inference`     — Silueta (1 canal, 64x44) con bloques Pseudo-3D (P3D).
  2. `SkeletonGaitInference`   — Solo Skeleton Maps (2 canales, 64x44) con bloques P3D.
  3. `SkeletonGaitPPInference` — Fusión temprana AttentionFusion (Silueta 1-ch + Skeleton Maps 2-ch, P3D).

Todos devuelven embeddings `4096-D` ordenados por franja `(16, 256)` (`part-major`),
compatibles directamente con `VectorizedGaitMatcher` y `PartWisePCAWhitener`.
"""

from __future__ import annotations

import copy
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

from opengait.modeling.modules import (
    BasicBlock2D,
    BasicBlockP3D,
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
    SeparateBNNecks,
    SeparateFCs,
    SetBlockWrapper,
    conv1x1,
    conv3x3,
)


class _AttentionFusion(nn.Module):
    """Exacta a `AttentionFusion` en `third_party/OpenGait/opengait/modeling/models/skeletongait++.py`."""

    def __init__(self, in_channels: int = 64, squeeze_ratio: int = 16):
        super().__init__()
        hidden_dim = int(in_channels / squeeze_ratio)
        self.conv = SetBlockWrapper(
            nn.Sequential(
                conv1x1(in_channels * 2, hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                conv3x3(hidden_dim, hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                conv1x1(hidden_dim, in_channels * 2),
            )
        )

    def forward(self, sil_feat: torch.Tensor, map_feat: torch.Tensor) -> torch.Tensor:
        # sil_feat, map_feat: [n, c, s, h, w]
        feats = torch.cat([sil_feat, map_feat], dim=1)
        score = self.conv(feats)  # [n, 2*c, s, h, w]
        n, dc, s, h, w = score.shape
        score = score.view(n, 2, dc // 2, s, h, w)
        score = F.softmax(score, dim=1)
        return sil_feat * score[:, 0] + map_feat * score[:, 1]


def _make_p3d_layer(inplanes: int, block_cls, planes: int, stride: list[int], blocks_num: int, mode: str = "p3d"):
    if max(stride) > 1 or inplanes != planes * block_cls.expansion:
        if mode == "2d":
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block_cls.expansion, stride=stride),
                nn.BatchNorm2d(planes * block_cls.expansion),
            )
        else:
            downsample = nn.Sequential(
                nn.Conv3d(
                    inplanes,
                    planes * block_cls.expansion,
                    kernel_size=[1, 1, 1],
                    stride=[1, *stride],
                    padding=[0, 0, 0],
                    bias=False,
                ),
                nn.BatchNorm3d(planes * block_cls.expansion),
            )
    else:
        downsample = lambda x: x  # noqa: E731

    layers = [block_cls(inplanes, planes, stride=stride, downsample=downsample)]
    new_inplanes = planes * block_cls.expansion
    for _ in range(1, blocks_num):
        layers.append(block_cls(new_inplanes, planes, stride=[1, 1]))
    return nn.Sequential(*layers), new_inplanes


def _detect_layers_from_sd(sd: dict) -> list[int]:
    counts = []
    for li in (1, 2, 3, 4):
        prefix = f"layer{li}."
        idxs = set()
        for k in sd:
            if k.startswith(prefix):
                parts = k[len(prefix) :].split(".")
                if parts[0] == "forward_block" and len(parts) > 1 and parts[1].isdigit():
                    idxs.add(int(parts[1]))
                elif parts[0].isdigit():
                    idxs.add(int(parts[0]))
        counts.append(len(idxs) if idxs else 1)
    return counts


class DeepGaitV2Inference(nn.Module):
    """Loader CPU limpio para DeepGaitV2-P3D (1 canal silueta) y SkeletonGait (2 canales mapa)."""

    def __init__(self, ckpt_path: str | Path, use_emb2: bool = False):
        super().__init__()
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt)

        # Detección automática de hiperparámetros desde el checkpoint
        in_channels = int(sd["layer0.forward_block.0.weight"].shape[1])
        c0 = int(sd["layer0.forward_block.0.weight"].shape[0])
        channels = [c0, c0 * 2, c0 * 4, c0 * 8]
        layers = _detect_layers_from_sd(sd)
        class_num = int(sd["BNNecks.fc_bin"].shape[2])

        self.in_channels = in_channels
        self.use_emb2 = use_emb2

        self.layer0 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(in_channels, channels[0], 1),
                nn.BatchNorm2d(channels[0]),
                nn.ReLU(inplace=True),
            )
        )
        l1, inp = _make_p3d_layer(channels[0], BasicBlock2D, channels[0], [1, 1], layers[0], mode="2d")
        self.layer1 = SetBlockWrapper(l1)
        self.layer2, inp = _make_p3d_layer(inp, BasicBlockP3D, channels[1], [2, 2], layers[1], mode="p3d")
        self.layer3, inp = _make_p3d_layer(inp, BasicBlockP3D, channels[2], [2, 2], layers[2], mode="p3d")
        self.layer4, inp = _make_p3d_layer(inp, BasicBlockP3D, channels[3], [1, 1], layers[3], mode="p3d")

        self.FCs = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(16, channels[2], class_num=class_num)
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

        self.load_state_dict(sd, strict=True)
        self.eval()

    @torch.no_grad()
    def forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `[1, C_in, S, 64, 44]` en `[0, 1]` -> `[1, 4096]` (part-major)."""
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.max(out, dim=2)[0]  # [1, 512, H', W']
        feat = self.HPP(out)  # [1, 512, 16]
        embed_1 = self.FCs(feat)  # [1, 256, 16]
        if self.use_emb2:
            embed_2, _ = self.BNNecks(embed_1)
            emb = embed_2
        else:
            emb = embed_1
        return emb.transpose(1, 2).contiguous().reshape(1, -1)

    @torch.no_grad()
    def extract_embedding(self, seq: np.ndarray) -> np.ndarray | None:
        """`seq`: `(T, 64, 44)` para DeepGaitV2 o `(T, 2, 64, 44)` para SkeletonGait."""
        if seq is None or len(seq) < 5:
            return None
        if self.in_channels == 1:
            valid = seq.sum(axis=(1, 2)) > 0
            s = seq[valid]
            if len(s) < 5:
                return None
            # [1, 1, S, 64, 44]
            x = torch.from_numpy(s).to(torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        else:
            valid = seq.sum(axis=(1, 2, 3)) > 0
            s = seq[valid]
            if len(s) < 5:
                return None
            # (S, 2, 64, 44) -> [1, 2, S, 64, 44]
            x = torch.from_numpy(s).to(torch.float32).permute(1, 0, 2, 3).unsqueeze(0) / 255.0

        vec = self.forward_tensor(x).squeeze(0).cpu().numpy().astype(np.float32)
        if np.linalg.norm(vec) < 1e-8:
            return None
        return vec


class SkeletonGaitPPInference(nn.Module):
    """Loader CPU limpio para SkeletonGait++ (AttentionFusion temprana Silueta + Skeleton Maps)."""

    def __init__(self, ckpt_path: str | Path, use_emb2: bool = False):
        super().__init__()
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt)

        c0 = int(sd["sil_layer0.forward_block.0.weight"].shape[0])  # 32 * C (64 para C=2)
        c_mult = c0 // 32
        class_num = int(sd["BNNecks.fc_bin"].shape[2])

        # Contar bloques
        def _cnt(prefix: str) -> int:
            idxs = set()
            for k in sd:
                if k.startswith(prefix):
                    tok = k[len(prefix) :].split(".")[0]
                    if tok == "forward_block":
                        tok = k[len(prefix) :].split(".")[1]
                    if tok.isdigit():
                        idxs.add(int(tok))
            return len(idxs) if idxs else 1

        b0 = _cnt("sil_layer1.")
        b1 = _cnt("layer2.")
        b2 = _cnt("layer3.")
        b3 = _cnt("layer4.")

        self.use_emb2 = use_emb2
        self.sil_layer0 = SetBlockWrapper(
            nn.Sequential(conv3x3(1, c0, 1), nn.BatchNorm2d(c0), nn.ReLU(inplace=True))
        )
        self.map_layer0 = SetBlockWrapper(
            nn.Sequential(conv3x3(2, c0, 1), nn.BatchNorm2d(c0), nn.ReLU(inplace=True))
        )
        sil1, inp = _make_p3d_layer(c0, BasicBlock2D, c0, [1, 1], b0, mode="2d")
        self.sil_layer1 = SetBlockWrapper(sil1)
        self.map_layer1 = copy.deepcopy(self.sil_layer1)
        self.fusion = _AttentionFusion(c0)

        self.layer2, inp = _make_p3d_layer(inp, BasicBlockP3D, 64 * c_mult, [2, 2], b1, mode="p3d")
        self.layer3, inp = _make_p3d_layer(inp, BasicBlockP3D, 128 * c_mult, [2, 2], b2, mode="p3d")
        self.layer4, inp = _make_p3d_layer(inp, BasicBlockP3D, 256 * c_mult, [1, 1], b3, mode="p3d")

        self.FCs = SeparateFCs(16, 256 * c_mult, 128 * c_mult)
        self.BNNecks = SeparateBNNecks(16, 128 * c_mult, class_num=class_num)
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

        self.load_state_dict(sd, strict=True)
        self.eval()

    @torch.no_grad()
    def forward_tensors(self, sils: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        """`sils`: `[1, 1, S, 64, 44]`, `maps`: `[1, 2, S, 64, 44]` en `[0, 1]` -> `[1, 4096]`."""
        m0 = self.map_layer0(maps)
        m1 = self.map_layer1(m0)
        s0 = self.sil_layer0(sils)
        s1 = self.sil_layer1(s0)
        out = self.fusion(s1, m1)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.max(out, dim=2)[0]
        feat = self.HPP(out)
        embed_1 = self.FCs(feat)
        if self.use_emb2:
            embed_2, _ = self.BNNecks(embed_1)
            emb = embed_2
        else:
            emb = embed_1
        return emb.transpose(1, 2).contiguous().reshape(1, -1)

    @torch.no_grad()
    def extract_embedding(self, sil_seq: np.ndarray, map_seq: np.ndarray) -> np.ndarray | None:
        if sil_seq is None or map_seq is None:
            return None
        n = min(len(sil_seq), len(map_seq))
        if n < 5:
            return None
        sil_seq = sil_seq[:n]
        map_seq = map_seq[:n]
        valid = (sil_seq.sum(axis=(1, 2)) > 0) | (map_seq.sum(axis=(1, 2, 3)) > 0)
        s = sil_seq[valid]
        m = map_seq[valid]
        if len(s) < 5:
            return None
        s_t = torch.from_numpy(s).to(torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        m_t = torch.from_numpy(m).to(torch.float32).permute(1, 0, 2, 3).unsqueeze(0) / 255.0
        vec = self.forward_tensors(s_t, m_t).squeeze(0).cpu().numpy().astype(np.float32)
        if np.linalg.norm(vec) < 1e-8:
            return None
        return vec
