"""Adaptación ligera al dominio sin sobreajuste (PCA-Whitening por franja).

Por qué PCA-Whitening por franja (y NO sobreentrenar un clasificador a 31 sujetos):
  - Es 100% **no supervisado** respecto a las identidades de prueba (cero fuga de datos).
  - Elimina direcciones dominantes de cámara/iluminación/segmentador YOLO comunes a
    toda la galería y ecualiza la varianza de las dimensiones discriminativas.
  - Se ajusta en < 0.2 segundos en CPU y generaliza a sujetos nunca vistos (Open-Set).
"""

from __future__ import annotations

import numpy as np


class PartWisePCAWhitener:
    """Aplica PCA + Whitening suave (`power` en [0, 1]) independiente por cada franja."""

    def __init__(
        self,
        parts: int = 16,
        channels: int = 256,
        out_dim: int | None = None,
        whiten_power: float = 0.5,
        reg_eps: float = 1e-4,
    ):
        self.parts = parts
        self.channels = channels
        self.out_dim = out_dim or channels
        self.whiten_power = whiten_power
        self.reg_eps = reg_eps
        self.means_: list[np.ndarray] = []
        self.proj_: list[np.ndarray] = []

    def fit(self, embeddings: np.ndarray) -> "PartWisePCAWhitener":
        """`embeddings`: `(N, P*C)`."""
        arr = np.asarray(embeddings, dtype=np.float32).reshape(-1, self.parts, self.channels)
        n = arr.shape[0]
        self.means_.clear()
        self.proj_.clear()

        for p in range(self.parts):
            xp = arr[:, p, :]  # (N, C)
            mu = xp.mean(axis=0)
            xc = xp - mu
            # SVD económica
            _, s, vt = np.linalg.svd(xc, full_matrices=False)
            eigvals = (s ** 2) / max(n - 1, 1)
            k = min(self.out_dim, vt.shape[0])
            comps = vt[:k]  # (k, C)
            scale = (eigvals[:k] + self.reg_eps) ** (-0.5 * self.whiten_power)
            proj = (comps * scale[:, None]).T.astype(np.float32)  # (C, k)
            # Rellenar con ceros si k < out_dim (n < C)
            if k < self.out_dim:
                pad = np.zeros((self.channels, self.out_dim - k), dtype=np.float32)
                proj = np.concatenate([proj, pad], axis=1)
            self.means_.append(mu.astype(np.float32))
            self.proj_.append(proj)
        return self

    def transform_one(self, vec: np.ndarray) -> np.ndarray:
        parts_in = np.asarray(vec, dtype=np.float32).reshape(self.parts, self.channels)
        parts_out = np.empty((self.parts, self.out_dim), dtype=np.float32)
        for p in range(self.parts):
            parts_out[p] = (parts_in[p] - self.means_[p]) @ self.proj_[p]
        return parts_out.reshape(-1)
