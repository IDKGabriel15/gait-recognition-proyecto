"""Shared part-wise centering and normalization for GaitBase silhouette features."""

import numpy as np

PARTS = 16
CHANNELS = 256
EMBEDDING_DIM = PARTS * CHANNELS
EPSILON = 1e-8


def _parts(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.shape[-1] != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM}-D embeddings, got shape {values.shape}")
    return values.reshape(*values.shape[:-1], PARTS, CHANNELS)


def center_and_normalize_embedding(
    embedding: np.ndarray, mean: np.ndarray
) -> np.ndarray:
    """Subtract a gallery mean and L2-normalize each of the 16 horizontal parts."""
    parts = _parts(embedding)
    mean_parts = _parts(np.asarray(mean, dtype=np.float32).reshape(-1))
    if parts.ndim != 2 or mean_parts.shape != (PARTS, CHANNELS):
        raise ValueError("Expected one 4096-D embedding and one 4096-D gallery mean")
    centered = parts - mean_parts
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = np.divide(
        centered,
        norms,
        out=np.zeros_like(centered),
        where=norms > EPSILON,
    )
    return normalized.reshape(EMBEDDING_DIM)


def center_and_normalize_gallery(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the per-part gallery mean and normalize every centered entry."""
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Expected a non-empty matrix of gallery embeddings")
    parts = _parts(values)
    mean = parts.mean(axis=0)
    centered = parts - mean
    norms = np.linalg.norm(centered, axis=2, keepdims=True)
    normalized = np.divide(
        centered,
        norms,
        out=np.zeros_like(centered),
        where=norms > EPSILON,
    )
    return normalized.reshape(-1, EMBEDDING_DIM), mean.reshape(EMBEDDING_DIM)
