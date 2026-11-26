"""Exporta los assets móviles 100% offline para `MetodoAlternativo/android_app_v2/`.

Genera en `MetodoAlternativo/android_app_v2/app/src/main/assets/`:
  - deepgaitv2_fp32.onnx   (~52.9 MB, cos_sim=1.0000 vs PyTorch)
  - yolov8n-seg-512.onnx   (copia desde mobile_assets/ sin tocar Plan 1)
  - yolov8n-pose-512.onnx  (copia desde mobile_assets/ sin tocar Plan 1)
  - pca_v2_means.bin       (16 * 256 float32 = 16 KB)
  - pca_v2_proj.bin        (16 * 256 * 256 float32 = 4.0 MB)
  - gallery_v2_sil.bin     (N * 4096 float32, YA transformados con PCA + centrados + L2 por parte)
  - gallery_v2_sil_mean.bin (4096 float32, media post-PCA por parte)
  - gallery_v2_index.json  (metadatos compactos)
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA")
V2_ROOT = ROOT / "MetodoAlternativo"
OUT_V2 = V2_ROOT / "outputs"
ASSETS_V2 = V2_ROOT / "android_app_v2" / "app" / "src" / "main" / "assets"

sys.path.insert(0, str(V2_ROOT / "src"))
from domain_adapter import PartWisePCAWhitener
from matcher_v2 import center_and_l2_parts


def main():
    ASSETS_V2.mkdir(parents=True, exist_ok=True)

    # 1. Copiar ONNX (DeepGaitV2 FP32 + YOLOv8 seg/pose 512)
    src_dg = OUT_V2 / "onnx" / "deepgaitv2_fp32.onnx"
    if not src_dg.exists():
        raise FileNotFoundError(f"Falta {src_dg}. Ejecuta primero export_v2_onnx.py.")
    shutil.copy2(src_dg, ASSETS_V2 / "deepgaitv2_fp32.onnx")

    for yolo_name in ("yolov8n-seg-512.onnx", "yolov8n-pose-512.onnx"):
        src_y = ROOT / "android_app" / "app" / "src" / "main" / "assets" / yolo_name
        if not src_y.exists():
            src_y = ROOT / "mobile_assets" / yolo_name
        shutil.copy2(src_y, ASSETS_V2 / yolo_name)

    # 2. Cargar cache_deepgaitv2.pkl (solo split == gallery)
    with open(OUT_V2 / "cache_deepgaitv2.pkl", "rb") as f:
        payload = pickle.load(f)

    gal_entries = [
        r for r in payload["records"]
        if r["split"] == "gallery" and r["emb"] is not None
    ]
    if not gal_entries:
        raise RuntimeError("No hay entradas de galería en cache_deepgaitv2.pkl")

    raw_mat = np.stack([r["emb"].astype(np.float32) for r in gal_entries], axis=0)  # (N, 4096)

    # 3. Ajustar PartWisePCAWhitener (whiten_power=0.5) sobre la galería
    whitener = PartWisePCAWhitener(parts=16, channels=256, whiten_power=0.5).fit(raw_mat)
    pca_means = np.stack(whitener.means_, axis=0).astype(np.float32)  # (16, 256)
    pca_proj = np.stack(whitener.proj_, axis=0).astype(np.float32)    # (16, 256, 256)

    # Transformar toda la galería con PCA-Whitening y luego centrar + L2 por parte
    trans_mat = np.stack([whitener.transform_one(v) for v in raw_mat], axis=0)  # (N, 4096)
    norm_parts, mean_parts = center_and_l2_parts(trans_mat, mean_parts=None, parts=16, channels=256)
    sil_mat = norm_parts.reshape(-1, 4096).astype(np.float32)
    sil_mean = mean_parts.reshape(4096).astype(np.float32)

    entries_meta = [
        {"s": r["subject"], "a": r["angle"], "si": idx}
        for idx, r in enumerate(gal_entries)
    ]

    # 4. Escribir binarios little-endian float32 + JSON
    (ASSETS_V2 / "pca_v2_means.bin").write_bytes(pca_means.tobytes())
    (ASSETS_V2 / "pca_v2_proj.bin").write_bytes(pca_proj.tobytes())
    (ASSETS_V2 / "gallery_v2_sil.bin").write_bytes(sil_mat.tobytes())
    (ASSETS_V2 / "gallery_v2_sil_mean.bin").write_bytes(sil_mean.tobytes())
    (ASSETS_V2 / "gallery_v2_index.json").write_text(
        json.dumps(
            {
                "version": "deepgaitv2-pca-v2",
                "num_subjects": len({r["subject"] for r in gal_entries}),
                "num_entries": len(entries_meta),
                "parts": 16,
                "channels": 256,
                "top_k_parts": 10,
                "top_k_gallery": 3,
                "entries": entries_meta,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    total_mb = sum(p.stat().st_size for p in ASSETS_V2.iterdir() if p.is_file()) / (1024 * 1024)
    print(f"[OK] Assets móviles v2 exportados en {ASSETS_V2}:")
    for p in sorted(ASSETS_V2.iterdir()):
        print(f"  - {p.name:28s}: {p.stat().st_size / (1024 * 1024):6.2f} MB")
    print(f"  TOTAL assets v2: {total_mb:.2f} MB ({len(entries_meta)} entradas)")


if __name__ == "__main__":
    main()
