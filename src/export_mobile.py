"""Paso 1 (APK Offline): Exporta GaitBase, YOLOv8-seg/pose (512) y Galería a formato móvil.

Genera en `mobile_assets/`:
  - gaitbase_mobile.onnx  (~19 MB, sin el cabezal de 20,000 clases de GREW)
  - yolov8n-seg-512.onnx  (~13 MB)
  - yolov8n-pose-512.onnx (~13 MB)
  - gallery_index.json + gallery_vectors.bin (float32 compacto para Android)
"""

import json
import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from ultralytics import YOLO

from phase3_gait_engine import GaitBaseInference

CKPT_PATH = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\weights\GaitBase_DA-180000.pt")
GALLERY_PKL = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\outputs\gallery_db.pkl")
OUT_DIR = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\mobile_assets")


class GaitBaseMobileWrapper(nn.Module):
    """Wrapper limpio de GaitBase para exportar a ONNX con longitud temporal dinámica."""

    def __init__(self, gait_model: GaitBaseInference):
        super().__init__()
        self.Backbone = gait_model.Backbone
        self.HPP = gait_model.HPP
        self.FCs = gait_model.FCs

    def forward(self, sils: torch.Tensor) -> torch.Tensor:
        # sils: [1, 1, S, 64, 44] en [0, 1]
        outs = self.Backbone(sils)  # [1, 512, S, H', W']
        outs = torch.max(outs, dim=2)[0]  # Temporal Max Pooling -> [1, 512, H', W']
        feat = self.HPP(outs)  # [1, 512, 16]
        embed = self.FCs(feat)  # [1, 256, 16]
        vec = embed.reshape(1, -1)  # [1, 4096]
        return torch.nn.functional.normalize(vec, p=2, dim=1)


def export_gaitbase_onnx(out_path: Path):
    base = GaitBaseInference(CKPT_PATH)
    wrapper = GaitBaseMobileWrapper(base).eval()

    dummy = torch.rand(1, 1, 24, 64, 44, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["sil_seq"],
        output_names=["embedding"],
        dynamic_axes={"sil_seq": {2: "seq_len"}},
        opset_version=17,
        dynamo=False,
    )

    # Verificar equivalencia numérica con PyTorch usando una secuencia real
    sample_sil = np.load(
        r"outputs\features\subjects\Alejandro\toma_08\pass_01\frente\silhouettes.npy"
    )
    pt_emb = base.extract_embedding(sample_sil)

    valid = sample_sil.sum(axis=(1, 2)) > 0
    x_np = (sample_sil[valid].astype(np.float32) / 255.0)[None, None, :, :, :]
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_emb = sess.run(None, {"sil_seq": x_np})[0][0]
    cos_sim = float(np.dot(pt_emb, onnx_emb))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[OK] {out_path.name}: {size_mb:.2f} MB | cos_sim(PyTorch vs ONNX) = {cos_sim:.6f}")


def export_yolo_onnx():
    for name, pt_file, task in [
        ("yolov8n-seg-512.onnx", "yolov8n-seg.pt", "segment"),
        ("yolov8n-pose-512.onnx", "yolov8n-pose.pt", "pose"),
    ]:
        m = YOLO(pt_file)
        exported = Path(m.export(format="onnx", imgsz=512, opset=17))
        dst = OUT_DIR / name
        exported.replace(dst)
        print(f"[OK] {dst.name}: {dst.stat().st_size / (1024*1024):.2f} MB")


def export_gallery_binary():
    import pickle

    with open(GALLERY_PKL, "rb") as f:
        db = pickle.load(f)

    # Guardar vectores en binario contiguo float32 + índice JSON
    # Cada entrada: (subject, angle, has_sil, has_skel)
    entries_meta = []
    sil_vecs = []
    skel_vecs = []

    all_skel = [g["skel"] for entries in db.values() for g in entries if g.get("skel") is not None]
    skel_mean = np.mean(all_skel, axis=0).astype(np.float32)

    for subj in sorted(db.keys()):
        for g in db[subj]:
            if g.get("sil") is None:
                continue
            sil_idx = len(sil_vecs)
            sil_vecs.append(g["sil"].astype(np.float32))
            if g.get("skel") is not None:
                vc = g["skel"].astype(np.float32) - skel_mean
                n = np.linalg.norm(vc)
                vc = (vc / n) if n > 1e-8 else np.zeros(99, dtype=np.float32)
                skel_idx = len(skel_vecs)
                skel_vecs.append(vc.astype(np.float32))
            else:
                skel_idx = -1
            entries_meta.append(
                {"s": subj, "a": g["angle"], "si": sil_idx, "ki": skel_idx}
            )

    sil_mat = np.stack(sil_vecs, axis=0).astype(np.float32)  # (N, 4096)
    skel_mat = np.stack(skel_vecs, axis=0).astype(np.float32)  # (M, 99)

    (OUT_DIR / "gallery_sil.bin").write_bytes(sil_mat.tobytes())
    (OUT_DIR / "gallery_skel.bin").write_bytes(skel_mat.tobytes())
    (OUT_DIR / "gallery_skel_mean.bin").write_bytes(skel_mean.tobytes())
    (OUT_DIR / "gallery_index.json").write_text(
        json.dumps(
            {
                "num_subjects": len(db),
                "num_entries": len(entries_meta),
                "sil_dim": 4096,
                "skel_dim": 99,
                "entries": entries_meta,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    total_mb = sum(
        (OUT_DIR / fn).stat().st_size
        for fn in ("gallery_sil.bin", "gallery_skel.bin", "gallery_skel_mean.bin", "gallery_index.json")
    ) / (1024 * 1024)
    print(f"[OK] Galería móvil exportada: {len(entries_meta)} entradas, {total_mb:.2f} MB")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export_gaitbase_onnx(OUT_DIR / "gaitbase_mobile.onnx")
    export_yolo_onnx()
    export_gallery_binary()

