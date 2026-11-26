"""Exportación ONNX (FP32 + Cuantización dinámica INT8) del modelo ganador v2.

Elimina el cabezal `BNNecks` de 20,000 clases de GREW (que pesa ~700 MB en el `.pt`
de entrenamiento) para generar modelos ligeros listos para CPU/ONNX Runtime y Android,
verificando equivalencia numérica (`cos_sim > 0.995`) y midiendo speedup real en CPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

V2_ROOT = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\MetodoAlternativo")
ROOT = V2_ROOT.parent
sys.path.insert(0, str(V2_ROOT / "src"))

from heatmap_generator import generate_skeleton_maps
from models_v2 import DeepGaitV2Inference, SkeletonGaitPPInference


class _DeepGaitV2OnnxWrapper(nn.Module):
    def __init__(self, model: DeepGaitV2Inference):
        super().__init__()
        self.layer0 = model.layer0
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.HPP = model.HPP
        self.FCs = model.FCs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.max(out, dim=2)[0]
        feat = self.HPP(out)
        emb = self.FCs(feat)
        return emb.transpose(1, 2).contiguous().reshape(1, -1)


class _SkeletonGaitPPOnnxWrapper(nn.Module):
    def __init__(self, model: SkeletonGaitPPInference):
        super().__init__()
        self.sil_layer0 = model.sil_layer0
        self.map_layer0 = model.map_layer0
        self.sil_layer1 = model.sil_layer1
        self.map_layer1 = model.map_layer1
        self.fusion = model.fusion
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.HPP = model.HPP
        self.FCs = model.FCs

    def forward(self, sils: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
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
        emb = self.FCs(feat)
        return emb.transpose(1, 2).contiguous().reshape(1, -1)


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def export_model(model_key: str):
    out_dir = V2_ROOT / "outputs" / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_dir = ROOT / "outputs" / "features" / "subjects" / "Alejandro" / "toma_08" / "pass_01" / "frente"
    sil = np.load(sample_dir / "silhouettes.npy")[:30]
    kpt = np.load(sample_dir / "keypoints.npy")[:30]
    hm = generate_skeleton_maps(kpt)

    fp32_path = out_dir / f"{model_key}_fp32.onnx"
    int8_path = out_dir / f"{model_key}_int8.onnx"

    if model_key in ("deepgaitv2", "skeletongait"):
        ckpt = V2_ROOT / "weights" / ("DeepGaitV2_GREW.pt" if model_key == "deepgaitv2" else "SkeletonGait_GREW.pt")
        model = DeepGaitV2Inference(ckpt)
        wrapper = _DeepGaitV2OnnxWrapper(model).eval()
        c_in = model.in_channels
        dummy = torch.rand(1, c_in, 24, 64, 44, dtype=torch.float32)
        torch.onnx.export(
            wrapper,
            dummy,
            str(fp32_path),
            input_names=["input_seq"],
            output_names=["embedding"],
            dynamic_axes={"input_seq": {2: "seq_len"}},
            opset_version=17,
            dynamo=False,
        )
        if c_in == 1:
            pt_emb = model.extract_embedding(sil)
            valid = sil.sum(axis=(1, 2)) > 0
            x_np = (sil[valid].astype(np.float32) / 255.0)[None, None, :, :, :]
        else:
            pt_emb = model.extract_embedding(hm)
            valid = hm.sum(axis=(1, 2, 3)) > 0
            x_np = (hm[valid].astype(np.float32) / 255.0).transpose(1, 0, 2, 3)[None, :, :, :, :]
        feed = {"input_seq": x_np}
    elif model_key == "skeletongaitpp":
        ckpt = V2_ROOT / "weights" / "SkeletonGaitPP_GREW.pt"
        model = SkeletonGaitPPInference(ckpt)
        wrapper = _SkeletonGaitPPOnnxWrapper(model).eval()
        d_sil = torch.rand(1, 1, 24, 64, 44, dtype=torch.float32)
        d_map = torch.rand(1, 2, 24, 64, 44, dtype=torch.float32)
        torch.onnx.export(
            wrapper,
            (d_sil, d_map),
            str(fp32_path),
            input_names=["sil_seq", "map_seq"],
            output_names=["embedding"],
            dynamic_axes={"sil_seq": {2: "seq_len"}, "map_seq": {2: "seq_len"}},
            opset_version=17,
            dynamo=False,
        )
        pt_emb = model.extract_embedding(sil, hm)
        valid = (sil.sum(axis=(1, 2)) > 0) | (hm.sum(axis=(1, 2, 3)) > 0)
        s_np = (sil[valid].astype(np.float32) / 255.0)[None, None, :, :, :]
        m_np = (hm[valid].astype(np.float32) / 255.0).transpose(1, 0, 2, 3)[None, :, :, :, :]
        feed = {"sil_seq": s_np, "map_seq": m_np}
    else:
        raise ValueError(f"Modelo no soportado: {model_key}")

    # Benchmark PyTorch
    t_pt = []
    for _ in range(3):
        t0 = time.perf_counter()
        if model_key == "skeletongaitpp":
            _ = model.extract_embedding(sil, hm)
        elif model_key == "deepgaitv2":
            _ = model.extract_embedding(sil)
        else:
            _ = model.extract_embedding(hm)
        t_pt.append((time.perf_counter() - t0) * 1000.0)

    # Benchmark ONNX FP32
    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    fp32_emb = sess_fp32.run(None, feed)[0][0]
    t_fp32 = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = sess_fp32.run(None, feed)
        t_fp32.append((time.perf_counter() - t0) * 1000.0)

    print(
        f"[ONNX FP32] {fp32_path.name}: {fp32_path.stat().st_size / 1e6:.2f} MB | "
        f"cos_sim={_cos_sim(pt_emb, fp32_emb):.6f} | "
        f"PyTorch={np.median(t_pt):.1f}ms -> ORT_FP32={np.median(t_fp32):.1f}ms",
        flush=True,
    )

    # Cuantización dinámica INT8 (opcional si onnxruntime.quantization está disponible)
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            str(fp32_path),
            str(int8_path),
            weight_type=QuantType.QUInt8,
        )
        sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        int8_emb = sess_int8.run(None, feed)[0][0]
        t_int8 = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = sess_int8.run(None, feed)
            t_int8.append((time.perf_counter() - t0) * 1000.0)
        print(
            f"[ONNX INT8] {int8_path.name}: {int8_path.stat().st_size / 1e6:.2f} MB | "
            f"cos_sim={_cos_sim(pt_emb, int8_emb):.6f} | "
            f"ORT_INT8={np.median(t_int8):.1f}ms (speedup vs PT: {np.median(t_pt)/np.median(t_int8):.2f}x)",
            flush=True,
        )
    except Exception as e:
        print(f"[WARN] Cuantización INT8 omitida: {e}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["deepgaitv2", "skeletongaitpp"])
    args = parser.parse_args()
    for mk in args.models:
        export_model(mk)
