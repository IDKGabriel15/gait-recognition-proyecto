"""Benchmark Científico A/B: Plan 1 vs. Método Alternativo (Plan 2).

Evalúa con partición estricta idéntica (`toma_01..07` Gallery vs `toma_08+` Probe):
  1. Multi-cámara y 1 cámara mismo ángulo: `all_available`, `only_frente`, `only_45_izq`, `only_45_der`
  2. Cross-View estricto (ángulo excluido de galería): `cross_frente`, `cross_45_izq`, `cross_45_der`
  3. Cambio de ropa externo (`*Prueba`): 3 cámaras y 1 cámara (`only_frente`)
  4. Open-Set riguroso (Leave-5-Subjects-Out): ROC-AUC, EER y TAR@FAR=0% / TAR@FAR=1%
  5. Latencia real en CPU: extracción (ms / 30 frames) y matching (ms / consulta)

Cachea embeddings extraídos por modelo en `MetodoAlternativo/outputs/cache_<model>.pkl`
para iterar instantáneamente sobre matching (Top-K, TTA Flip, AS-Norm, PCA-Whitening).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA")
V2_ROOT = ROOT / "MetodoAlternativo"
MANIFEST_PATH = ROOT / "configs" / "dataset_manifest.json"
FEATURES_ROOT = ROOT / "outputs" / "features"
WEIGHTS_V2 = V2_ROOT / "weights"
OUT_V2 = V2_ROOT / "outputs"

sys.path.insert(0, str(V2_ROOT / "src"))
sys.path.insert(0, str(ROOT / "src"))

from domain_adapter import PartWisePCAWhitener
from heatmap_generator import flip_sil_and_maps, generate_skeleton_maps
from matcher_v2 import VectorizedGaitMatcher
from models_v2 import DeepGaitV2Inference, SkeletonGaitPPInference
from quality_and_motion import filter_sequence_quality


def _iter_manifest_clips(manifest: dict):
    """Itera todos los clips (gallery, probe, external_test) disponibles en `outputs/features/`."""
    items = []
    for subj, sdata in manifest["subjects"].items():
        for split in ("gallery", "probe"):
            for take in sdata[split]:
                take_dir = FEATURES_ROOT / "subjects" / subj / take["take_name"]
                if not take_dir.exists():
                    continue
                for pass_dir in sorted(take_dir.iterdir()):
                    if not pass_dir.is_dir() or not pass_dir.name.startswith("pass_"):
                        continue
                    for ang_dir in sorted(pass_dir.iterdir()):
                        if not ang_dir.is_dir():
                            continue
                        sil_p = ang_dir / "silhouettes.npy"
                        kpt_p = ang_dir / "keypoints.npy"
                        if sil_p.exists() and kpt_p.exists():
                            items.append(
                                {
                                    "split": split,
                                    "subject": subj,
                                    "take": take["take_name"],
                                    "pass": pass_dir.name,
                                    "angle": ang_dir.name,
                                    "sil_path": sil_p,
                                    "kpt_path": kpt_p,
                                }
                            )
    for ext_subj, takes in manifest["external_test"].items():
        for take in takes:
            take_dir = FEATURES_ROOT / "external_test" / ext_subj / take["take_name"]
            if not take_dir.exists():
                continue
            for pass_dir in sorted(take_dir.iterdir()):
                if not pass_dir.is_dir() or not pass_dir.name.startswith("pass_"):
                    continue
                for ang_dir in sorted(pass_dir.iterdir()):
                    if not ang_dir.is_dir():
                        continue
                    sil_p = ang_dir / "silhouettes.npy"
                    kpt_p = ang_dir / "keypoints.npy"
                    if sil_p.exists() and kpt_p.exists():
                        items.append(
                            {
                                "split": "external",
                                "subject": ext_subj,
                                "take": take["take_name"],
                                "pass": pass_dir.name,
                                "angle": ang_dir.name,
                                "sil_path": sil_p,
                                "kpt_path": kpt_p,
                            }
                        )
    return items


def _uniform_subsample(arr: np.ndarray, max_frames: int) -> np.ndarray:
    if len(arr) <= max_frames:
        return arr
    idx = np.linspace(0, len(arr) - 1, max_frames).round().astype(np.int64)
    return arr[idx]


def build_or_load_embeddings(
    model_key: str,
    manifest: dict,
    use_quality_gate: bool = True,
    max_frames: int = 24,
    overwrite: bool = False,
) -> dict:
    """Extrae y cachea embeddings (original + TTA horizontal flip) para un modelo.

    Usa ONNX Runtime FP32 automáticamente si el archivo exportado existe en
    `MetodoAlternativo/outputs/onnx/<model_key>_fp32.onnx` para máxima velocidad en CPU.
    """
    import onnxruntime as ort

    OUT_V2.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_V2 / f"cache_{model_key}.pkl"
    if cache_path.exists() and not overwrite:
        print(f"[CACHE] Cargando {cache_path.name}...", flush=True)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    onnx_fp32 = OUT_V2 / "onnx" / f"{model_key}_fp32.onnx"
    ort_sess = None
    model = None
    if onnx_fp32.exists() and model_key in ("deepgaitv2", "skeletongait", "skeletongaitpp"):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 10
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_sess = ort.InferenceSession(str(onnx_fp32), sess_options=so, providers=["CPUExecutionProvider"])
        print(f"\n[EXTRACT-ORT] Extrayendo embeddings para '{model_key}' vía {onnx_fp32.name} (max_frames={max_frames})...", flush=True)
    else:
        print(f"\n[EXTRACT-PT] Extrayendo embeddings para '{model_key}' (max_frames={max_frames})...", flush=True)
        if model_key == "deepgaitv2":
            model = DeepGaitV2Inference(WEIGHTS_V2 / "DeepGaitV2_GREW.pt")
        elif model_key == "skeletongait":
            model = DeepGaitV2Inference(WEIGHTS_V2 / "SkeletonGait_GREW.pt")
        elif model_key == "skeletongaitpp":
            model = SkeletonGaitPPInference(WEIGHTS_V2 / "SkeletonGaitPP_GREW.pt")
        elif model_key == "gaitbase_p1":
            from phase3_gait_engine import GaitBaseInference
            model = GaitBaseInference(ROOT / "weights" / "GaitBase_DA-180000.pt")
        else:
            raise ValueError(f"Modelo desconocido: {model_key}")

    def _run_one(sil_s: np.ndarray | None, hm_s: np.ndarray | None) -> np.ndarray | None:
        if ort_sess is not None:
            if model_key == "deepgaitv2":
                valid = sil_s.sum(axis=(1, 2)) > 0
                s = sil_s[valid]
                if len(s) < 5:
                    return None
                x = (s.astype(np.float32) / 255.0)[None, None, :, :, :]
                vec = ort_sess.run(None, {"input_seq": x})[0][0].astype(np.float32)
            elif model_key == "skeletongait":
                valid = hm_s.sum(axis=(1, 2, 3)) > 0
                h = hm_s[valid]
                if len(h) < 5:
                    return None
                x = (h.astype(np.float32) / 255.0).transpose(1, 0, 2, 3)[None, :, :, :, :]
                vec = ort_sess.run(None, {"input_seq": x})[0][0].astype(np.float32)
            else:
                n = min(len(sil_s), len(hm_s))
                sil_s, hm_s = sil_s[:n], hm_s[:n]
                valid = (sil_s.sum(axis=(1, 2)) > 0) | (hm_s.sum(axis=(1, 2, 3)) > 0)
                s, h = sil_s[valid], hm_s[valid]
                if len(s) < 5:
                    return None
                xs = (s.astype(np.float32) / 255.0)[None, None, :, :, :]
                xh = (h.astype(np.float32) / 255.0).transpose(1, 0, 2, 3)[None, :, :, :, :]
                vec = ort_sess.run(None, {"sil_seq": xs, "map_seq": xh})[0][0].astype(np.float32)
            return vec if np.linalg.norm(vec) >= 1e-8 else None
        else:
            if model_key in ("deepgaitv2", "gaitbase_p1"):
                return model.extract_embedding(sil_s)
            elif model_key == "skeletongait":
                return model.extract_embedding(hm_s)
            else:
                return model.extract_embedding(sil_s, hm_s)

    items = _iter_manifest_clips(manifest)
    t_start = time.perf_counter()
    latencies_30f = []
    records = []

    for idx, it in enumerate(items):
        sil = np.load(it["sil_path"])
        kpt = np.load(it["kpt_path"])
        if use_quality_gate:
            sil_q, kpt_q, _ = filter_sequence_quality(sil, kpt, min_valid_frames=10)
        else:
            sil_q, kpt_q = sil, kpt

        sil_q = _uniform_subsample(sil_q, max_frames)
        kpt_q = _uniform_subsample(kpt_q, max_frames)

        need_flip = it["split"] != "gallery"
        t0 = time.perf_counter()
        if model_key in ("deepgaitv2", "gaitbase_p1"):
            emb = _run_one(sil_q, None)
            if need_flip:
                sil_f, _ = flip_sil_and_maps(sil_q, None)
                emb_flip = _run_one(sil_f, None) if sil_f is not None else None
            else:
                emb_flip = None
        elif model_key == "skeletongait":
            hm = generate_skeleton_maps(kpt_q)
            emb = _run_one(None, hm)
            if need_flip:
                _, hm_f = flip_sil_and_maps(None, hm)
                emb_flip = _run_one(None, hm_f) if hm_f is not None else None
            else:
                emb_flip = None
        elif model_key == "skeletongaitpp":
            hm = generate_skeleton_maps(kpt_q)
            emb = _run_one(sil_q, hm)
            if need_flip:
                sil_f, hm_f = flip_sil_and_maps(sil_q, hm)
                emb_flip = _run_one(sil_f, hm_f)
            else:
                emb_flip = None
        else:
            emb, emb_flip = None, None
        dt = (time.perf_counter() - t0) * 1000.0
        passes_run = 2.0 if need_flip else 1.0
        if len(sil_q) >= 15 and len(latencies_30f) < 60:
            latencies_30f.append((dt / passes_run) * (30.0 / max(len(sil_q), 1)))

        records.append(
            {
                "split": it["split"],
                "subject": it["subject"],
                "take": it["take"],
                "pass": it["pass"],
                "angle": it["angle"],
                "emb": emb,
                "emb_flip": emb_flip,
            }
        )
        if (idx + 1) % 300 == 0 or (idx + 1) == len(items):
            el = time.perf_counter() - t_start
            print(f"  {model_key}: {idx + 1}/{len(items)} clips ({el:.1f}s)", flush=True)

    payload = {
        "model_key": model_key,
        "records": records,
        "avg_ms_30f": float(np.median(latencies_30f)) if latencies_30f else 0.0,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
    print(
        f"[OK] {model_key} cacheado en {cache_path.name} "
        f"(mediana CPU ~{payload['avg_ms_30f']:.1f} ms / 30 frames)",
        flush=True,
    )
    return payload


def _group_passes(records: list[dict], split: str) -> list[dict]:
    grouped = defaultdict(dict)
    for r in records:
        if r["split"] != split:
            continue
        key = (r["subject"], r["take"], r["pass"])
        grouped[key][r["angle"]] = (r["emb"], r["emb_flip"])
    out = []
    for (subj, take, p_name), views in sorted(grouped.items()):
        out.append({"subject": subj, "take": take, "pass": p_name, "views": views})
    return out


def evaluate_config(
    payload: dict,
    top_k_parts: int = 10,
    top_k_gallery: int = 3,
    use_tta: bool = True,
    use_as_norm: bool = False,
    use_whitening: bool = False,
    whiten_power: float = 0.5,
) -> dict:
    """Evalúa un modelo + configuración de matching en todos los escenarios + Open-Set."""
    records = payload["records"]
    gal_entries = []
    for r in records:
        if r["split"] != "gallery" or r["emb"] is None:
            continue
        gal_entries.append(
            {
                "subject": r["subject"],
                "angle": r["angle"],
                "emb": r["emb"],
                "emb_flip": r["emb_flip"] if use_tta else None,
            }
        )

    adapter = None
    if use_whitening:
        gal_mat = np.stack([e["emb"] for e in gal_entries], axis=0)
        adapter = PartWisePCAWhitener(parts=16, channels=256, whiten_power=whiten_power).fit(gal_mat)

    matcher = VectorizedGaitMatcher(
        gal_entries,
        top_k_parts=top_k_parts,
        top_k_gallery=top_k_gallery,
        use_as_norm=use_as_norm,
        adapter=adapter,
    )

    probes = _group_passes(records, "probe")
    scenarios = [
        ("all_available", None, False),
        ("only_frente", "frente", False),
        ("only_45_izq", "45_izq", False),
        ("only_45_der", "45_der", False),
        ("cross_frente", "frente", True),
        ("cross_45_izq", "45_izq", True),
        ("cross_45_der", "45_der", True),
    ]

    scen_results = {}
    match_times = []
    for scen_name, filter_ang, excl_same in scenarios:
        top1, top5, total = 0, 0, 0
        for p in probes:
            views = p["views"]
            if filter_ang is not None:
                if filter_ang not in views:
                    continue
                views = {filter_ang: views[filter_ang]}
            if not use_tta:
                views = {a: (e, None) for a, (e, _) in views.items()}
            t0 = time.perf_counter()
            ranked = matcher.match_views(views, exclude_same_angle=excl_same)
            match_times.append((time.perf_counter() - t0) * 1000.0)
            if not ranked:
                continue
            total += 1
            preds = [x[0] for x in ranked[:5]]
            if preds[0] == p["subject"]:
                top1 += 1
            if p["subject"] in preds:
                top5 += 1
        scen_results[scen_name] = {
            "top1": 100.0 * top1 / max(total, 1),
            "top5": 100.0 * top5 / max(total, 1),
            "correct1": top1,
            "total": total,
        }

    # *Prueba (cambio de ropa externo)
    ext_map = {
        "AlejandroPrueba": "Alejandro",
        "DanielPrueba": "DanielOntiveros",
        "JoelPrueba": "JoelFelix",
    }
    externals = _group_passes(records, "external")
    ext_all_ok, ext_all_tot = 0, 0
    ext_1cam_ok, ext_1cam_tot = 0, 0
    for p in externals:
        gt = ext_map.get(p["subject"])
        if not gt:
            continue
        views = p["views"] if use_tta else {a: (e, None) for a, (e, _) in p["views"].items()}
        r_all = matcher.match_views(views)
        if r_all:
            ext_all_tot += 1
            if r_all[0][0] == gt:
                ext_all_ok += 1
        if "frente" in views:
            r_1 = matcher.match_views({"frente": views["frente"]})
            if r_1:
                ext_1cam_tot += 1
                if r_1[0][0] == gt:
                    ext_1cam_ok += 1

    # Open-Set: Leave-5-Subjects-Out (3 folds determinísticos -> 15 sujetos desconocidos evaluados)
    all_subjs = sorted({e["subject"] for e in gal_entries})
    gen_scores, imp_scores = [], []
    for fold_start in (0, 10, 20):
        held_out = set(all_subjs[fold_start : fold_start + 5])
        fold_gal = [e for e in gal_entries if e["subject"] not in held_out]
        fold_adapter = None
        if use_whitening:
            fm = np.stack([e["emb"] for e in fold_gal], axis=0)
            fold_adapter = PartWisePCAWhitener(parts=16, channels=256, whiten_power=whiten_power).fit(fm)
        fold_matcher = VectorizedGaitMatcher(
            fold_gal,
            top_k_parts=top_k_parts,
            top_k_gallery=top_k_gallery,
            use_as_norm=use_as_norm,
            adapter=fold_adapter,
        )
        for p in probes:
            views = p["views"] if use_tta else {a: (e, None) for a, (e, _) in p["views"].items()}
            # Evaluar en 1 cámara (`frente`, el caso más exigente para Open-Set)
            if "frente" not in views:
                continue
            ranked = fold_matcher.match_views({"frente": views["frente"]})
            if not ranked:
                continue
            sc = ranked[0][1]
            if p["subject"] in held_out:
                imp_scores.append(sc)
            else:
                gen_scores.append(sc)

    y_true = np.array([1] * len(gen_scores) + [0] * len(imp_scores))
    y_score = np.array(gen_scores + imp_scores, dtype=np.float32)
    auc = float(roc_auc_score(y_true, y_score)) if len(imp_scores) > 0 else 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float(0.5 * (fpr[eer_idx] + fnr[eer_idx]) * 100.0)
    tar_at_far1 = float(tpr[np.searchsorted(fpr, 0.01, side="right") - 1] * 100.0)

    return {
        "scenarios": scen_results,
        "ext_3cam": (ext_all_ok, ext_all_tot),
        "ext_1cam": (ext_1cam_ok, ext_1cam_tot),
        "openset_auc": auc,
        "openset_eer": eer,
        "openset_tar_far1": tar_at_far1,
        "avg_match_ms": float(np.median(match_times)) if match_times else 0.0,
        "avg_inf_ms_30f": payload["avg_ms_30f"],
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark A/B Plan 1 vs Plan 2")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gaitbase_p1", "deepgaitv2", "skeletongait", "skeletongaitpp"],
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    payloads = {}
    for mk in args.models:
        payloads[mk] = build_or_load_embeddings(mk, manifest, use_quality_gate=True, overwrite=args.overwrite)

    # Configuraciones a comparar
    experiments = []
    if "gaitbase_p1" in payloads:
        experiments.append(("P1: GaitBase (Top8, sin TTA)", "gaitbase_p1", dict(top_k_parts=8, top_k_gallery=1, use_tta=False, use_as_norm=False, use_whitening=False)))
        experiments.append(("P1+: GaitBase + MatcherV2 (TTA+ASNorm)", "gaitbase_p1", dict(top_k_parts=10, top_k_gallery=3, use_tta=True, use_as_norm=True, use_whitening=False)))
    if "deepgaitv2" in payloads:
        experiments.append(("P2A: DeepGaitV2-P3D (Base)", "deepgaitv2", dict(top_k_parts=10, top_k_gallery=3, use_tta=False, use_as_norm=False, use_whitening=False)))
        experiments.append(("P2A+: DeepGaitV2-P3D (TTA+ASNorm)", "deepgaitv2", dict(top_k_parts=10, top_k_gallery=3, use_tta=True, use_as_norm=True, use_whitening=False)))
        experiments.append(("P2A++: DeepGaitV2-P3D (TTA+ASNorm+PCA)", "deepgaitv2", dict(top_k_parts=10, top_k_gallery=3, use_tta=True, use_as_norm=True, use_whitening=True)))
    if "skeletongait" in payloads:
        experiments.append(("P2B: SkeletonGait-P3D (Solo Esqueleto)", "skeletongait", dict(top_k_parts=12, top_k_gallery=3, use_tta=True, use_as_norm=True, use_whitening=False)))
    if "skeletongaitpp" in payloads:
        experiments.append(("P2C: SkeletonGait++ (Base)", "skeletongaitpp", dict(top_k_parts=10, top_k_gallery=3, use_tta=False, use_as_norm=False, use_whitening=False)))
        experiments.append(("P2C+: SkeletonGait++ (TTA+ASNorm)", "skeletongaitpp", dict(top_k_parts=10, top_k_gallery=3, use_tta=True, use_as_norm=True, use_whitening=False)))
        experiments.append(("P2C++: SkeletonGait++ (TTA+ASNorm+PCA)", "skeletongaitpp", dict(top_k_parts=10, top_k_gallery=3, use_tta=True, use_as_norm=True, use_whitening=True)))

    print("\n" + "=" * 124)
    print(
        f"{'Experimento':40s} | {'All':5s} | {'Frnt':5s} | {'45_I':5s} | {'45_D':5s} | "
        f"{'X-Frt':5s} | {'X-45I':5s} | {'X-45D':5s} | {'Ropa3':5s} | {'Ropa1':5s} | {'AUC':5s} | {'EER%':5s} | {'ms30f':5s}"
    )
    print("-" * 124)

    summary = {}
    for label, mk, cfg in experiments:
        res = evaluate_config(payloads[mk], **cfg)
        summary[label] = res
        sc = res["scenarios"]
        e3 = f"{res['ext_3cam'][0]}/{res['ext_3cam'][1]}"
        e1 = f"{res['ext_1cam'][0]}/{res['ext_1cam'][1]}"
        print(
            f"{label:40s} | "
            f"{sc['all_available']['top1']:5.1f} | "
            f"{sc['only_frente']['top1']:5.1f} | "
            f"{sc['only_45_izq']['top1']:5.1f} | "
            f"{sc['only_45_der']['top1']:5.1f} | "
            f"{sc['cross_frente']['top1']:5.1f} | "
            f"{sc['cross_45_izq']['top1']:5.1f} | "
            f"{sc['cross_45_der']['top1']:5.1f} | "
            f"{e3:5s} | {e1:5s} | "
            f"{res['openset_auc']:5.3f} | "
            f"{res['openset_eer']:5.2f} | "
            f"{res['avg_inf_ms_30f']:5.0f}",
            flush=True,
        )
    print("=" * 124)

    out_json = OUT_V2 / "benchmark_ab_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nReporte guardado en: {out_json}", flush=True)


if __name__ == "__main__":
    main()
