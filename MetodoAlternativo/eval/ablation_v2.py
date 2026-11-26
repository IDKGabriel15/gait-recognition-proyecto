"""Ablación rápida (<3s) sobre embeddings cacheados: sin AS-Norm, Top-K, TTA, PCA-Whitening y *Prueba."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA")
V2_ROOT = ROOT / "MetodoAlternativo"
sys.path.insert(0, str(V2_ROOT / "src"))
sys.path.insert(0, str(V2_ROOT / "eval"))

from benchmark_ab import _group_passes, evaluate_config
from domain_adapter import PartWisePCAWhitener
from matcher_v2 import VectorizedGaitMatcher

manifest = json.load(open(ROOT / "configs" / "dataset_manifest.json", encoding="utf-8"))
print("Sujetos en galería (Alej/Dan/Joel):", [s for s in manifest["subjects"] if any(k in s.lower() for k in ("alej", "dan", "joel"))])

print("\n" + "=" * 116)
print(f"{'Modelo':12s} | {'Configuración':28s} | {'All':5s} | {'X-Frt':5s} | {'X-45I':5s} | {'X-45D':5s} | {'AUC':6s} | {'EER%':5s} | {'TAR@1%':6s}")
print("-" * 116)

for mk in ("gaitbase_p1", "deepgaitv2", "skeletongaitpp"):
    cpath = V2_ROOT / "outputs" / f"cache_{mk}.pkl"
    if not cpath.exists():
        continue
    p = pickle.load(open(cpath, "rb"))
    for name, cfg in [
        ("Base (Top10, g=3)", dict(top_k_parts=10, top_k_gallery=3, use_tta=False, use_as_norm=False, use_whitening=False)),
        ("TTA Flip (Top10, g=3)", dict(top_k_parts=10, top_k_gallery=3, use_tta=True, use_as_norm=False, use_whitening=False)),
        ("TTA + PCA (Top10, sin AS)", dict(top_k_parts=10, top_k_gallery=3, use_tta=True, use_as_norm=False, use_whitening=True)),
        ("TTA + PCA (Top8, sin AS)", dict(top_k_parts=8, top_k_gallery=3, use_tta=True, use_as_norm=False, use_whitening=True)),
        ("TTA + PCA (Top12, sin AS)", dict(top_k_parts=12, top_k_gallery=3, use_tta=True, use_as_norm=False, use_whitening=True)),
    ]:
        r = evaluate_config(p, **cfg)
        sc = r["scenarios"]
        print(
            f"{mk:12s} | {name:28s} | {sc['all_available']['top1']:5.1f} | "
            f"{sc['cross_frente']['top1']:5.1f} | {sc['cross_45_izq']['top1']:5.1f} | "
            f"{sc['cross_45_der']['top1']:5.1f} | {r['openset_auc']:6.4f} | "
            f"{r['openset_eer']:5.2f} | {r['openset_tar_far1']:6.1f}",
            flush=True,
        )

    gal = [{"subject": r["subject"], "angle": r["angle"], "emb": r["emb"]} for r in p["records"] if r["split"] == "gallery" and r["emb"] is not None]
    ad = PartWisePCAWhitener(16, 256, whiten_power=0.5).fit(np.stack([e["emb"] for e in gal]))
    mt = VectorizedGaitMatcher(gal, top_k_parts=8, top_k_gallery=3, use_as_norm=False, adapter=ad)
    print(f"  --- Predicciones *Prueba ({mk} TTA+PCA Top8) ---")
    for ep in _group_passes(p["records"], "external"):
        rk = mt.match_views(ep["views"])[:3]
        print(f"    {ep['subject']:16s} {ep['take']} {ep['pass']} -> {[(x[0], round(x[1], 3)) for x in rk]}")
