"""Evaluación del Ensamble Final v2 (`DeepGaitV2-P3D + GaitBase` con TTA + PCA-Whitening)
y verificación de la forma de máscaras en los videos no controlados.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA")
V2_ROOT = ROOT / "MetodoAlternativo"
OUT_V2 = V2_ROOT / "outputs"
sys.path.insert(0, str(V2_ROOT / "src"))
sys.path.insert(0, str(V2_ROOT / "eval"))

from benchmark_ab import _group_passes
from domain_adapter import PartWisePCAWhitener
from matcher_v2 import VectorizedGaitMatcher


def build_matcher(payload: dict, top_k_parts: int = 10, top_k_gallery: int = 3, use_pca: bool = True, exclude_subjs: set | None = None):
    gal = [
        {"subject": r["subject"], "angle": r["angle"], "emb": r["emb"]}
        for r in payload["records"]
        if r["split"] == "gallery" and r["emb"] is not None and (exclude_subjs is None or r["subject"] not in exclude_subjs)
    ]
    ad = PartWisePCAWhitener(16, 256, whiten_power=0.5).fit(np.stack([e["emb"] for e in gal])) if use_pca else None
    return VectorizedGaitMatcher(gal, top_k_parts=top_k_parts, top_k_gallery=top_k_gallery, use_as_norm=False, adapter=ad)


def match_ensemble(m_dg, m_gb, views_dg, views_gb, w_dg: float = 0.65, exclude_same_angle: bool = False):
    r_dg = dict(m_dg.match_views(views_dg, exclude_same_angle=exclude_same_angle))
    r_gb = dict(m_gb.match_views(views_gb, exclude_same_angle=exclude_same_angle))
    merged = [(s, w_dg * r_dg.get(s, 0.0) + (1.0 - w_dg) * r_gb.get(s, 0.0)) for s in r_dg]
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged


def main():
    p_dg = pickle.load(open(OUT_V2 / "cache_deepgaitv2.pkl", "rb"))
    p_gb = pickle.load(open(OUT_V2 / "cache_gaitbase_p1.pkl", "rb"))

    m_dg = build_matcher(p_dg, top_k_parts=10, top_k_gallery=3, use_pca=True)
    m_gb = build_matcher(p_gb, top_k_parts=8, top_k_gallery=3, use_pca=True)

    probes_dg = {(p["subject"], p["take"], p["pass"]): p["views"] for p in _group_passes(p_dg["records"], "probe")}
    probes_gb = {(p["subject"], p["take"], p["pass"]): p["views"] for p in _group_passes(p_gb["records"], "probe")}
    keys = sorted(set(probes_dg.keys()) & set(probes_gb.keys()))

    scenarios = [
        ("all_available", None, False),
        ("only_frente", "frente", False),
        ("only_45_izq", "45_izq", False),
        ("only_45_der", "45_der", False),
        ("cross_frente", "frente", True),
        ("cross_45_izq", "45_izq", True),
        ("cross_45_der", "45_der", True),
    ]

    print("=== ENSAMBLE V2: 0.60 DeepGaitV2-P3D + 0.40 GaitBase (TTA + PCA-Whitening) ===")
    ens_res = {}
    for sname, fang, excl in scenarios:
        t1, tot = 0, 0
        for k in keys:
            vd, vg = probes_dg[k], probes_gb[k]
            if fang is not None:
                if fang not in vd or fang not in vg:
                    continue
                vd, vg = {fang: vd[fang]}, {fang: vg[fang]}
            rk = match_ensemble(m_dg, m_gb, vd, vg, w_dg=0.60, exclude_same_angle=excl)
            if not rk:
                continue
            tot += 1
            if rk[0][0] == k[0]:
                t1 += 1
        ens_res[sname] = round(100.0 * t1 / max(tot, 1), 1)
        print(f"  {sname:14s}: {100.0 * t1 / max(tot, 1):5.1f}% ({t1}/{tot})")

    # *Prueba (cambio de ropa)
    ext_map = {"AlejandroPrueba": "Alejandro", "DanielPrueba": "DanielOntiveros", "JoelPrueba": "JoelFelix"}
    ext_dg = {(p["subject"], p["take"], p["pass"]): p["views"] for p in _group_passes(p_dg["records"], "external")}
    ext_gb = {(p["subject"], p["take"], p["pass"]): p["views"] for p in _group_passes(p_gb["records"], "external")}
    ok3, tot3, ok1, tot1 = 0, 0, 0, 0
    for k in sorted(set(ext_dg.keys()) & set(ext_gb.keys())):
        gt = ext_map[k[0]]
        rk3 = match_ensemble(m_dg, m_gb, ext_dg[k], ext_gb[k], w_dg=0.60)
        tot3 += 1
        ok3 += int(rk3[0][0] == gt)
        if "frente" in ext_dg[k] and "frente" in ext_gb[k]:
            rk1 = match_ensemble(m_dg, m_gb, {"frente": ext_dg[k]["frente"]}, {"frente": ext_gb[k]["frente"]}, w_dg=0.60)
            tot1 += 1
            ok1 += int(rk1[0][0] == gt)
        print(f"  *Prueba {k[0]:16s} {k[1]} {k[2]} -> 3cam={rk3[0][0]} ({rk3[0][1]:.3f})")
    print(f"  >>> *Prueba 3-cam: {ok3}/{tot3} | 1-cam (frente): {ok1}/{tot1}")

    # Open-Set Leave-5-Subjects-Out (3 folds, 1 cámara frente)
    all_subjs = sorted({r["subject"] for r in p_dg["records"] if r["split"] == "gallery"})
    gen, imp = [], []
    for f0 in (0, 10, 20):
        held = set(all_subjs[f0 : f0 + 5])
        md = build_matcher(p_dg, top_k_parts=10, top_k_gallery=3, use_pca=True, exclude_subjs=held)
        mg = build_matcher(p_gb, top_k_parts=8, top_k_gallery=3, use_pca=True, exclude_subjs=held)
        for k in keys:
            if "frente" not in probes_dg[k] or "frente" not in probes_gb[k]:
                continue
            rk = match_ensemble(md, mg, {"frente": probes_dg[k]["frente"]}, {"frente": probes_gb[k]["frente"]}, w_dg=0.60)
            if not rk:
                continue
            (imp if k[0] in held else gen).append(rk[0][1])

    y_t = np.array([1] * len(gen) + [0] * len(imp))
    y_s = np.array(gen + imp, dtype=np.float32)
    auc = float(roc_auc_score(y_t, y_s))
    fpr, tpr, _ = roc_curve(y_t, y_s)
    eer_i = int(np.nanargmin(np.abs((1.0 - tpr) - fpr)))
    eer = float(0.5 * (fpr[eer_i] + (1.0 - tpr[eer_i])) * 100.0)
    tar1 = float(tpr[np.searchsorted(fpr, 0.01, side="right") - 1] * 100.0)
    print(f"  >>> Open-Set (1-cam frente): AUC={auc:.4f} | EER={eer:.2f}% | TAR@FAR=1%={tar1:.1f}%")

    ens_res.update({
        "ext_3cam": f"{ok3}/{tot3}",
        "ext_1cam": f"{ok1}/{tot1}",
        "openset_auc": round(auc, 4),
        "openset_eer": round(eer, 2),
        "openset_tar_far1": round(tar1, 1),
    })
    with open(OUT_V2 / "ensemble_v2_results.json", "w", encoding="utf-8") as f:
        json.dump(ens_res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
