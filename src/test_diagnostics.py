"""Test 3: Diagnóstico de errores (31 sujetos) y calibración de umbral Desconocido."""

import json
import pickle
from pathlib import Path

import numpy as np

from phase3_evaluate import load_take_pass_embeddings
from phase3_gait_engine import GaitBaseInference, match_probe_against_gallery

MANIFEST_PATH = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\configs\dataset_manifest.json")
CKPT_PATH = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\weights\GaitBase_DA-180000.pt")
GALLERY_DB_PATH = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\outputs\gallery_db.pkl")


def main():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(GALLERY_DB_PATH, "rb") as f:
        gallery_db = pickle.load(f)

    model = GaitBaseInference(CKPT_PATH)
    alpha_sil = 0.85

    # 1. Cargar todos los probes en memoria una sola vez
    all_probes = []
    for subj, sdata in sorted(manifest["subjects"].items()):
        if subj not in gallery_db:
            continue
        for take in sdata["probe"]:
            passes = load_take_pass_embeddings("subjects", subj, take["take_name"], model)
            for p in passes:
                all_probes.append((subj, take["take_name"], p["pass_name"], p["views"]))

    print(f"Total probes cargados: {len(all_probes)} pasadas de {len(gallery_db)} sujetos\n")

    # 2. Detectar errores exactos en all_available
    known_scores = []
    known_margins = []
    errors = []

    for subj, take_name, pass_name, views in all_probes:
        ranked = match_probe_against_gallery(views, gallery_db, alpha_sil=alpha_sil)
        top1_name, top1_sc = ranked[0]
        top2_name, top2_sc = ranked[1]
        margin = top1_sc - top2_sc
        if top1_name == subj:
            known_scores.append(top1_sc)
            known_margins.append(margin)
        else:
            errors.append((subj, take_name, pass_name, list(views.keys()), top1_name, top1_sc, top2_name, top2_sc))

    print(f"=== Errores exactos en all_available ({len(errors)}/{len(all_probes)}) ===")
    for e in errors:
        subj, tname, pname, angs, p1, s1, p2, s2 = e
        print(f"  [FALLO] Real={subj} ({tname}/{pname}, {angs}) -> Pred1={p1} ({s1:.4f}) | Pred2={p2} ({s2:.4f})")

    # 3. Simulación Open-Set (sacar 5 sujetos de la galería para medir Desconocidos)
    held_out = sorted(gallery_db.keys())[:5]
    reduced_gallery = {k: v for k, v in gallery_db.items() if k not in held_out}
    unknown_scores = []
    unknown_margins = []

    for subj, take_name, pass_name, views in all_probes:
        if subj not in held_out:
            continue
        ranked = match_probe_against_gallery(views, reduced_gallery, alpha_sil=alpha_sil)
        top1_sc = ranked[0][1]
        margin = top1_sc - ranked[1][1]
        unknown_scores.append(top1_sc)
        unknown_margins.append(margin)

    ks = np.array(known_scores)
    km = np.array(known_margins)
    us = np.array(unknown_scores)
    um = np.array(unknown_margins)

    print("\n=== Calibracion Open-Set (Conocidos vs Desconocidos) ===")
    print(f"  Conocidos   (n={len(ks):3d}): score min={ks.min():.4f}, p5={np.percentile(ks,5):.4f}, media={ks.mean():.4f} | margen p5={np.percentile(km,5):.4f}, media={km.mean():.4f}")
    print(f"  Desconocidos(n={len(us):3d}): score max={us.max():.4f}, p95={np.percentile(us,95):.4f}, media={us.mean():.4f} | margen p95={np.percentile(um,95):.4f}, media={um.mean():.4f}")

    # Probar rejilla de umbrales
    print("\n  Rejilla de umbrales (TAR = Conocidos aceptados, TRR = Desconocidos rechazados):")
    for sc_th in (0.940, 0.950, 0.955, 0.960, 0.965):
        for mg_th in (0.005, 0.010, 0.015):
            tar = float(np.mean((ks >= sc_th) & (km >= mg_th))) * 100
            trr = float(np.mean((us < sc_th) | (um < mg_th))) * 100
            print(f"    min_score={sc_th:.3f}, min_margin={mg_th:.3f} -> TAR={tar:5.1f}% | TRR={trr:5.1f}%")


if __name__ == "__main__":
    main()

