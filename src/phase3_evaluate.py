"""Fase 3b: Construcción de Galería (toma_01..toma_07) y Evaluación (toma_08+ y *Prueba).

Evalúa la precisión Top-1 y Top-5 del motor multimodal (GaitBase GREW + Esqueleto)
en escenarios de 1 ángulo (`frente`, `45_izq`, `45_der`) y multi-ángulo flexible (1 a 3).
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from phase3_gait_engine import (
    GaitBaseInference,
    extract_skeleton_embedding,
    match_probe_against_gallery,
)

MANIFEST_PATH = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\configs\dataset_manifest.json")
FEATURES_ROOT = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\outputs\features")
CKPT_PATH = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\weights\GaitBase_DA-180000.pt")
GALLERY_DB_PATH = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\outputs\gallery_db.pkl")


def load_take_pass_embeddings(
    group: str, subject: str, take_name: str, model: GaitBaseInference
) -> list[dict]:
    """Carga y extrae embeddings por cada (pass_XX, angle) disponible de una toma."""
    take_dir = FEATURES_ROOT / group / subject / take_name
    if not take_dir.exists():
        return []

    passes_out = []
    for pass_dir in sorted(take_dir.iterdir()):
        if not pass_dir.is_dir() or not pass_dir.name.startswith("pass_"):
            continue
        views = {}
        for angle_dir in sorted(pass_dir.iterdir()):
            if not angle_dir.is_dir():
                continue
            sil_p = angle_dir / "silhouettes.npy"
            kpt_p = angle_dir / "keypoints.npy"
            if not sil_p.exists() or not kpt_p.exists():
                continue
            sil_emb = model.extract_embedding(np.load(sil_p))
            skel_emb = extract_skeleton_embedding(np.load(kpt_p))
            if sil_emb is not None or skel_emb is not None:
                views[angle_dir.name] = (sil_emb, skel_emb)
        if views:
            passes_out.append({"pass_name": pass_dir.name, "views": views})
    return passes_out


def build_gallery(manifest: dict, model: GaitBaseInference) -> dict[str, list[dict]]:
    gallery_db = defaultdict(list)
    for subj, sdata in manifest["subjects"].items():
        for take in sdata["gallery"]:
            passes = load_take_pass_embeddings("subjects", subj, take["take_name"], model)
            for p in passes:
                for angle, (sil_emb, skel_emb) in p["views"].items():
                    gallery_db[subj].append(
                        {
                            "take": take["take_name"],
                            "pass": p["pass_name"],
                            "angle": angle,
                            "sil": sil_emb,
                            "sil_format": "part-major-raw-v1",
                            "skel": skel_emb,
                        }
                    )
    return dict(gallery_db)


def evaluate_probes(
    manifest: dict,
    gallery_db: dict[str, list[dict]],
    model: GaitBaseInference,
    alpha_sil: float = 0.78,
):
    scenarios = [
        ("all_available", None, False),
        ("only_frente", "frente", False),
        ("only_45_izq", "45_izq", False),
        ("only_45_der", "45_der", False),
        ("cross_frente", "frente", True),
        ("cross_45_izq", "45_izq", True),
        ("cross_45_der", "45_der", True),
    ]

    for scen_name, filter_angle, excl_same in scenarios:
        top1, top5, total = 0, 0, 0
        for subj, sdata in manifest["subjects"].items():
            if subj not in gallery_db:
                continue
            for take in sdata["probe"]:
                passes = load_take_pass_embeddings("subjects", subj, take["take_name"], model)
                for p in passes:
                    views = p["views"]
                    if filter_angle is not None:
                        if filter_angle not in views:
                            continue
                        views = {filter_angle: views[filter_angle]}
                    ranked = match_probe_against_gallery(
                        views, gallery_db, alpha_sil=alpha_sil, exclude_same_angle=excl_same
                    )
                    if not ranked:
                        continue
                    total += 1
                    preds = [r[0] for r in ranked[:5]]
                    if preds[0] == subj:
                        top1 += 1
                    if subj in preds:
                        top5 += 1
        if total > 0:
            print(
                f"  [{scen_name:14s}] Probe passes={total:3d} | "
                f"Top-1={100*top1/total:5.1f}% ({top1}/{total}) | "
                f"Top-5={100*top5/total:5.1f}% ({top5}/{total})"
            )

    # Evaluar *Prueba (external_test) para ver a quién se parecen más
    print("\n--- Resultados en *Prueba (external_test) ---")
    for ext_subj, takes in manifest["external_test"].items():
        for take in takes:
            passes = load_take_pass_embeddings("external_test", ext_subj, take["take_name"], model)
            for p in passes:
                ranked = match_probe_against_gallery(p["views"], gallery_db, alpha_sil=alpha_sil)
                top3 = ", ".join([f"{name} ({sc:.3f})" for name, sc in ranked[:3]])
                print(
                    f"  {ext_subj}/{take['take_name']}/{p['pass_name']} "
                    f"[{list(p['views'].keys())}] -> Top-3: {top3}"
                )


def main():
    parser = argparse.ArgumentParser(description="Fase 3: Galería y Evaluación")
    parser.add_argument("--alpha-sil", type=float, default=0.65, help="Peso de silueta vs esqueleto")
    args = parser.parse_args()

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    model = GaitBaseInference(CKPT_PATH)
    gallery_db = build_gallery(manifest, model)
    GALLERY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GALLERY_DB_PATH, "wb") as f:
        pickle.dump(gallery_db, f)

    print(f"Galería construida: {len(gallery_db)} sujetos, "
          f"{sum(len(v) for v in gallery_db.values())} secuencias en {GALLERY_DB_PATH}")
    evaluate_probes(manifest, gallery_db, model, alpha_sil=args.alpha_sil)


if __name__ == "__main__":
    main()

