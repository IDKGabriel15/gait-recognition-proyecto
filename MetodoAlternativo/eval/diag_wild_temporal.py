"""Diagnóstico rápido en `wild_windows_cache.pkl`:
Comprueba el impacto de:
  1. Alineación de tasa temporal (30 fps -> ~10 fps, stride=3) para modelos P3D (`DeepGaitV2`).
  2. Sin PCA-Whitening vs con PCA-Whitening en dominio no controlado (*In-the-Wild*).
  3. Fusión multimodal `DeepGaitV2-P3D + GaitBase + Esqueleto 99-D`.
"""

import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA")
V2_ROOT = ROOT / "MetodoAlternativo"
OUT_V2 = V2_ROOT / "outputs"

sys.path.insert(0, str(V2_ROOT / "src"))
sys.path.insert(0, str(ROOT / "src"))

from domain_adapter import PartWisePCAWhitener
from heatmap_generator import flip_sil_and_maps
from matcher_v2 import VectorizedGaitMatcher
from phase3_gait_engine import GaitBaseInference, extract_skeleton_embedding


def load_gallery_entries(model_key: str):
    p = pickle.load(open(OUT_V2 / f"cache_{model_key}.pkl", "rb"))
    return [
        {"subject": r["subject"], "angle": r["angle"], "emb": r["emb"]}
        for r in p["records"]
        if r["split"] == "gallery" and r["emb"] is not None
    ]


def eval_wild_fast():
    wild = pickle.load(open(OUT_V2 / "wild_windows_cache.pkl", "rb"))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 10
    dg_sess = ort.InferenceSession(str(OUT_V2 / "onnx" / "deepgaitv2_fp32.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
    gb_model = GaitBaseInference(ROOT / "weights" / "GaitBase_DA-180000.pt")

    gal_dg = load_gallery_entries("deepgaitv2")
    gal_gb = load_gallery_entries("gaitbase_p1")

    m_dg_raw = VectorizedGaitMatcher(gal_dg, top_k_parts=8, top_k_gallery=3, use_as_norm=False)
    ad_dg = PartWisePCAWhitener(16, 256, whiten_power=0.5).fit(np.stack([e["emb"] for e in gal_dg]))
    m_dg_pca = VectorizedGaitMatcher(gal_dg, top_k_parts=8, top_k_gallery=3, use_as_norm=False, adapter=ad_dg)
    m_gb_raw = VectorizedGaitMatcher(gal_gb, top_k_parts=8, top_k_gallery=3, use_as_norm=False)

    # Cargar embeddings de esqueleto 99-D desde galería Plan 1 (solo lectura)
    p1_db = pickle.load(open(ROOT / "outputs" / "gallery_db.pkl", "rb"))
    all_sk = [g["skel"] for entries in p1_db.values() for g in entries if g.get("skel") is not None]
    sk_mean = np.mean(all_sk, axis=0).astype(np.float32)
    sk_gal = {}
    for subj, entries in p1_db.items():
        vecs = []
        for g in entries:
            if g.get("skel") is not None:
                v = g["skel"].astype(np.float32) - sk_mean
                n = np.linalg.norm(v)
                if n > 1e-8:
                    vecs.append(v / n)
        if vecs:
            sk_gal[subj] = np.stack(vecs, axis=0)

    def run_dg(sil_seq: np.ndarray):
        s = sil_seq[sil_seq.sum(axis=(1, 2)) > 0]
        if len(s) < 5:
            return None
        x = (s.astype(np.float32) / 255.0)[None, None, :, :, :]
        return dg_sess.run(None, {"input_seq": x})[0][0].astype(np.float32)

    def skel_scores(kpt_seq: np.ndarray) -> dict[str, float]:
        emb = extract_skeleton_embedding(kpt_seq)
        if emb is None:
            return {}
        v = emb - sk_mean
        n = np.linalg.norm(v)
        if n < 1e-8:
            return {}
        v = v / n
        out = {}
        for subj, mat in sk_gal.items():
            sims = mat @ v
            k = min(3, len(sims))
            out[subj] = float(np.sort(sims)[-k:].mean())
        return out

    modes = [
        "DG_Raw_10fps_TTA",
        "DG_PCA_10fps_TTA",
        "GB_Raw_TTA",
        "Fusion_DG10fps+GB+Skel",
    ]
    totals = {m: 0 for m in modes}

    for fname, info in wild.items():
        gt = info["gt"]
        wins = info["windows"]
        votes = {m: Counter() for m in modes}
        scores = {m: defaultdict(float) for m in modes}

        for sil_seq, kpt_seq in wins:
            # Alinear tasa temporal a ~10 fps (stride=3 sobre 30 fps) para P3D
            sil_10 = sil_seq[::3]
            if len(sil_10) < 8:
                sil_10 = sil_seq
            sf_10, _ = flip_sil_and_maps(sil_10, None)
            e_dg = run_dg(sil_10)
            ef_dg = run_dg(sf_10)

            e_gb = gb_model.extract_embedding(sil_10)
            ef_gb = gb_model.extract_embedding(sf_10)

            r_dg_raw = dict(m_dg_raw.match_views({"w": (e_dg, ef_dg)}))
            r_dg_pca = dict(m_dg_pca.match_views({"w": (e_dg, ef_dg)}))
            r_gb_raw = dict(m_gb_raw.match_views({"w": (e_gb, ef_gb)}))
            r_sk = skel_scores(kpt_seq)

            # Fusión multimodal: 0.45 DeepGaitV2-P3D (10fps) + 0.30 GaitBase + 0.25 Esqueleto
            r_fus = {}
            for subj in r_dg_raw:
                s = 0.45 * r_dg_raw[subj] + 0.30 * r_gb_raw.get(subj, 0.0) + 0.25 * r_sk.get(subj, 0.0)
                r_fus[subj] = s

            for m_name, r_dict in [
                ("DG_Raw_10fps_TTA", r_dg_raw),
                ("DG_PCA_10fps_TTA", r_dg_pca),
                ("GB_Raw_TTA", r_gb_raw),
                ("Fusion_DG10fps+GB+Skel", r_fus),
            ]:
                ranked = sorted(r_dict.items(), key=lambda x: x[1], reverse=True)
                if ranked:
                    votes[m_name][ranked[0][0]] += 1
                    for s_nm, s_v in ranked[:5]:
                        scores[m_name][s_nm] += s_v

        print(f"\nVideo: {fname} (GT={gt}, {len(wins)} ventanas)")
        for m_name in modes:
            pred = max(votes[m_name].keys(), key=lambda s: (votes[m_name][s], scores[m_name][s]))
            ok = pred == gt
            if ok:
                totals[m_name] += 1
            tag = "OK  " if ok else "FAIL"
            print(f"  [{tag}] {m_name:24s} -> Pred={pred:16s} | GT={votes[m_name].get(gt,0):2d}/{len(wins)} | Top3={votes[m_name].most_common(3)}")

    print("\n=== RESUMEN WILD (5 videos no controlados) ===")
    for m_name in modes:
        print(f"  {m_name:24s}: {totals[m_name]}/{len(wild)}")


if __name__ == "__main__":
    eval_wild_fast()
