"""Motor de Matching v2: Vectorizado (<1 ms), Agnóstico al Ángulo, TTA Flip y AS-Norm.

Mejoras sobre Plan 1 (`match_probe_against_gallery`):
  1. Precomputa la galería normalizada por partes una sola vez en formato matricial
     `(N, P, C)` -> matching completo en < 1 ms vía `np.einsum`.
  2. Elimina el bonus fijo `+0.02` por etiqueta de ángulo (funciona igual en escenas
     no controladas donde se desconoce el ángulo de cámara).
  3. Reemplaza `Max` simple por `Top-K Mean` por sujeto (robusto a pasadas ruidosas).
  4. Soporta Test-Time Augmentation por espejo horizontal (original + flipped).
  5. Soporta **AS-Norm (Adaptive Score Normalization)** usando el cohort de impostores
     para estabilizar el umbral Open-Set (`Desconocido`) en 1 cámara, 3 cámaras y Wild.
"""

from __future__ import annotations

import numpy as np


EPS = 1e-8


def center_and_l2_parts(
    vecs: np.ndarray,
    mean_parts: np.ndarray | None = None,
    parts: int = 16,
    channels: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Centra por franja (part-wise) y normaliza L2 cada franja.

    `vecs`: `(N, P*C)` o `(P*C,)`.
    Retorna `(norm_parts, mean_parts)` con `mean_parts` de forma `(P, C)`.
    """
    arr = np.asarray(vecs, dtype=np.float32)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    p_arr = arr.reshape(-1, parts, channels)
    if mean_parts is None:
        mean_parts = p_arr.mean(axis=0)
    centered = p_arr - mean_parts[None, :, :]
    norms = np.linalg.norm(centered, axis=2, keepdims=True)
    normed = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > EPS)
    if single:
        return normed[0], mean_parts
    return normed, mean_parts


class VectorizedGaitMatcher:
    """Índice vectorizado para evaluación rápida y calibrada (Open-Set + Cross-View)."""

    def __init__(
        self,
        entries: list[dict],
        parts: int = 16,
        channels: int = 256,
        top_k_parts: int = 10,
        top_k_gallery: int = 3,
        use_as_norm: bool = False,
        cohort_size: int = 20,
        adapter=None,
    ):
        """`entries`: lista de dicts `{subject, angle, emb, [emb_flip]}`."""
        self.parts = parts
        self.channels = channels
        self.top_k_parts = min(top_k_parts, parts)
        self.top_k_gallery = top_k_gallery
        self.use_as_norm = use_as_norm
        self.cohort_size = cohort_size
        self.adapter = adapter

        raw_embs = []
        self.subjects: list[str] = []
        self.angles: list[str] = []
        for e in entries:
            if e.get("emb") is None:
                continue
            vec = e["emb"].astype(np.float32)
            if self.adapter is not None:
                vec = self.adapter.transform_one(vec)
            raw_embs.append(vec)
            self.subjects.append(e["subject"])
            self.angles.append(e.get("angle", "unknown"))

            # Si la galería incluye versión espejada (TTA), también la indexamos
            if e.get("emb_flip") is not None:
                vec_f = e["emb_flip"].astype(np.float32)
                if self.adapter is not None:
                    vec_f = self.adapter.transform_one(vec_f)
                raw_embs.append(vec_f)
                self.subjects.append(e["subject"])
                self.angles.append(e.get("angle", "unknown"))

        if not raw_embs:
            raise ValueError("La galería v2 está vacía.")

        self.subjects_arr = np.array(self.subjects)
        self.angles_arr = np.array(self.angles)
        self.unique_subjects = sorted(set(self.subjects))
        self.subj_indices = {
            s: np.where(self.subjects_arr == s)[0] for s in self.unique_subjects
        }

        mat = np.stack(raw_embs, axis=0)  # (N, P*C)
        self.gallery_parts, self.mean_parts = center_and_l2_parts(
            mat, mean_parts=None, parts=self.parts, channels=self.channels
        )  # (N, P, C), (P, C)

        # Precomputar estadísticas de cohort por sujeto para AS-Norm (impostor stats)
        self.subj_cohort_stats: dict[str, tuple[float, float]] = {}
        if self.use_as_norm:
            self._precompute_as_norm_stats()

    def _part_sim_matrix(self, query_parts: np.ndarray, target_parts: np.ndarray) -> np.ndarray:
        """Calcula similitud Top-K por partes entre `query_parts (Q, P, C)` y `target_parts (N, P, C)`."""
        # sims_per_part: (Q, N, P)
        sims_per_part = np.einsum("qpc,npc->qnp", query_parts, target_parts, optimize=True)
        if self.top_k_parts >= self.parts:
            return sims_per_part.mean(axis=2)
        # np.partition es O(P) mucho más rápido que sort completo
        topk = np.partition(sims_per_part, self.parts - self.top_k_parts, axis=2)[
            :, :, -self.top_k_parts :
        ]
        return topk.mean(axis=2)

    def _precompute_as_norm_stats(self) -> None:
        """Calcula (mu_g, std_g) de las mejores coincidencias impostoras para cada sujeto."""
        # Prototipo medio por sujeto para estimación rápida del cohort de galería
        protos = []
        for s in self.unique_subjects:
            idxs = self.subj_indices[s]
            p = self.gallery_parts[idxs].mean(axis=0)
            n = np.linalg.norm(p, axis=1, keepdims=True)
            p = np.divide(p, n, out=np.zeros_like(p), where=n > EPS)
            protos.append(p)
        proto_arr = np.stack(protos, axis=0)  # (S, P, C)

        # Similitud de cada prototipo contra toda la galería
        sim_all = self._part_sim_matrix(proto_arr, self.gallery_parts)  # (S, N)
        for i, s in enumerate(self.unique_subjects):
            imp_mask = self.subjects_arr != s
            imp_scores = sim_all[i, imp_mask]
            k = min(self.cohort_size, len(imp_scores))
            top_imp = np.partition(imp_scores, len(imp_scores) - k)[-k:]
            self.subj_cohort_stats[s] = (
                float(top_imp.mean()),
                float(max(top_imp.std(), 1e-3)),
            )

    def _prep_query(self, emb: np.ndarray) -> np.ndarray:
        vec = emb.astype(np.float32)
        if self.adapter is not None:
            vec = self.adapter.transform_one(vec)
        q_parts, _ = center_and_l2_parts(
            vec, mean_parts=self.mean_parts, parts=self.parts, channels=self.channels
        )
        return q_parts  # (P, C)

    def match_views(
        self,
        probe_views: dict[str, tuple[np.ndarray | None, np.ndarray | None]],
        exclude_same_angle: bool = False,
    ) -> list[tuple[str, float]]:
        """Compara 1 a 3 vistas de un probe contra toda la galería.

        `probe_views`: `{angle: (emb, emb_flip_or_None)}`.
        Retorna `[(subject, score), ...]` ordenado descendentemente.
        """
        view_subject_scores: dict[str, list[float]] = {s: [] for s in self.unique_subjects}

        for p_angle, (emb, emb_flip) in probe_views.items():
            q_list = []
            if emb is not None:
                q_list.append(self._prep_query(emb))
            if emb_flip is not None:
                q_list.append(self._prep_query(emb_flip))
            if not q_list:
                continue

            q_arr = np.stack(q_list, axis=0)  # (1 o 2, P, C)
            sim_mat = self._part_sim_matrix(q_arr, self.gallery_parts)  # (Q, N)
            # Para cada entrada de galería, tomar el mejor entre original y espejado (TTA)
            sim_vec = sim_mat.max(axis=0)  # (N,)

            if exclude_same_angle:
                valid_mask = self.angles_arr != p_angle
            else:
                valid_mask = np.ones(len(self.angles_arr), dtype=bool)

            # Calcular score crudo por sujeto usando Top-K Mean de sus entradas válidas
            raw_subj_scores = {}
            for s in self.unique_subjects:
                idxs = self.subj_indices[s]
                idxs = idxs[valid_mask[idxs]]
                if len(idxs) == 0:
                    continue
                s_vals = sim_vec[idxs]
                k = min(self.top_k_gallery, len(s_vals))
                if k == 1:
                    raw_subj_scores[s] = float(s_vals.max())
                else:
                    top_vals = np.partition(s_vals, len(s_vals) - k)[-k:]
                    raw_subj_scores[s] = float(top_vals.mean())

            if not raw_subj_scores:
                continue

            if self.use_as_norm:
                # Estadísticas del probe contra sus top impostores (Z-Norm + T-Norm = AS-Norm)
                all_valid_scores = sim_vec[valid_mask]
                kp = min(self.cohort_size, len(all_valid_scores))
                top_p = np.partition(all_valid_scores, len(all_valid_scores) - kp)[-kp:]
                mu_p = float(top_p.mean())
                std_p = float(max(top_p.std(), 1e-3))

                for s, sc in raw_subj_scores.items():
                    mu_g, std_g = self.subj_cohort_stats[s]
                    z_p = (sc - mu_p) / std_p
                    z_g = (sc - mu_g) / std_g
                    view_subject_scores[s].append(0.5 * (z_p + z_g))
            else:
                for s, sc in raw_subj_scores.items():
                    view_subject_scores[s].append(sc)

        results = []
        for s, sc_list in view_subject_scores.items():
            if sc_list:
                results.append((s, float(np.mean(sc_list))))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
