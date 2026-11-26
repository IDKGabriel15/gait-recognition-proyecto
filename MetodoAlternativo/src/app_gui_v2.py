"""Interfaz Gráfica Interactiva v2 — Método Alternativo (`MetodoAlternativo/`).

Permite probar en tiempo real (Webcam o 1 a 3 videos .mp4):
  - Motor v2 seleccionable:
      1. DeepGaitV2-P3D (ONNX FP32, 91.0% Cross-View, EER 3.66%)
      2. Ensamble v2 (0.60 DeepGaitV2-P3D + 0.40 GaitBase, 91.5% Cross-View, AUC 0.9933)
      3. GaitBase + MatcherV2 (TTA+PCA, 8/8 cambio de ropa)
  - Tracker multi-persona (Húngaro IoU + velocidad) + ROI-Crop Pose.
  - Frame Quality Gate + Detector cinemático de marcha (lateral, diagonal o frontal).
  - Toggles en vivo: PCA-Whitening, TTA Horizontal Flip, Quality Gate y grabación .mp4.
  - Matching matricial vectorizado (< 1 ms) y Conclusión Final consolidada.
"""

from __future__ import annotations

import os
import pickle
import sys
import threading
import time
import tkinter as tk
from collections import Counter, defaultdict, deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageTk

ROOT = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA")
V2_ROOT = ROOT / "MetodoAlternativo"
OUT_V2 = V2_ROOT / "outputs"
VISUALS_V2 = OUT_V2 / "visuals"

os.environ.setdefault("YOLO_CONFIG_DIR", str((ROOT / "outputs").resolve()))
sys.path.insert(0, str(V2_ROOT / "src"))
sys.path.insert(0, str(ROOT / "src"))

from domain_adapter import PartWisePCAWhitener
from heatmap_generator import flip_sil_and_maps
from matcher_v2 import VectorizedGaitMatcher
from phase2_extract_and_visualize import calc_angle_deg, draw_skeleton
from phase3_gait_engine import GaitBaseInference
from quality_and_motion import compute_frame_quality, is_actively_walking
from tracker_extractor import MultiPersonGaitExtractor


ENGINE_MODES = [
    "DeepGaitV2-P3D (ONNX)",
    "Ensamble v2 (DeepGaitV2 + GaitBase)",
    "GaitBase + MatcherV2",
]


def _uniform_subsample(arr: np.ndarray, max_frames: int = 24) -> np.ndarray:
    if len(arr) <= max_frames:
        return arr
    idx = np.linspace(0, len(arr) - 1, max_frames).round().astype(np.int64)
    return arr[idx]


class GaitAppGUIv2:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Método Alternativo v2 — DeepGaitV2-P3D + Tracker Multi-Persona (31 Sujetos)")
        self.root.geometry("1320x800")
        self.root.configure(bg="#101318")

        self.running = False
        self.reset_requested = False
        self.worker_thread: threading.Thread | None = None
        self.video_sources: list[tuple[str, str]] = [("frente", "0")]

        self.mode_var = tk.StringVar(value=ENGINE_MODES[0])
        self.pca_var = tk.BooleanVar(value=True)
        self.tta_var = tk.BooleanVar(value=True)
        self.qgate_var = tk.BooleanVar(value=True)
        self.save_video_var = tk.BooleanVar(value=False)

        self.min_score_var = tk.DoubleVar(value=0.28)
        self.min_margin_var = tk.DoubleVar(value=0.025)
        self.status_var = tk.StringVar(value="Cargando motores v2 (DeepGaitV2 ONNX + GaitBase + Galerías)...")

        # Modelos y matchers precomputados (con y sin PCA)
        self.dg_sess: ort.InferenceSession | None = None
        self.gb_model: GaitBaseInference | None = None
        self.extractor: MultiPersonGaitExtractor | None = None
        self.matchers: dict[tuple[str, bool], VectorizedGaitMatcher] = {}

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        threading.Thread(target=self._load_models, daemon=True).start()

    def _build_ui(self):
        # Fila 1: Controles principales
        top = tk.Frame(self.root, bg="#191e26", pady=8, padx=12)
        top.pack(fill=tk.X)

        btn_cfg = {
            "bg": "#28303d",
            "fg": "#f0f4f8",
            "relief": tk.FLAT,
            "padx": 10,
            "pady": 6,
            "font": ("Segoe UI", 9, "bold"),
        }
        tk.Button(top, text="Webcam (Cam 0)", command=self._select_webcam, **btn_cfg).pack(side=tk.LEFT, padx=3)
        tk.Button(top, text="Abrir Videos (.mp4)", command=self._select_videos, **btn_cfg).pack(side=tk.LEFT, padx=3)

        self.btn_start = tk.Button(
            top,
            text="▶ Iniciar",
            command=self._toggle_start,
            bg="#1f8b4c",
            fg="white",
            relief=tk.FLAT,
            padx=14,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            state=tk.DISABLED,
        )
        self.btn_start.pack(side=tk.LEFT, padx=8)

        tk.Button(
            top,
            text="↻ Reiniciar",
            command=self._request_reset,
            bg="#b87314",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=6,
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            top,
            text="■ Finalizar",
            command=self._stop_stream,
            bg="#b32d2e",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=6,
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=3)

        tk.Label(top, text="Motor v2:", bg="#191e26", fg="#9fb0c7", font=("Segoe UI", 9, "bold")).pack(
            side=tk.LEFT, padx=(14, 4)
        )
        mode_cb = ttk.Combobox(
            top, textvariable=self.mode_var, values=ENGINE_MODES, state="readonly", width=32
        )
        mode_cb.pack(side=tk.LEFT, padx=4)

        chk_cfg = {
            "bg": "#191e26",
            "fg": "#d0d7de",
            "selectcolor": "#101318",
            "activebackground": "#191e26",
            "font": ("Segoe UI", 9),
        }
        tk.Checkbutton(top, text="Guardar .mp4", variable=self.save_video_var, **chk_cfg).pack(side=tk.RIGHT, padx=6)

        # Fila 2: Opciones v2 y sliders de umbral
        bar2 = tk.Frame(self.root, bg="#141820", pady=5, padx=12)
        bar2.pack(fill=tk.X)

        chk2 = {
            "bg": "#141820",
            "fg": "#79c0ff",
            "selectcolor": "#101318",
            "activebackground": "#141820",
            "font": ("Segoe UI", 9, "bold"),
        }
        tk.Checkbutton(
            bar2, text="PCA-Whitening", variable=self.pca_var, command=self._on_pca_toggle, **chk2
        ).pack(side=tk.LEFT, padx=(0, 10))
        tk.Checkbutton(bar2, text="TTA Flip", variable=self.tta_var, **chk2).pack(side=tk.LEFT, padx=(0, 10))
        tk.Checkbutton(bar2, text="Quality Gate", variable=self.qgate_var, **chk2).pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(bar2, text="Score mín:", bg="#141820", fg="#c9d1d9", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Scale(
            bar2,
            from_=0.10,
            to=0.90,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.min_score_var,
            bg="#141820",
            fg="#58a6ff",
            highlightthickness=0,
            length=150,
        ).pack(side=tk.LEFT, padx=(4, 14))

        tk.Label(bar2, text="Margen mín:", bg="#141820", fg="#c9d1d9", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Scale(
            bar2,
            from_=0.00,
            to=0.15,
            resolution=0.005,
            orient=tk.HORIZONTAL,
            variable=self.min_margin_var,
            bg="#141820",
            fg="#58a6ff",
            highlightthickness=0,
            length=140,
        ).pack(side=tk.LEFT, padx=(4, 12))

        self.source_lbl = tk.Label(bar2, text="Fuente: Webcam (Cam 0)", bg="#141820", fg="#8b949e", font=("Segoe UI", 9))
        self.source_lbl.pack(side=tk.RIGHT)

        # Lienzo principal de video (3 paneles + HUD)
        self.video_label = tk.Label(self.root, bg="#0b0e13")
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=8)

        # Barra de estado inferior
        bottom = tk.Frame(self.root, bg="#191e26", pady=6, padx=12)
        bottom.pack(fill=tk.X)
        tk.Label(bottom, textvariable=self.status_var, bg="#191e26", fg="#7ee787", font=("Segoe UI", 10, "bold")).pack(
            side=tk.LEFT
        )

    def _on_pca_toggle(self):
        if self.pca_var.get():
            self.min_score_var.set(0.28)
            self.min_margin_var.set(0.025)
        else:
            self.min_score_var.set(0.45)
            self.min_margin_var.set(0.025)

    def _build_matcher(self, model_key: str, use_pca: bool, top_k_parts: int) -> VectorizedGaitMatcher:
        with open(OUT_V2 / f"cache_{model_key}.pkl", "rb") as f:
            payload = pickle.load(f)
        gal = [
            {"subject": r["subject"], "angle": r["angle"], "emb": r["emb"]}
            for r in payload["records"]
            if r["split"] == "gallery" and r["emb"] is not None
        ]
        adapter = None
        if use_pca:
            adapter = PartWisePCAWhitener(16, 256, whiten_power=0.5).fit(
                np.stack([e["emb"] for e in gal], axis=0)
            )
        return VectorizedGaitMatcher(
            gal, top_k_parts=top_k_parts, top_k_gallery=3, use_as_norm=False, adapter=adapter
        )

    def _load_models(self):
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 8
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.dg_sess = ort.InferenceSession(
                str(OUT_V2 / "onnx" / "deepgaitv2_fp32.onnx"),
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )
            self.gb_model = GaitBaseInference(ROOT / "weights" / "GaitBase_DA-180000.pt")
            self.extractor = MultiPersonGaitExtractor()

            for mk, tk_parts in (("deepgaitv2", 10), ("gaitbase_p1", 8)):
                for pca_flag in (True, False):
                    self.matchers[(mk, pca_flag)] = self._build_matcher(mk, pca_flag, tk_parts)

            self.root.after(0, self._on_models_ready)
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Error cargando modelos v2", str(exc)))

    def _on_models_ready(self):
        self.btn_start.config(state=tk.NORMAL)
        self.status_var.set("Listo — Motores v2 cargados (DeepGaitV2-P3D ONNX + GaitBase + 31 sujetos).")

    def _select_webcam(self):
        if self.running:
            self._stop_stream()
        self.video_sources = [("frente", "0")]
        self.source_lbl.config(text="Fuente: Webcam (Cam 0)")

    def _select_videos(self):
        if self.running:
            self._stop_stream()
        paths = filedialog.askopenfilenames(
            title="Seleccionar 1 a 3 videos (.mp4)",
            initialdir=str(ROOT),
            filetypes=[("Videos MP4", "*.mp4;*.avi;*.mov"), ("Todos", "*.*")],
        )
        if not paths:
            return
        sources = []
        for idx, p in enumerate(paths[:3]):
            low = Path(p).name.lower()
            if "45_izq" in low:
                ang = "45_izq"
            elif "45_der" in low:
                ang = "45_der"
            elif "frente" in low:
                ang = "frente"
            else:
                ang = f"cam_{idx + 1}"
            sources.append((ang, p))
        self.video_sources = sources
        names = ", ".join([Path(p).name for _, p in sources])
        self.source_lbl.config(text=f"Fuente ({len(sources)}): {names[:55]}")

    def _toggle_start(self):
        if self.running:
            self._stop_stream()
        else:
            self.running = True
            self.reset_requested = False
            self.btn_start.config(text="⏸ Pausar/Parar", bg="#c95b12")
            self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
            self.worker_thread.start()

    def _request_reset(self):
        self.reset_requested = True

    def _stop_stream(self):
        self.running = False
        self.btn_start.config(text="▶ Iniciar", bg="#1f8b4c")

    def _extract_dg_onnx(self, sil_seq: np.ndarray) -> np.ndarray | None:
        s = sil_seq[sil_seq.sum(axis=(1, 2)) > 0]
        if len(s) < 5:
            return None
        x = (s.astype(np.float32) / 255.0)[None, None, :, :, :]
        vec = self.dg_sess.run(None, {"input_seq": x})[0][0].astype(np.float32)
        return vec if np.linalg.norm(vec) >= 1e-8 else None

    def _score_views(self, sil_views: dict[str, np.ndarray]) -> list[tuple[str, float]]:
        mode = self.mode_var.get()
        use_pca = self.pca_var.get()
        use_tta = self.tta_var.get()

        dg_views, gb_views = {}, {}
        for ang, s_raw in sil_views.items():
            # Alinear a 24 frames uniformes (~10 fps) para P3D
            s_sub = _uniform_subsample(s_raw, max_frames=24)
            s_flip, _ = flip_sil_and_maps(s_sub, None) if use_tta else (None, None)

            if mode in (ENGINE_MODES[0], ENGINE_MODES[1]):
                e = self._extract_dg_onnx(s_sub)
                ef = self._extract_dg_onnx(s_flip) if s_flip is not None else None
                if e is not None:
                    dg_views[ang] = (e, ef)

            if mode in (ENGINE_MODES[1], ENGINE_MODES[2]):
                e = self.gb_model.extract_embedding(s_sub)
                ef = self.gb_model.extract_embedding(s_flip) if s_flip is not None else None
                if e is not None:
                    gb_views[ang] = (e, ef)

        if mode == ENGINE_MODES[0] and dg_views:
            return self.matchers[("deepgaitv2", use_pca)].match_views(dg_views)
        if mode == ENGINE_MODES[2] and gb_views:
            return self.matchers[("gaitbase_p1", use_pca)].match_views(gb_views)
        if mode == ENGINE_MODES[1] and dg_views and gb_views:
            r_dg = dict(self.matchers[("deepgaitv2", use_pca)].match_views(dg_views))
            r_gb = dict(self.matchers[("gaitbase_p1", use_pca)].match_views(gb_views))
            merged = [(s, 0.60 * r_dg.get(s, 0.0) + 0.40 * r_gb.get(s, 0.0)) for s in r_dg]
            merged.sort(key=lambda x: x[1], reverse=True)
            return merged
        return []

    def _run_loop(self):
        caps = []
        for ang, src in self.video_sources:
            cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW) if src.isdigit() else cv2.VideoCapture(src)
            if not cap.isOpened():
                self.root.after(0, lambda s=src: messagebox.showerror("Error", f"No se pudo abrir: {s}"))
                self.root.after(0, self._stop_stream)
                return
            caps.append((ang, cap))

        main_angle, main_cap = caps[0]
        fps = main_cap.get(cv2.CAP_PROP_FPS) or 30.0
        cam_fps = {a: (c.get(cv2.CAP_PROP_FPS) or fps) for a, c in caps}
        cam_pos = {a: 0 for a, _ in caps}

        self.extractor.reset_tracker()
        panel_w, panel_h = 410, 330
        total_w, total_h = panel_w * 3, panel_h + 150

        writer = None
        if self.save_video_var.get():
            VISUALS_V2.mkdir(parents=True, exist_ok=True)
            out_p = VISUALS_V2 / f"gui_v2_{int(time.time())}.mp4"
            writer = cv2.VideoWriter(str(out_p), cv2.VideoWriter_fourcc(*"mp4v"), min(fps, 30.0), (total_w, total_h))

        window_frames = 60
        eval_interval = 15
        sil_bufs: dict[str, deque] = {a: deque(maxlen=window_frames) for a, _ in caps}
        kpt_buf_main: deque = deque(maxlen=window_frames)
        box_hist: deque = deque(maxlen=10)
        l_trail: deque = deque(maxlen=18)
        r_trail: deque = deque(maxlen=18)

        valid_cnt = 0
        current_top5: list[tuple[str, float]] = []
        current_decision = ("Analizando...", 0.0, 0.0)
        vote_history: list[tuple[str, float, float]] = []
        last_inf_ms = 0.0

        frame_idx = 0
        while self.running:
            if self.reset_requested:
                for b in sil_bufs.values():
                    b.clear()
                kpt_buf_main.clear()
                box_hist.clear()
                l_trail.clear()
                r_trail.clear()
                vote_history.clear()
                valid_cnt = 0
                current_top5 = []
                current_decision = ("Analizando...", 0.0, 0.0)
                self.extractor.reset_tracker()
                self.reset_requested = False

            ret, main_frm = main_cap.read()
            if not ret:
                break
            t_sec = frame_idx / fps
            frames = {main_angle: main_frm}
            for ang, cap in caps[1:]:
                target_idx = int(round(t_sec * cam_fps[ang]))
                while cam_pos[ang] <= target_idx:
                    r2, frm2 = cap.read()
                    cam_pos[ang] += 1
                    if r2:
                        frames[ang] = frm2

            mh, mw = main_frm.shape[:2]
            main_dets = self.extractor.process_frame(main_frm, use_tracker=True)
            main_det = main_dets[0] if main_dets else None

            is_walking = False
            q_score = 0.0
            ok_q = False
            main_kpts = np.zeros((17, 3), dtype=np.float32)
            main_sil = None

            if main_det is not None:
                box = main_det["box"]
                main_kpts = main_det["kpts"]
                main_sil = main_det["sil"]
                cx = float(0.5 * (box[0] + box[2]))
                bh = float(box[3] - box[1])
                box_hist.append((t_sec, cx, bh))

                ok_q, q_score = compute_frame_quality(main_sil, kpts_17x3=main_kpts, box_xyxy=box, frame_shape=(mh, mw))
                if not self.qgate_var.get():
                    ok_q = main_sil is not None
                is_walking, _ = is_actively_walking(list(kpt_buf_main) + [main_kpts], list(box_hist))

                if ok_q and is_walking and main_sil is not None:
                    sil_bufs[main_angle].append(main_sil)
                    kpt_buf_main.append(main_kpts)
                    valid_cnt += 1

                    # Extraer siluetas de cámaras secundarias si hay 2-3 videos
                    for ang, frm in frames.items():
                        if ang == main_angle:
                            continue
                        sdets = self.extractor.process_frame(frm, use_tracker=False)
                        if sdets and sdets[0]["sil"] is not None:
                            sil_bufs[ang].append(sdets[0]["sil"])

                    if len(sil_bufs[main_angle]) >= 16 and (valid_cnt % eval_interval == 0):
                        sil_views = {
                            a: np.stack(b, axis=0) for a, b in sil_bufs.items() if len(b) >= 12
                        }
                        t0 = time.perf_counter()
                        ranked = self._score_views(sil_views)
                        last_inf_ms = (time.perf_counter() - t0) * 1000.0
                        if ranked:
                            current_top5 = ranked[:5]
                            top1_nm, top1_sc = ranked[0]
                            top2_sc = ranked[1][1] if len(ranked) > 1 else 0.0
                            margin = top1_sc - top2_sc
                            if top1_sc < self.min_score_var.get() or margin < self.min_margin_var.get():
                                current_decision = ("Desconocido", top1_sc, margin)
                            else:
                                current_decision = (top1_nm, top1_sc, margin)
                            vote_history.append(current_decision)

            # Renderizado de 3 paneles
            p1 = main_frm.copy()
            dec_lbl, dec_sc, dec_mg = current_decision
            box_color = (0, 230, 140) if dec_lbl not in ("Analizando...", "Desconocido") else (0, 165, 255)
            for idx_d, d in enumerate(main_dets):
                x1, y1, x2, y2 = [int(v) for v in d["box"]]
                c = box_color if idx_d == 0 else (180, 180, 90)
                cv2.rectangle(p1, (x1, y1), (x2, y2), c, 2)
                tag = f"ID:{d['track_id']} {dec_lbl} ({dec_sc:.2f})" if idx_d == 0 else f"ID:{d['track_id']}"
                cv2.putText(p1, tag, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2)
                draw_skeleton(p1, d["kpts"], thickness=2, radius=3)
            p1_r = cv2.resize(p1, (panel_w, panel_h))
            st_txt = f"t={t_sec:.1f}s | {'CAMINANDO' if is_walking else 'PAUSA'} | Q={q_score:.2f} | Personas:{len(main_dets)}"
            cv2.putText(p1_r, st_txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, box_color, 2)

            p2 = np.full((mh, mw, 3), 20, dtype=np.uint8)
            if main_kpts[15, 2] >= 0.3:
                l_trail.append((int(main_kpts[15, 0]), int(main_kpts[15, 1])))
            if main_kpts[16, 2] >= 0.3:
                r_trail.append((int(main_kpts[16, 0]), int(main_kpts[16, 1])))
            for i in range(1, len(l_trail)):
                cv2.line(p2, l_trail[i - 1], l_trail[i], (60, 200, 95), 2)
            for i in range(1, len(r_trail)):
                cv2.line(p2, r_trail[i - 1], r_trail[i], (40, 150, 235), 2)
            draw_skeleton(p2, main_kpts, thickness=4, radius=5)
            p2_r = cv2.resize(p2, (panel_w, panel_h))
            lk = calc_angle_deg(main_kpts[11], main_kpts[13], main_kpts[15])
            rk = calc_angle_deg(main_kpts[12], main_kpts[14], main_kpts[16])
            cv2.putText(
                p2_r,
                f"ROI-Pose | Rodillas I:{lk:.0f} D:{rk:.0f}" if (lk and rk) else "ROI-Pose & Estela",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (240, 220, 90),
                2,
            )

            p3_r = np.full((panel_h, panel_w, 3), 16, dtype=np.uint8)
            if main_sil is not None:
                sil_big = cv2.resize(main_sil, (176, 256), interpolation=cv2.INTER_NEAREST)
                x0 = (panel_w - 176) // 2
                y0 = (panel_h - 256) // 2 + 12
                p3_r[y0 : y0 + 256, x0 : x0 + 176] = cv2.cvtColor(sil_big, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(p3_r, (x0 - 2, y0 - 2), (x0 + 178, y0 + 258), (0, 220, 255), 1)
            cv2.putText(
                p3_r,
                f"Silueta 64x44 | Buffer: {len(sil_bufs[main_angle])}/{window_frames} | {last_inf_ms:.0f}ms",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 220, 255),
                2,
            )

            top_row = np.hstack([p1_r, p2_r, p3_r])

            # HUD inferior con Top-5
            hud = np.full((150, total_w, 3), 18, dtype=np.uint8)
            cv2.putText(
                hud,
                f"Motor: {self.mode_var.get()} | Veredicto: {dec_lbl} (score={dec_sc:.3f}, margen={dec_mg:+.3f})",
                (16, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                box_color,
                2,
            )
            for rk_i, (nm, sc) in enumerate(current_top5):
                y_pos = 54 + rk_i * 20
                bar_w = int(np.clip(sc * 1.4, 0.0, 1.0) * 340)
                c = box_color if (rk_i == 0 and dec_lbl != "Desconocido") else (175, 185, 200)
                cv2.putText(hud, f"#{rk_i + 1} {nm:20s} {sc:.3f}", (16, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
                cv2.rectangle(hud, (330, y_pos - 12), (330 + bar_w, y_pos), c, -1)

            canvas = np.vstack([top_row, hud])
            if writer is not None:
                writer.write(canvas)

            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.root.after(0, self._update_canvas, img, f"{dec_lbl} | ventanas={len(vote_history)} | {last_inf_ms:.0f} ms")
            frame_idx += 1

        for _, cap in caps:
            cap.release()
        if writer is not None:
            writer.release()

        # Conclusión final consolidada
        votes = Counter([v[0] for v in vote_history if v[0] != "Analizando..."])
        if votes:
            total_v = sum(votes.values())
            known = Counter({k: v for k, v in votes.items() if k != "Desconocido"})
            if known:
                best_k, best_cnt = known.most_common(1)[0]
                if (best_cnt / total_v >= 0.12) and (best_cnt / sum(known.values()) >= 0.50):
                    final_id, win_cnt = best_k, best_cnt
                else:
                    final_id, win_cnt = "Desconocido", votes.get("Desconocido", 0)
            else:
                final_id, win_cnt = "Desconocido", votes.get("Desconocido", 0)
            summary_msg = (
                f"Identidad Final: {final_id}\n"
                f"Motor: {self.mode_var.get()}\n"
                f"Consenso: {win_cnt}/{total_v} ventanas\n"
                f"Votos: {dict(votes)}"
            )
        else:
            summary_msg = "No se acumularon suficientes ventanas de caminata válida."

        self.root.after(0, self._on_finished, summary_msg)

    def _update_canvas(self, img: ImageTk.PhotoImage, status_txt: str):
        self.video_label.imgtk = img
        self.video_label.config(image=img)
        self.status_var.set(status_txt)

    def _on_finished(self, summary_msg: str):
        self._stop_stream()
        self.status_var.set("Finalizado — " + summary_msg.splitlines()[0])
        messagebox.showinfo("Conclusión Final v2", summary_msg)

    def _on_close(self):
        self.running = False
        self.root.destroy()


def main():
    root = tk.Tk()
    GaitAppGUIv2(root)
    root.mainloop()


if __name__ == "__main__":
    main()
