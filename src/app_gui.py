"""Interfaz Gráfica Interactiva (GUI) para Reconocimiento de Marcha en Tiempo Real.

Permite elegir entre Cámara en Vivo (Webcam) o 1 a 3 videos (.mp4),
visualizar Silueta + Esqueleto + Estela + Top-3 en vivo, ajustar umbrales
en tiempo real, reiniciar caminatas y cerrar limpiamente con Conclusión Final.
"""

import pickle
import threading
import tkinter as tk
from collections import Counter, deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

from phase2_extract_and_visualize import align_silhouette_opengait, calc_angle_deg, draw_skeleton
from phase3_gait_engine import (
    GaitBaseInference,
    extract_skeleton_embedding,
    match_probe_against_gallery,
)
from phase4_realtime_sim import classify_with_unknown

CKPT_PATH = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\weights\GaitBase_DA-180000.pt")
GALLERY_DB_PATH = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\outputs\gallery_db.pkl")
OUTPUTS_DIR = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\outputs\visuals")


class GaitAppGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Reconocimiento de Marcha — Silueta + Esqueleto (31 Sujetos)")
        self.root.geometry("1280x760")
        self.root.configure(bg="#121417")

        self.running = False
        self.reset_requested = False
        self.worker_thread: threading.Thread | None = None
        self.video_sources: list[tuple[str, str]] = [("frente", "0")]

        self.min_score_var = tk.DoubleVar(value=0.950)
        self.min_margin_var = tk.DoubleVar(value=0.015)
        self.save_video_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Cargando modelos (GaitBase + YOLOv8 seg/pose)...")

        self.gait_model = None
        self.seg_model = None
        self.pose_model = None
        self.gallery_db = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        threading.Thread(target=self._load_models, daemon=True).start()

    def _build_ui(self):
        top = tk.Frame(self.root, bg="#1b1f24", pady=8, padx=12)
        top.pack(fill=tk.X)

        btn_cfg = {"bg": "#2b313a", "fg": "#f0f4f8", "relief": tk.FLAT, "padx": 10, "pady": 6, "font": ("Segoe UI", 9, "bold")}
        tk.Button(top, text="Usar Webcam (Cam 0)", command=self._select_webcam, **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Abrir Videos (.mp4)", command=self._select_videos, **btn_cfg).pack(side=tk.LEFT, padx=4)

        self.btn_start = tk.Button(
            top, text="▶ Iniciar", command=self._toggle_start,
            bg="#1f8b4c", fg="white", relief=tk.FLAT, padx=14, pady=6, font=("Segoe UI", 9, "bold"), state=tk.DISABLED
        )
        self.btn_start.pack(side=tk.LEFT, padx=8)

        tk.Button(
            top, text="↻ Reiniciar Caminata", command=self._request_reset,
            bg="#b87314", fg="white", relief=tk.FLAT, padx=10, pady=6, font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            top, text="■ Finalizar y Conclusión", command=self._stop_stream,
            bg="#b32d2e", fg="white", relief=tk.FLAT, padx=10, pady=6, font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=4)

        tk.Checkbutton(
            top, text="Guardar .mp4", variable=self.save_video_var,
            bg="#1b1f24", fg="#d0d7de", selectcolor="#121417", activebackground="#1b1f24"
        ).pack(side=tk.LEFT, padx=10)

        # Sliders de umbral
        sliders = tk.Frame(self.root, bg="#16191d", pady=5, padx=12)
        sliders.pack(fill=tk.X)

        tk.Label(sliders, text="Umbral Score:", bg="#16191d", fg="#c9d1d9", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Scale(
            sliders, from_=0.85, to=0.98, resolution=0.005, orient=tk.HORIZONTAL,
            variable=self.min_score_var, bg="#16191d", fg="#58a6ff", highlightthickness=0, length=180
        ).pack(side=tk.LEFT, padx=(4, 16))

        tk.Label(sliders, text="Umbral Margen:", bg="#16191d", fg="#c9d1d9", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Scale(
            sliders, from_=0.00, to=0.05, resolution=0.005, orient=tk.HORIZONTAL,
            variable=self.min_margin_var, bg="#16191d", fg="#58a6ff", highlightthickness=0, length=180
        ).pack(side=tk.LEFT, padx=(4, 16))

        self.source_lbl = tk.Label(sliders, text="Fuente: Webcam (Cam 0)", bg="#16191d", fg="#8b949e", font=("Segoe UI", 9))
        self.source_lbl.pack(side=tk.RIGHT)

        # Lienzo de video principal
        self.video_label = tk.Label(self.root, bg="#0d1117")
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=8)

        # Barra de estado inferior
        bottom = tk.Frame(self.root, bg="#1b1f24", pady=6, padx=12)
        bottom.pack(fill=tk.X)
        tk.Label(bottom, textvariable=self.status_var, bg="#1b1f24", fg="#7ee787", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)

    def _load_models(self):
        try:
            with open(GALLERY_DB_PATH, "rb") as f:
                self.gallery_db = pickle.load(f)
            self.gait_model = GaitBaseInference(CKPT_PATH)
            self.seg_model = YOLO("yolov8n-seg.pt")
            self.pose_model = YOLO("yolov8n-pose.pt")
            n_subjs = len(self.gallery_db)
            self.root.after(0, lambda: self._on_models_ready(n_subjs))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error cargando modelos", str(e)))

    def _on_models_ready(self, n_subjs: int):
        self.btn_start.config(state=tk.NORMAL)
        self.status_var.set(f"Listo ({n_subjs} sujetos en galería). Elige Webcam o Videos y presiona Iniciar.")

    def _select_webcam(self):
        self.video_sources = [("frente", "0")]
        self.source_lbl.config(text="Fuente: Webcam (Cam 0)")

    def _select_videos(self):
        files = filedialog.askopenfilenames(
            title="Selecciona 1 a 3 videos (.mp4)",
            filetypes=[("Videos MP4", "*.mp4"), ("Todos", "*.*")],
        )
        if not files:
            return
        files = list(files)[:3]
        sources = []
        for idx, fpath in enumerate(files):
            low = Path(fpath).name.lower()
            if "45_izq" in low:
                ang = "45_izq"
            elif "45_der" in low:
                ang = "45_der"
            elif "frente" in low:
                ang = "frente"
            else:
                ang = ["frente", "45_izq", "45_der"][idx]
            sources.append((ang, fpath))
        self.video_sources = sources
        names = ", ".join([f"{a}:{Path(p).name}" for a, p in sources])
        self.source_lbl.config(text=f"Fuente: {names}")

    def _toggle_start(self):
        if self.running:
            return
        self.running = True
        self.reset_requested = False
        self.btn_start.config(state=tk.DISABLED)
        self.status_var.set("Ejecutando reconocimiento en vivo...")
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.worker_thread.start()

    def _request_reset(self):
        self.reset_requested = True

    def _stop_stream(self):
        self.running = False

    def _run_loop(self):
        caps = []
        for angle, src in self.video_sources:
            cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW) if src.isdigit() else cv2.VideoCapture(src)
            if not cap.isOpened():
                self.root.after(0, lambda a=angle, s=src: messagebox.showerror("Error", f"No se pudo abrir {a}: {s}"))
                self.running = False
                self.root.after(0, lambda: self.btn_start.config(state=tk.NORMAL))
                return
            caps.append((angle, cap))

        main_angle, main_cap = caps[0]
        fps = main_cap.get(cv2.CAP_PROP_FPS) or 30.0
        cam_fps = {angle: (cap.get(cv2.CAP_PROP_FPS) or fps) for angle, cap in caps}
        cam_pos = {angle: 0 for angle, _ in caps}

        panel_w, panel_h = 400, 320
        total_w, total_h = panel_w * 3, panel_h + 125

        writer = None
        out_path = None
        if self.save_video_var.get():
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUTPUTS_DIR / "gui_session.mp4"
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (total_w, total_h))

        sil_buffers = {angle: deque(maxlen=60) for angle, _ in caps}
        kpt_buffers = {angle: deque(maxlen=60) for angle, _ in caps}
        l_trail = deque(maxlen=18)
        r_trail = deque(maxlen=18)
        cx_history = deque(maxlen=8)

        current_top3: list[tuple[str, float]] = []
        current_decision = ("Analizando...", 0.0, 0.0)
        vote_history: list[tuple[str, float, float]] = []
        frame_idx = 0

        while self.running:
            if self.reset_requested:
                for b in sil_buffers.values():
                    b.clear()
                for b in kpt_buffers.values():
                    b.clear()
                l_trail.clear()
                r_trail.clear()
                cx_history.clear()
                vote_history.clear()
                current_top3 = []
                current_decision = ("Analizando...", 0.0, 0.0)
                self.reset_requested = False

            ret, main_frm = main_cap.read()
            if not ret:
                break
            t_sec = frame_idx / fps
            frames = {main_angle: main_frm}
            for angle, cap in caps[1:]:
                target_idx = int(round(t_sec * cam_fps[angle]))
                while cam_pos[angle] <= target_idx:
                    r2, frm2 = cap.read()
                    cam_pos[angle] += 1
                    if r2:
                        frames[angle] = frm2

            mh, mw = main_frm.shape[:2]
            main_mask = np.zeros((mh, mw), dtype=np.uint8)
            main_box = None
            main_kpts = np.zeros((17, 3), dtype=np.float32)
            main_sil = None
            is_walking_now = True
            frame_data = {}

            for angle, frm in frames.items():
                fh, fw = frm.shape[:2]
                seg_res = self.seg_model.predict(frm, classes=[0], conf=0.45, imgsz=512, verbose=False)
                pose_res = self.pose_model.predict(frm, classes=[0], conf=0.45, imgsz=512, verbose=False)

                mask_full = np.zeros((fh, fw), dtype=np.uint8)
                box = None
                if seg_res and seg_res[0].masks is not None and len(seg_res[0].boxes) > 0:
                    boxes = seg_res[0].boxes.xyxy.cpu().numpy()
                    bi = int(np.argmax(boxes[:, 3] - boxes[:, 1]))
                    box = boxes[bi]
                    mraw = seg_res[0].masks.data[bi].cpu().numpy()
                    mres = cv2.resize(mraw, (fw, fh), interpolation=cv2.INTER_NEAREST)
                    mask_full = (mres > 0.5).astype(np.uint8) * 255

                aligned_sil = align_silhouette_opengait(mask_full)
                kpts = np.zeros((17, 3), dtype=np.float32)
                if pose_res and pose_res[0].keypoints is not None and len(pose_res[0].boxes) > 0:
                    pboxes = pose_res[0].boxes.xyxy.cpu().numpy()
                    pbi = int(np.argmax(pboxes[:, 3] - pboxes[:, 1]))
                    kpts = pose_res[0].keypoints.data[pbi].cpu().numpy()

                frame_data[angle] = (mask_full, box, kpts, aligned_sil)
                if angle == main_angle:
                    main_mask, main_box, main_kpts, main_sil = mask_full, box, kpts, aligned_sil
                    if box is not None:
                        cx = float((box[0] + box[2]) / 2.0)
                        bh = float(box[3] - box[1])
                        cx_history.append((t_sec, cx, bh))
                        if len(cx_history) >= 5:
                            dt = max(cx_history[-1][0] - cx_history[0][0], 1e-3)
                            vx = abs(cx_history[-1][1] - cx_history[0][1]) / dt
                            vh = abs(cx_history[-1][2] - cx_history[0][2]) / dt
                            is_walking_now = (vx >= 65.0) or (vh >= 18.0)
                    else:
                        is_walking_now = False

            if is_walking_now:
                for angle, (_, _, kpts, aligned_sil) in frame_data.items():
                    if aligned_sil is not None:
                        sil_buffers[angle].append(aligned_sil)
                    if np.any(kpts[:, 2] > 0.3):
                        kpt_buffers[angle].append(kpts)

            if is_walking_now and frame_idx > 0 and frame_idx % 12 == 0:
                probe_views = {}
                for angle in frames:
                    if len(sil_buffers[angle]) >= 15:
                        s_arr = np.stack(sil_buffers[angle], axis=0)
                        k_arr = (
                            np.stack(kpt_buffers[angle], axis=0)
                            if len(kpt_buffers[angle]) >= 10
                            else np.zeros((0, 17, 3), dtype=np.float32)
                        )
                        s_emb = self.gait_model.extract_embedding(s_arr)
                        k_emb = extract_skeleton_embedding(k_arr) if len(k_arr) >= 10 else None
                        if s_emb is not None or k_emb is not None:
                            probe_views[angle] = (s_emb, k_emb)
                if probe_views:
                    current_top3 = match_probe_against_gallery(probe_views, self.gallery_db, alpha_sil=0.85)[:3]
                    current_decision = classify_with_unknown(
                        current_top3,
                        min_score=float(self.min_score_var.get()),
                        min_margin=float(self.min_margin_var.get()),
                    )
                    vote_history.append(current_decision)

            # Render 3 paneles + HUD
            p1 = main_frm.copy()
            dec_label, dec_sc, dec_margin = current_decision
            box_color = (0, 165, 255) if dec_label == "Desconocido" else ((220, 220, 80) if dec_label == "Analizando..." else (0, 230, 140))
            if main_box is not None:
                x1, y1, x2, y2 = [int(v) for v in main_box]
                cv2.rectangle(p1, (x1, y1), (x2, y2), box_color, 2)
                lbl = f"{dec_label} ({dec_sc:.2f})" if dec_sc > 0 else dec_label
                cv2.putText(p1, lbl, (x1, max(28, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
            draw_skeleton(p1, main_kpts, thickness=2, radius=4)
            p1_r = cv2.resize(p1, (panel_w, panel_h))
            cv2.putText(p1_r, f"{'CAMINANDO' if is_walking_now else 'PAUSA'} | F:{frame_idx}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

            p2 = np.full((mh, mw, 3), 22, dtype=np.uint8)
            if main_kpts[15, 2] >= 0.3:
                l_trail.append((int(main_kpts[15, 0]), int(main_kpts[15, 1])))
            if main_kpts[16, 2] >= 0.3:
                r_trail.append((int(main_kpts[16, 0]), int(main_kpts[16, 1])))
            for i in range(1, len(l_trail)):
                cv2.line(p2, l_trail[i - 1], l_trail[i], (60, 190, 90), 2)
            for i in range(1, len(r_trail)):
                cv2.line(p2, r_trail[i - 1], r_trail[i], (40, 140, 230), 2)
            draw_skeleton(p2, main_kpts, thickness=4, radius=6)
            p2_r = cv2.resize(p2, (panel_w, panel_h))
            lk = calc_angle_deg(main_kpts[11], main_kpts[13], main_kpts[15])
            rk = calc_angle_deg(main_kpts[12], main_kpts[14], main_kpts[16])
            cv2.putText(p2_r, f"Esqueleto | Rodillas I:{lk:.0f} D:{rk:.0f}" if (lk and rk) else "Esqueleto & Estela", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 220, 90), 2)

            p3 = cv2.cvtColor(main_mask, cv2.COLOR_GRAY2BGR)
            p3_r = cv2.resize(p3, (panel_w, panel_h))
            if main_sil is not None:
                sil_big = cv2.resize(main_sil, (88, 128), interpolation=cv2.INTER_NEAREST)
                p3_r[12:140, panel_w - 100 : panel_w - 12] = cv2.cvtColor(sil_big, cv2.COLOR_GRAY2BGR)
            cv2.putText(p3_r, "Silueta + OpenGait 64x44", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

            top_row = np.hstack([p1_r, p2_r, p3_r])
            hud = np.full((125, total_w, 3), 18, dtype=np.uint8)
            cv2.putText(hud, f"Veredicto en vivo: {dec_label} (margen={dec_margin:+.3f})", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, box_color, 2)
            for rank_i, (name, sc) in enumerate(current_top3):
                y_pos = 58 + rank_i * 22
                bar_w = int(np.clip(sc, 0.0, 1.0) * 320)
                c = box_color if (rank_i == 0 and dec_label != "Desconocido") else (170, 170, 170)
                cv2.putText(hud, f"#{rank_i+1} {name:20s} {sc:.3f}", (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
                cv2.rectangle(hud, (360, y_pos - 14), (360 + bar_w, y_pos), c, -1)

            canvas = np.vstack([top_row, hud])
            if writer is not None:
                writer.write(canvas)

            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.root.after(0, lambda im=img_tk: self._update_frame(im))
            frame_idx += 1

        # Conclusión final
        votes = Counter([v[0] for v in vote_history if v[0] != "Analizando..."])
        if votes:
            total_votes = sum(votes.values())
            known_votes = Counter({k: v for k, v in votes.items() if k != "Desconocido"})
            if known_votes:
                best_k, best_cnt = known_votes.most_common(1)[0]
                if best_cnt / total_votes >= 0.25:
                    final_id, top_v = best_k, best_cnt
                else:
                    final_id, top_v = "Desconocido", votes.get("Desconocido", 0)
            else:
                final_id, top_v = "Desconocido", votes.get("Desconocido", 0)
            avg_sc = float(np.mean([v[1] for v in vote_history if v[0] == final_id]))
            avg_mg = float(np.mean([v[2] for v in vote_history if v[0] == final_id]))
        else:
            final_id, top_v, total_votes, avg_sc, avg_mg = "Desconocido", 0, 0, 0.0, 0.0

        for _, cap in caps:
            cap.release()
        if writer is not None:
            writer.release()

        self.running = False
        summary = (
            f"Identidad Final: {final_id}\n"
            f"Consenso: {top_v}/{total_votes} ventanas\n"
            f"Score promedio: {avg_sc:.3f} | Margen: {avg_mg:+.3f}\n"
            f"Votos: {dict(votes)}"
        )
        if out_path:
            summary += f"\n\nVideo guardado en:\n{out_path}"
        self.root.after(0, lambda: self._on_finished(summary, final_id))

    def _update_frame(self, img_tk):
        self.video_label.imgtk = img_tk
        self.video_label.configure(image=img_tk)

    def _on_finished(self, summary: str, final_id: str):
        self.btn_start.config(state=tk.NORMAL)
        self.status_var.set(f"Sesión finalizada -> Conclusión Final: {final_id}")
        messagebox.showinfo("Conclusión Final", summary)

    def _on_close(self):
        self.running = False
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = GaitAppGUI(root)
    root.mainloop()

