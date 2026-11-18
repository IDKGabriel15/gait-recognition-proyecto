"""Fase 4: Reconocimiento en Tiempo Real sobre Video Completo (1 a 3 cámaras).

Procesa videos completos (~20 s con tiempos muertos incluidos) de 1, 2 o 3 cámaras
sincronizadas por tiempo real:
  - Detecta movimiento activo en vivo (ignora pausas/giros en los extremos).
  - Muestra silueta 64x44 + esqueleto + estela de pasos + Top-3 en tiempo real.
  - Aplica umbral de confianza y margen para clasificar como `Desconocido`.
  - Al finalizar el video, emite e imprime la CONCLUSIÓN FINAL consolidada.
"""

import argparse
import pickle
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from phase2_extract_and_visualize import align_silhouette_opengait, calc_angle_deg, draw_skeleton
from phase3_gait_engine import (
    GaitBaseInference,
    extract_skeleton_embedding,
    match_probe_against_gallery,
)

CKPT_PATH = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\weights\GaitBase_DA-180000.pt")
GALLERY_DB_PATH = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\outputs\gallery_db.pkl")


def classify_with_unknown(
    ranked: list[tuple[str, float]],
    min_score: float = 0.950,
    min_margin: float = 0.015,
) -> tuple[str, float, float]:
    """Determina si el Top-1 es válido o debe marcarse como Desconocido."""
    if not ranked:
        return "Analizando...", 0.0, 0.0
    top1_name, top1_sc = ranked[0]
    top2_sc = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top1_sc - top2_sc
    if top1_sc < min_score or (len(ranked) > 1 and margin < min_margin):
        return "Desconocido", top1_sc, margin
    return top1_name, top1_sc, margin


def run_realtime_recognition(
    sources: list[tuple[str, str]],
    out_video: Path | None = None,
    show: bool = False,
    window_frames: int = 60,
    eval_interval: int = 15,
    alpha_sil: float = 0.85,
    min_score: float = 0.950,
    min_margin: float = 0.015,
):
    with open(GALLERY_DB_PATH, "rb") as f:
        gallery_db = pickle.load(f)

    gait_model = GaitBaseInference(CKPT_PATH)
    seg_model = YOLO("yolov8n-seg.pt")
    pose_model = YOLO("yolov8n-pose.pt")

    caps = []
    for angle, src in sources:
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW) if src.isdigit() else cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir fuente {angle}: {src}")
        caps.append((angle, cap))

    main_angle, main_cap = caps[0]
    fps = main_cap.get(cv2.CAP_PROP_FPS) or 30.0
    cam_fps = {angle: (cap.get(cv2.CAP_PROP_FPS) or fps) for angle, cap in caps}
    cam_pos = {angle: 0 for angle, _ in caps}

    panel_w, panel_h = 420, 340
    total_w, total_h = panel_w * 3, panel_h + 125

    writer = None
    if out_video is not None:
        out_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (total_w, total_h)
        )

    sil_buffers = {angle: deque(maxlen=window_frames) for angle, _ in caps}
    kpt_buffers = {angle: deque(maxlen=window_frames) for angle, _ in caps}
    l_trail = deque(maxlen=18)
    r_trail = deque(maxlen=18)
    cx_history = deque(maxlen=8)

    current_top3: list[tuple[str, float]] = []
    current_decision = ("Analizando...", 0.0, 0.0)
    vote_history: list[tuple[str, float, float]] = []
    score_accum: dict[str, list[float]] = defaultdict(list)

    frame_idx = 0
    while True:
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

        # Detectar si la persona está caminando activamente en main_angle
        is_walking_now = True
        frame_data = {}

        for angle, frm in frames.items():
            fh, fw = frm.shape[:2]
            seg_res = seg_model.predict(frm, classes=[0], conf=0.45, imgsz=512, verbose=False)
            pose_res = pose_model.predict(frm, classes=[0], conf=0.45, imgsz=512, verbose=False)

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

        # Solo acumular siluetas/esqueleto cuando el sujeto camina (ignorar tiempo muerto)
        if is_walking_now:
            for angle, (_, _, kpts, aligned_sil) in frame_data.items():
                if aligned_sil is not None:
                    sil_buffers[angle].append(aligned_sil)
                if np.any(kpts[:, 2] > 0.3):
                    kpt_buffers[angle].append(kpts)

        # Evaluar cada `eval_interval` frames si hay caminata suficiente
        if is_walking_now and frame_idx > 0 and frame_idx % eval_interval == 0:
            probe_views = {}
            for angle in frames:
                if len(sil_buffers[angle]) >= 15:
                    s_arr = np.stack(sil_buffers[angle], axis=0)
                    k_arr = (
                        np.stack(kpt_buffers[angle], axis=0)
                        if len(kpt_buffers[angle]) >= 10
                        else np.zeros((0, 17, 3), dtype=np.float32)
                    )
                    s_emb = gait_model.extract_embedding(s_arr)
                    k_emb = extract_skeleton_embedding(k_arr) if len(k_arr) >= 10 else None
                    if s_emb is not None or k_emb is not None:
                        probe_views[angle] = (s_emb, k_emb)
            if probe_views:
                current_top3 = match_probe_against_gallery(
                    probe_views, gallery_db, alpha_sil=alpha_sil
                )[:3]
                current_decision = classify_with_unknown(
                    current_top3, min_score=min_score, min_margin=min_margin
                )
                vote_history.append(current_decision)
                for nm, sc in current_top3:
                    score_accum[nm].append(sc)

        # Renderizado de los 3 paneles
        p1 = main_frm.copy()
        dec_label, dec_sc, dec_margin = current_decision
        if dec_label == "Desconocido":
            box_color = (0, 165, 255)  # Naranja
        elif dec_label == "Analizando...":
            box_color = (220, 220, 80)
        else:
            box_color = (0, 230, 140)  # Verde

        status_txt = "CAMINANDO" if is_walking_now else "PAUSA / GIRO"
        if main_box is not None:
            x1, y1, x2, y2 = [int(v) for v in main_box]
            cv2.rectangle(p1, (x1, y1), (x2, y2), box_color, 2)
            lbl = f"{dec_label} ({dec_sc:.2f})" if dec_sc > 0 else dec_label
            cv2.putText(
                p1, lbl, (x1, max(28, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.85, box_color, 2
            )
        draw_skeleton(p1, main_kpts, thickness=2, radius=4)
        p1_r = cv2.resize(p1, (panel_w, panel_h))
        cv2.putText(p1_r, f"t={t_sec:.1f}s | {status_txt}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

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
        cv2.putText(
            p2_r,
            f"Esqueleto | Rodillas I:{lk:.0f} D:{rk:.0f}" if (lk and rk) else "Esqueleto & Estela",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 220, 90), 2
        )

        p3 = cv2.cvtColor(main_mask, cv2.COLOR_GRAY2BGR)
        p3_r = cv2.resize(p3, (panel_w, panel_h))
        if main_sil is not None:
            sil_big = cv2.resize(main_sil, (88, 128), interpolation=cv2.INTER_NEAREST)
            p3_r[12:140, panel_w - 100 : panel_w - 12] = cv2.cvtColor(sil_big, cv2.COLOR_GRAY2BGR)
        cv2.putText(p3_r, "Silueta + OpenGait 64x44", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

        top_row = np.hstack([p1_r, p2_r, p3_r])

        # Barra inferior HUD
        hud = np.full((125, total_w, 3), 18, dtype=np.uint8)
        angles_str = ", ".join([a for a, _ in caps])
        cv2.putText(
            hud,
            f"Tiempo Real | Camaras: [{angles_str}] | Veredicto actual: {dec_label} (margen={dec_margin:+.3f})",
            (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2,
        )
        for rank_i, (name, sc) in enumerate(current_top3):
            y_pos = 58 + rank_i * 22
            bar_w = int(np.clip(sc, 0.0, 1.0) * 320)
            c = box_color if (rank_i == 0 and dec_label != "Desconocido") else (170, 170, 170)
            cv2.putText(hud, f"#{rank_i+1} {name:20s} {sc:.3f}", (18, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
            cv2.rectangle(hud, (360, y_pos - 14), (360 + bar_w, y_pos), c, -1)

        canvas = np.vstack([top_row, hud])
        if writer is not None:
            writer.write(canvas)
        if show:
            cv2.imshow("Gait Recognition - Live", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                for buf in sil_buffers.values():
                    buf.clear()
                for buf in kpt_buffers.values():
                    buf.clear()
                l_trail.clear()
                r_trail.clear()
                vote_history.clear()
                current_top3 = []
                current_decision = ("Analizando...", 0.0, 0.0)

        frame_idx += 1

    # Conclusión final consolidada al terminar el video
    votes = Counter([v[0] for v in vote_history if v[0] != "Analizando..."])
    if votes:
        total_votes = sum(votes.values())
        known_votes = Counter({k: v for k, v in votes.items() if k != "Desconocido"})
        if known_votes:
            best_known, best_k_cnt = known_votes.most_common(1)[0]
            if best_k_cnt / total_votes >= 0.25:
                final_identity, top_votes = best_known, best_k_cnt
            else:
                final_identity, top_votes = "Desconocido", votes.get("Desconocido", 0)
        else:
            final_identity, top_votes = "Desconocido", votes.get("Desconocido", 0)
        avg_sc = float(np.mean([v[1] for v in vote_history if v[0] == final_identity]))
        avg_mg = float(np.mean([v[2] for v in vote_history if v[0] == final_identity]))
    else:
        final_identity, top_votes, total_votes, avg_sc, avg_mg = "Desconocido", 0, 0, 0.0, 0.0

    print("\n================ CONCLUSION FINAL ================", flush=True)
    print(f"Identidad Final : {final_identity}", flush=True)
    print(f"Votos por ventana: {dict(votes)} ({top_votes}/{total_votes})", flush=True)
    print(f"Score promedio  : {avg_sc:.3f} | Margen promedio: {avg_mg:+.3f}", flush=True)
    print("==================================================\n", flush=True)

    # Renderizar tarjeta de conclusión final (2 segundos) al final del video
    if writer is not None:
        end_card = np.full((total_h, total_w, 3), 20, dtype=np.uint8)
        v_color = (0, 165, 255) if final_identity == "Desconocido" else (0, 230, 140)
        cv2.putText(end_card, "CONCLUSION FINAL DEL VIDEO", (80, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (220, 220, 220), 2)
        cv2.putText(end_card, f"Identidad: {final_identity}", (80, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, v_color, 3)
        cv2.putText(
            end_card,
            f"Consenso: {top_votes}/{total_votes} ventanas | Score: {avg_sc:.3f} | Margen: {avg_mg:+.3f}",
            (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2,
        )
        cv2.putText(
            end_card,
            f"Votos detallados: {dict(votes)}",
            (80, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 170, 170), 2,
        )
        for _ in range(int(round(fps * 2))):
            writer.write(end_card)

    for _, cap in caps:
        cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Fase 4: Reconocimiento en tiempo real")
    parser.add_argument(
        "--source",
        nargs="+",
        required=True,
        help="Pares angulo=ruta_video_o_camara (ej. frente=path.mp4 45_izq=path2.mp4)",
    )
    parser.add_argument("--out-video", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--min-score", type=float, default=0.950)
    parser.add_argument("--min-margin", type=float, default=0.015)
    args = parser.parse_args()

    sources = []
    for item in args.source:
        ang, path = item.split("=", 1)
        sources.append((ang, path))

    run_realtime_recognition(
        sources,
        out_video=Path(args.out_video) if args.out_video else None,
        show=args.show,
        min_score=args.min_score,
        min_margin=args.min_margin,
    )


if __name__ == "__main__":
    main()
