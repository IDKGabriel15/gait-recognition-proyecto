"""Genera galería móvil 100% consistente con yolov8n-seg-512.onnx + gaitbase_mobile.onnx."""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

from phase2_extract_and_visualize import align_silhouette_opengait

TRIMMED_ROOT = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\dataset_trimmed\subjects")
MOBILE_ASSETS = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\mobile_assets")
ANDROID_ASSETS = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\android_app\app\src\main\assets")


def main():
    seg = YOLO(str(MOBILE_ASSETS / "yolov8n-seg-512.onnx"), task="segment")
    gb = ort.InferenceSession(str(MOBILE_ASSETS / "gaitbase_mobile.onnx"), providers=["CPUExecutionProvider"])

    entries_meta = []
    sil_vecs = []

    for subj_dir in sorted(TRIMMED_ROOT.iterdir()):
        if not subj_dir.is_dir():
            continue
        subj = subj_dir.name
        takes_done = 0
        for take_dir in sorted(subj_dir.iterdir()):
            if not take_dir.is_dir():
                continue
            # Usar toma_01..toma_07 (máx 2 tomas por sujeto para galería móvil rápida y ligera)
            try:
                tnum = int(take_dir.name.split("_")[-1])
            except ValueError:
                tnum = 99
            if tnum > 7 or takes_done >= 2:
                continue
            pass_dir = take_dir / "pass_01"
            if not pass_dir.exists():
                continue
            takes_done += 1
            for vid in sorted(pass_dir.glob("*.mp4")):
                angle = vid.stem
                cap = cv2.VideoCapture(str(vid))
                sils = []
                while True:
                    ret, f = cap.read()
                    if not ret:
                        break
                    sr = seg.predict(f, classes=[0], conf=0.45, imgsz=512, verbose=False)
                    if sr and sr[0].masks is not None and len(sr[0].boxes) > 0:
                        bi = int((sr[0].boxes.xyxy[:, 3] - sr[0].boxes.xyxy[:, 1]).argmax())
                        mraw = sr[0].masks.data[bi].cpu().numpy()
                        mres = cv2.resize(mraw, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_NEAREST)
                        al = align_silhouette_opengait((mres > 0.5).astype(np.uint8) * 255)
                        if al is not None:
                            sils.append(al)
                cap.release()
                if len(sils) < 5:
                    continue
                x = (np.stack(sils).astype(np.float32) / 255.0)[None, None, :, :, :]
                emb = gb.run(None, {"sil_seq": x})[0][0].astype(np.float32)
                entries_meta.append({"s": subj, "a": angle, "si": len(sil_vecs), "ki": -1})
                sil_vecs.append(emb)
        print(f"[OK] {subj}: {takes_done} tomas -> total {len(sil_vecs)} vectores", flush=True)

    ANDROID_ASSETS.mkdir(parents=True, exist_ok=True)
    sil_mat = np.stack(sil_vecs, axis=0).astype(np.float32)
    (ANDROID_ASSETS / "gallery_sil.bin").write_bytes(sil_mat.tobytes())
    (ANDROID_ASSETS / "gallery_index.json").write_text(
        json.dumps({"num_entries": len(entries_meta), "entries": entries_meta}, ensure_ascii=False),
        encoding="utf-8",
    )
    for fn in ("gaitbase_mobile.onnx", "yolov8n-seg-512.onnx", "yolov8n-pose-512.onnx"):
        shutil.copy2(MOBILE_ASSETS / fn, ANDROID_ASSETS / fn)

    print(f"\nAssets Android listos en {ANDROID_ASSETS} ({len(sil_vecs)} vectores)")


if __name__ == "__main__":
    main()

