"""Fase 0: Generación del manifiesto unificado del dataset de marcha.

Escanea `dataset/`, consolida nombres duplicados sin modificar las carpetas
originales, separa los sujetos `*Prueba` y divide las tomas en `gallery`
(toma_01..toma_07) y `probe` (toma_08+).
"""

import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

DATASET_ROOT = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\dataset")
OUTPUT_MANIFEST = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\configs\dataset_manifest.json")

SUBJECT_ALIASES = {
    "Jose_Gastelum": "JoseGastelum",
    "Jose_Gatelum": "JoseGastelum",
    "Irvin_Ramirez": "IrvingRamirez",
    "IrvingRamirez": "IrvingRamirez",
}

EXTERNAL_TEST_FOLDERS = {"AlejandroPrueba", "DanielPrueba", "JoelPrueba"}
CALIBRATION_FOLDER = "calibracion_referencia"
GALLERY_MAX_TAKE_NUM = 7


def extract_take_number(take_name: str) -> int:
    match = re.search(r"(\d+)", take_name)
    return int(match.group(1)) if match else 999


def detect_angle(filename: str) -> str | None:
    lower = filename.lower()
    if "mosaico" in lower:
        return None
    for angle in ("frente", "45_der", "45_izq"):
        if angle in lower:
            return angle
    return None


def scan_take(take_dir: Path) -> dict | None:
    mp4s = sorted(take_dir.glob("*.mp4"))
    videos = {}
    for mp4 in mp4s:
        angle = detect_angle(mp4.name)
        if angle is None:
            continue
        ts_csv = mp4.with_name(mp4.stem + "_timestamps.csv")
        videos[angle] = {
            "video_path": str(mp4),
            "timestamps_csv": str(ts_csv) if ts_csv.exists() else None,
        }

    if not videos:
        return None

    sync_path = take_dir / "sync_timestamps.json"
    sync_info = {}
    if sync_path.exists():
        with open(sync_path, "r", encoding="utf-8") as f:
            sync_data = json.load(f)
        for angle, cam_meta in sync_data.get("cameras", {}).items():
            if angle in videos:
                frames = cam_meta.get("frames", 0)
                dur = cam_meta.get("duration_sec", 0.0)
                videos[angle]["frames"] = frames
                videos[angle]["duration_sec"] = dur
                videos[angle]["fps"] = round(frames / dur, 2) if dur > 0 else 0.0
        sync_info = {
            "max_start_diff_ms": sync_data.get("max_start_diff_ms"),
            "cameras_received": sync_data.get("cameras_received", list(videos.keys())),
        }

    return {
        "take_dir": str(take_dir),
        "take_name": take_dir.name,
        "take_num": extract_take_number(take_dir.name),
        "angles": sorted(videos.keys()),
        "videos": videos,
        "sync": sync_info,
    }


def build_manifest() -> dict:
    subjects_takes = defaultdict(list)
    external_tests = defaultdict(list)

    for entry in sorted(DATASET_ROOT.iterdir()):
        if not entry.is_dir() or entry.name == CALIBRATION_FOLDER:
            continue

        is_external = entry.name in EXTERNAL_TEST_FOLDERS
        canonical_name = SUBJECT_ALIASES.get(entry.name, entry.name)

        for take_dir in sorted(entry.iterdir()):
            if not take_dir.is_dir():
                continue
            take_info = scan_take(take_dir)
            if take_info is None:
                continue
            take_info["source_folder"] = entry.name
            if is_external:
                external_tests[canonical_name].append(take_info)
            else:
                subjects_takes[canonical_name].append(take_info)

    subjects_out = {}
    total_gallery_takes = 0
    total_probe_takes = 0

    for idx, subj in enumerate(sorted(subjects_takes.keys()), start=1):
        takes = sorted(subjects_takes[subj], key=lambda t: (t["take_num"], t["take_name"]))
        # Re-index takes cleanly if merged from aliases while preserving take_name
        gallery = [t for t in takes if t["take_num"] <= GALLERY_MAX_TAKE_NUM]
        probe = [t for t in takes if t["take_num"] > GALLERY_MAX_TAKE_NUM]
        total_gallery_takes += len(gallery)
        total_probe_takes += len(probe)
        subjects_out[subj] = {
            "subject_id": f"{idx:03d}",
            "total_takes": len(takes),
            "gallery": gallery,
            "probe": probe,
        }

    cal_dir = DATASET_ROOT / CALIBRATION_FOLDER
    medidas_path = cal_dir / "medidas_fisicas.json"
    calibration = {}
    if medidas_path.exists():
        with open(medidas_path, "r", encoding="utf-8") as f:
            calibration = json.load(f)
    calibration["calibration_dir"] = str(cal_dir)

    manifest = {
        "dataset_root": str(DATASET_ROOT),
        "calibration": calibration,
        "summary": {
            "num_subjects": len(subjects_out),
            "total_gallery_takes": total_gallery_takes,
            "total_probe_takes": total_probe_takes,
            "num_external_test_subjects": len(external_tests),
            "total_external_test_takes": sum(len(v) for v in external_tests.values()),
        },
        "subjects": subjects_out,
        "external_test": dict(sorted(external_tests.items())),
    }

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


if __name__ == "__main__":
    m = build_manifest()
    print("Manifiesto generado en:", OUTPUT_MANIFEST)
    print(json.dumps(m["summary"], indent=2, ensure_ascii=False))

