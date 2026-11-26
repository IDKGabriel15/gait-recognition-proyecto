"""Descarga checkpoints SOTA oficiales desde HuggingFace `opengait/OpenGait`.

Rutas verificadas en `opengait/OpenGait`:
  1. `GREW/DeepGaitV2/DeepGaitV2/checkpoints/DeepGaitV2-180000.pt`
  2. `GREW/DeepGaitV2/SkeletonGait/checkpoints/SkeletonGait-180000.pt`
  3. `GREW/SkeletonGaitPP/SkeletonGaitPP/checkpoints/SkeletonGaitPP-180000.pt`
"""

from pathlib import Path
import requests

WEIGHTS_DIR = Path(r"c:\Users\angel.roman\Downloads\ProyectoIA\MetodoAlternativo\weights")
HF_RAW_BASE = "https://huggingface.co/opengait/OpenGait/resolve/main"

EXACT_TARGETS = [
    ("DeepGaitV2_GREW.pt", "GREW/DeepGaitV2/DeepGaitV2/checkpoints/DeepGaitV2-180000.pt"),
    ("SkeletonGait_GREW.pt", "GREW/DeepGaitV2/SkeletonGait/checkpoints/SkeletonGait-180000.pt"),
    ("SkeletonGaitPP_GREW.pt", "GREW/SkeletonGaitPP/SkeletonGaitPP/checkpoints/SkeletonGaitPP-180000.pt"),
]


def download_file(hf_path: str, local_name: str) -> Path:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = WEIGHTS_DIR / local_name
    if out_path.exists() and out_path.stat().st_size > 100_000_000:
        print(f"[CACHED] {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
        return out_path

    url = f"{HF_RAW_BASE}/{hf_path}"
    print(f"[DOWNLOAD] {hf_path} -> {out_path.name} ...", flush=True)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        tmp_path = out_path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total > 0 and (done % (50 * 1024 * 1024) < 1024 * 1024 or done >= total):
                        pct = 100.0 * done / total
                        print(f"  {local_name}: {done / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", flush=True)
        tmp_path.replace(out_path)
    print(f"[OK] Guardado: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return out_path


def main():
    for local_name, hf_path in EXACT_TARGETS:
        download_file(hf_path, local_name)
    print("\nTodos los checkpoints SOTA GREW listos en:", WEIGHTS_DIR, flush=True)


if __name__ == "__main__":
    main()
