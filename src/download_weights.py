"""Descarga los pesos preentrenados de OpenGait (GaitBase_DA GREW) a `weights/`."""

from pathlib import Path
import requests

WEIGHTS_URL = (
    "https://huggingface.co/opengait/OpenGait/resolve/main/"
    "GREW/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-180000.pt"
)
OUT_PATH = Path(__file__).resolve().parent.parent / "weights" / "GaitBase_DA-180000.pt"


def main():
    if OUT_PATH.exists():
        print(f"Ya existe: {OUT_PATH}")
        return
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {WEIGHTS_URL} -> {OUT_PATH} ...")
    with requests.get(WEIGHTS_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        done = 0
        with open(OUT_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                print(f"\r  {done / (1024 * 1024):.1f} MB descargados", end="", flush=True)
    print(f"\nListo: {OUT_PATH}")


if __name__ == "__main__":
    main()
