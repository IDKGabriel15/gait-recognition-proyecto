# Reconocimiento de Marcha (Gait Recognition) — Silueta + Esqueleto

Sistema modular de reconocimiento de personas por su forma de caminar (**Enfoque 1 preentrenado** con `OpenGait / GaitBase_DA` en GREW + descriptor biomecánico de esqueleto 2D con `YOLOv8-pose` + segmentación `YOLOv8-seg`), con soporte flexible de **1 a 3 cámaras** y visualización gráfica en tiempo real.

## Estructura del Proyecto

- `src/phase0_manifest.py` — Consolida `dataset/` en `configs/dataset_manifest.json` (`gallery`, `probe`, `external_test`).
- `src/phase1_trim.py` — Recorta automáticamente el tiempo muerto (pausas/giros) en pasadas limpias sincronizadas entre 1–3 cámaras (`dataset_trimmed/`).
- `src/phase2_extract_and_visualize.py` — Extrae siluetas binarias `64×44` (OpenGait) + 17 puntos clave COCO y renderiza vista de 3 paneles.
- `src/phase3_gait_engine.py` y `src/phase3_evaluate.py` — Motor `GaitBase_DA` + esqueleto centrado y evaluación Top-1 / Top-5 (mismo ángulo y Cross-View).
- `src/phase4_realtime_sim.py` — Simulador/demo en tiempo real de 1 a 3 cámaras con detección de `"Desconocido"` y conclusión final.
- `src/download_weights.py` — Descarga `weights/GaitBase_DA-180000.pt` (339 MB) desde Hugging Face.
- `outputs/gallery_db.pkl` — Base de datos de galería actual (9 sujetos, 404 secuencias).

## Instalación rápida

```powershell
# 1. Crear entorno Python 3.11 (requerido por OpenGait/PyTorch)
python -m uv venv --python 3.11 .venv
python -m uv pip install --python .venv\Scripts\python.exe -r requirements.txt tensorboard

# 2. Clonar OpenGait (si no existe third_party/OpenGait)
git clone --depth 1 https://github.com/ShiqiYu/OpenGait.git third_party/OpenGait

# 3. Descargar pesos preentrenados de GaitBase (GREW)
.venv\Scripts\python.exe src\download_weights.py
```

## Uso rápido

```powershell
# Evaluar galería y probes actuales
.venv\Scripts\python.exe src\phase3_evaluate.py --alpha-sil 0.85

# Inferencia gráfica en vivo sobre 1 a 3 videos (o webcam frente=0)
.venv\Scripts\python.exe src\phase4_realtime_sim.py --source frente=ruta_frente.mp4 45_izq=ruta_45izq.mp4 --out-video outputs\visuals\demo.mp4
```

## Exportación móvil

Después de cambiar el modelo o la galería, vuelve a generar los assets de Android. Primero regenera `outputs/gallery_db.pkl` con `src/phase3_evaluate.py`; después ejecuta `src/export_mobile.py` (o `src/build_onnx_mobile_gallery.py`). La exportación produce embeddings centrados y normalizados por cada una de las 16 franjas, junto con `gallery_sil_mean.bin`; ese archivo debe distribuirse con `gallery_sil.bin`, `gallery_index.json` y `gaitbase_mobile.onnx`. El exportador rechaza galerías pickle antiguas para evitar mezclar órdenes de características.

Para ejecutar el procesamiento en la nube, abre el [notebook de Google Colab](./notebooks/gait_pipeline_colab.ipynb). Usa los pesos preentrenados de GaitBase; no realiza fine-tuning.

