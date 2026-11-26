# Plan de Implementación v2 — Método Alternativo (`MetodoAlternativo/`)

> **Regla de aislamiento:** Este plan y todo su código viven 100% dentro de `MetodoAlternativo/`. Lee en modo **solo lectura** `configs/dataset_manifest.json`, `dataset_trimmed/`, `outputs/features/` y los videos no controlados de la raíz, sin modificar ni un solo archivo del Plan 1 (`src/`, `outputs/gallery_db.pkl`, `android_app/` ni el plan original).

---

## 1. Diagnóstico del Plan 1 (Qué se conserva vs. Qué se descarta)

### Conservado (Reutilización en solo lectura)
- **Manifiesto consolidado y partición estricta sin fuga:** `configs/dataset_manifest.json` (31 sujetos, 215 tomas Gallery `toma_01..07`, 98 tomas Probe `toma_08..11`, 3 externos `*Prueba` con cambio de ropa real).
- **Features ya extraídas:** `outputs/features/` (`silhouettes.npy` 64×44 y `keypoints.npy` T×17×3 COCO de 1,290 secuencias Gallery y 201 pasadas Probe).
- **Principio de invarianza parcial por franjas:** Centrado por franja respecto a la galería + agregación Top-K simétrica.
- **Evaluación Open-Set** (Leave-Subjects-Out) y lecciones de exportación ONNX móvil.

### Descartado / Reemplazado
1. **`GaitBase` (2D ResNet9 + Temporal MaxPool):** Destruye el orden temporal de la marcha (60.1% Rank-1 en GREW; 55.7% en nuestro `cross_frente`). → **Reemplazado por backbones Pseudo-3D (P3D): `SkeletonGait++`, `DeepGaitV2-P3D` y `SkeletonGait`.**
2. **Descriptor de esqueleto artesanal `99-D` en fusión tardía:** Estadísticos manuales sin dinámica temporal ni aprendizaje métrico. → **Reemplazado por Mapas de Calor de Esqueleto (`2×64×44`: articulaciones + extremidades) con `AttentionFusion` temprana (`SkeletonGait++`).**
3. **Doble inferencia YOLO en frame completo + `argmax(heights)` sin tracker:** Doble costo en CPU y falla con 2+ personas. → **Reemplazado por `ByteTrack` multi-persona + `ROI-Crop Pose` + ONNX.**
4. **Sin filtro de calidad de frames + detector de marcha solo por `|vx|`:** Frames cortados contaminan el `MaxPool` y `|vx|` falla en caminata frontal/diagonal (videos verticales de celular). → **Reemplazado por `Frame Quality Gate` + detector cinemático de oscilación de tobillos.**
5. **Matching 1-vs-1 por `Max` + bonus fijo `+0.02` por ángulo + bucle Python:** Asume conocer el ángulo de cámara, es vulnerable a *outliers* y recalcula la media en cada llamada. → **Reemplazado por matching matricial vectorizado (<1 ms), `TTA Horizontal Flip`, `Top-K Mean / Prototipos` agnósticos al ángulo y `AS-Norm`.**
6. **Zero-Shot puro sin adaptación de dominio:** → **Evaluación estricta (Leave-Subjects-Out) de `PCA-Whitening` y proyector métrico ligero.**

---

## 2. Arquitectura en `MetodoAlternativo/`

```text
MetodoAlternativo/
├── PLAN_IMPLEMENTACION.md         # Este documento
├── configs/
│   └── config_v2.yaml             # Hiperparámetros y rutas centralizadas
├── weights/                       # Checkpoints oficiales (HuggingFace opengait/OpenGait)
├── src/
│   ├── download_v2_weights.py     # Descarga verificada de SkeletonGait++, DeepGaitV2-P3D, SkeletonGait
│   ├── heatmap_generator.py       # Generador de Skeleton Maps (2x64x44) desde COCO-17
│   ├── models_v2.py               # Loaders CPU limpios (sin DDP) para los modelos P3D
│   ├── quality_and_motion.py      # Frame Quality Gate + detector de marcha invariante a perspectiva
│   ├── matcher_v2.py              # Matching vectorizado + TTA Flip + Top-K + AS-Norm
│   ├── domain_adapter.py          # PCA-Whitening y Metric Adapter ligero (Leave-Subjects-Out)
│   ├── tracker_extractor.py       # ByteTrack multi-persona + ROI-Crop Pose
│   ├── export_v2_onnx.py          # Exportación ONNX FP32 + Cuantización INT8
│   └── realtime_v2.py             # Inferencia en vivo / videos no controlados multi-persona
├── eval/
│   └── benchmark_ab.py            # Comparativa A/B directa: Plan 1 vs Plan 2
└── outputs/                       # Heatmaps, galerías v2, reportes JSON/MD y videos demo
```

---

## 3. Fases de Ejecución

- **Fase 0:** Estructura, `config_v2.yaml` y descarga de checkpoints SOTA desde `opengait/OpenGait`.
- **Fase 1:** `heatmap_generator.py` (`2×64×44`), `quality_and_motion.py` y `models_v2.py` con verificación en CPU.
- **Fase 2:** `matcher_v2.py` (vectorizado, TTA Flip, Top-K Mean, AS-Norm) y `domain_adapter.py` (PCA-Whitening + Adapter).
- **Fase 3:** `eval/benchmark_ab.py` comparando Plan 1 vs Plan 2 en: `all_available`, `only_*`, `cross_*`, `*Prueba` (cambio de ropa), `Leave-5-Subjects-Out` (TAR/FAR/EER/ROC-AUC), los **6 videos no controlados (*In-the-Wild*)** y latencia real en CPU.
- **Fase 4:** `tracker_extractor.py` (ByteTrack + ROI-Crop Pose) + `realtime_v2.py` validado en videos no controlados.
- **Fase 5:** `export_v2_onnx.py` (ONNX FP32 + INT8) y reporte final de velocidad, precisión y viabilidad móvil.
