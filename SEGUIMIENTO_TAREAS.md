# Seguimiento de Tareas — Proyecto IA-Tamizaje Cervical (TRL 3)

**Repositorio:** `https://github.com/MaycolPGR-aduch/ML_cancer_utero.git`  
**Versión actual del prototipo:** `v0.1-trl3` (`v0.1-trl3-cnn` + `v0.1-trl3-tabular`)  
**Última actualización:** 2026-11-19

---

## Estado General por Fase

| Fase | Descripción | Estado | Avance |
|---|---|---|---:|
| Fase 1 | Definición del alcance TRL 3, diagnóstico de entorno e inspección de datasets | Completada | 100% |
| Fase 2 | Módulo Tabular (Factores de Riesgo Cervical) + evidencia reproducible | Completada | 100% |
| Fase 3 | Módulo CNN (`MobileNetV2` sobre SIPaKMeD) + control anti-leakage + XAI | Completada | 100% |
| Fase 4 | Backend FastAPI, Interfaz Web multimodal y script CLI (`infer.py`) | Completada | 100% |
| Fase 5 | Documentación, control de versiones Git y preparación de repositorio | Completada | 100% |
| Fase 6 | Hoja de ruta futura (hacia TRL 4 / TRL 5) | Pendiente (Post-TRL 3) | 0% |

---

## Detalle de Tareas Completadas (TRL 3)

### Fase 1 — Análisis de Contexto y Datos
- [x] Revisión de criterios mínimos de madurez tecnológica en `Para evidenciar TRL 3.docx`.
- [x] Inventario de notebooks de referencia y extracción de `kag_risk_factors_cervical_cancer.csv` (858 filas × 36 columnas).
- [x] Verificación de estructura del dataset de imágenes `archive_base1` (SIPaKMeD `CROPPED`: 4,049 imágenes `.bmp` en 5 clases + 8,098 archivos de contorno `*_cyt.dat` y `*_nuc.dat`).
- [x] Diagnóstico de hardware y software local (Intel Core i7-1260P, 16 GB RAM, Python 3.10, PyTorch CPU).

### Fase 2 — Módulo Tabular (Factores de Riesgo)
- [x] Implementación de `prototipo_trl3/train_eval_tabular.py` con semilla fija (`random_state=42`).
- [x] Limpieza de valores `?`, exclusión de 8 variables post-prueba (anti-leakage) y 2 columnas con >50% de valores faltantes (quedando 26 variables de entrada).
- [x] Comparación de 3 modelos baseline (`LogisticRegression`, `RandomForest`, `ExtraTrees`) con `SimpleImputer(median)` + `StandardScaler` dentro de `Pipeline` y `StratifiedKFold(k=5)`.
- [x] Optimización del umbral de tamizaje (`0.44`), serialización de `models/tabular_trl3_v01.joblib` y benchmark de latencia (`35.34 ms`).
- [x] Generación de gráficas (`tabular_confusion_matrix.png`, `tabular_roc_pr_curves.png`) y `outputs/evidencia_tabular_trl3.json`.

### Fase 3 — Módulo CNN (Citología SIPaKMeD)
- [x] Implementación de `prototipo_trl3/train_eval_cnn.py` con partición agrupada por lámina (`StratifiedGroupKFold`, `k=7`) verificando **0 láminas compartidas** entre train/val/test.
- [x] Verificación y visualización de escalado proporcional de contornos `*_cyt.dat` y `*_nuc.dat` a `224×224` (`outputs/cnn_sample_annotations_224.png`).
- [x] Entrenamiento Transfer Learning en 2 etapas sobre `MobileNetV2` (2 épocas cabeza + 6 épocas fine-tuning con `CosineAnnealingLR` y *data augmentation* exclusivo en entrenamiento).
- [x] Evaluación rigurosa en Test (`n=614`): Accuracy `94.46%`, Macro-F1 `95.02%`, Macro ROC-AUC `0.9948`, Sensibilidad binaria Anormal `95.16%`, Especificidad `98.46%`, latencia CPU `13.23 ms`.
- [x] Implementación de explicabilidad visual Grad-CAM (`prototipo_trl3/gradcam.py`) y generación de `outputs/cnn_sample_gradcam.png`.
- [x] Consolidación de evidencia JSON unificada en `outputs/evidencia_trl3.json` y exportación de 5 muestras reales del conjunto Test en `data/sample_images/`.

### Fase 4 — Aplicación Web FastAPI, Filtro OOD e Inferencia CLI
- [x] Implementación de **Filtro de Validez de Dominio OOD** (`prototipo_trl3/train_eval_ood.py`, `models/ood_domain_gate_v01.joblib`) usando los 7 órganos NO cervicales de `archivebase2` (excluyendo `Cervical Cancer` de `archivebase2` para evitar data leakage) sobre embeddings compartidos de MobileNetV2: Recall 100%, Especificidad 100%, Leave-One-Organ-Out 96.4%–100%, sobrecosto `+0.178 ms`.
- [x] Implementación de script CLI verificable `prototipo_trl3/infer.py` (imagen + OOD y/o registro tabular).
- [x] Desarrollo de API REST en `prototipo_trl3/app/main.py` (`/api/health`, `/api/samples`, `/api/predict/image`, `/api/predict/sample/{filename}`, `/api/predict/ood-sample/{filename}`, `/api/tabular/schema`, `/api/predict/tabular`, `/api/evidence`).
- [x] Diseño e implementación de interfaz web (`index.html`, `styles.css`, `app.js`) con 3 pestañas, aviso clínico TRL 3, rechazo visual OOD con 3 muestras de prueba, visualización de Grad-CAM y salida JSON descargable.
- [x] Pruebas automatizadas end-to-end de todos los endpoints de la API (incluyendo rechazo OOD) y del script CLI.
- [x] Creación de lanzador rápido en Windows (`prototipo_trl3/start_web.bat`).

### Fase 5 — Repositorio, Informe Word y Documentación
- [x] Creación de `.gitignore` excluyendo datasets pesados (`archive_base1/`, `archivebase2/`, `*.zip`) y archivos temporales.
- [x] Creación de `README.md` raíz, `GUIA_EJECUCION.md`, `SEGUIMIENTO_TAREAS.md` e informe técnico en Word `Informe_Tecnico_TRL3_IA_Tamizaje_Cervical.docx` (con generador reproducible `generate_trl3_report.py`).
- [x] Inicialización de repositorio Git local y vinculación con `origin` (`https://github.com/MaycolPGR-aduch/ML_cancer_utero.git`).

---

## Próximos Pasos / Backlog (Post-TRL 3 → TRL 4 y TRL 5)

- [ ] **Evidencias documentales finales TRL 3:** Tomar capturas de pantalla de la interfaz web en ejecución (las 3 pestañas, incluyendo rechazo OOD) y adjuntarlas si se requiere complemento visual adicional.
- [x] **Detección fuera de dominio (OOD):** Filtro previo implementado y verificado (`v0.1-trl3-ood`, rechazo 100% en test retenido y 96.4%–100% en órganos nunca vistos).
- [ ] **Soporte de láminas completas (WSI / campo completo):** Integrar un detector/segmentador previo de células (usando las anotaciones `*_cyt.dat`/`*_nuc.dat` de las láminas completas de SIPaKMeD) para procesar campos microscópicos sin recorte manual.
- [ ] **Exportación y despliegue en contenedor:** Empaquetar la aplicación FastAPI en `Dockerfile` y exportar la CNN a `ONNX Runtime` para despliegue reproducible en entorno integrado (TRL 4).
- [ ] **Validación externa multicéntrica:** Evaluar generalización del modelo en un dataset independiente (ej. Herlev o cohorte hospitalaria local anonimizada con comité de ética) como paso hacia validación relevante (TRL 5).
