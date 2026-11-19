---
title: IA-Tamizaje Cervical TRL 3
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Prototipo TRL 3 CNN MobileNetV2 + Grad-CAM + Filtro OOD + Tabular
---

# IA-Tamizaje Cervical (ML_cancer_utero) — Prototipo TRL 3

Sistema experimental multimodal de apoyo temprano al tamizaje de **cáncer de cuello uterino (cervical)** basado en Aprendizaje Profundo (**CNN MobileNetV2 + Grad-CAM**) y Aprendizaje Automático Tabular (**RandomForest**), acompañado de una **aplicación web en FastAPI** y evidencia técnica reproducible para **TRL 3** (*Prueba de concepto analítica y experimental en entorno controlado*).

> **Aviso Clínico y Regulatorio:** Este proyecto es un prototipo de investigación en nivel **TRL 3**. Sus predicciones son exclusivamente de apoyo al tamizaje y exploración técnica; **no constituyen un diagnóstico médico ni sustituyen la valoración de profesionales en citopatología o ginecología**.

---

## Documentación Principal

* **[GUIA_EJECUCION.md](GUIA_EJECUCION.md):** Guía paso a paso para instalar dependencias, iniciar la app web, ejecutar inferencia por CLI, probar la API REST y reproducir los entrenamientos.
* **[SEGUIMIENTO_TAREAS.md](SEGUIMIENTO_TAREAS.md):** Registro detallado del avance del proyecto, hitos completados (TRL 3) y hoja de ruta hacia TRL 4–5.
* **[prototipo_trl3/README.md](prototipo_trl3/README.md):** Ficha Técnica de Evidencia Experimental TRL 3 y métricas por clase.
* **[prototipo_trl3/outputs/evidencia_trl3.json](prototipo_trl3/outputs/evidencia_trl3.json):** Archivo JSON consolidado y verificable con todas las métricas, entorno, particiones y tiempos de inferencia.

---

## Arquitectura del Sistema

```text
                       ┌──────────────────────────────────────┐
                       │      Interfaz Web (HTML/CSS/JS)      │
                       │  1) Citología CNN + Grad-CAM         │
                       │  2) Perfil Clínico Tabular           │
                       │  3) Panel Evidencia TRL 3            │
                       └──────────────────┬───────────────────┘
                                          │ HTTP / JSON
                       ┌──────────────────▼───────────────────┐
                       │         Backend API (FastAPI)        │
                       │       prototipo_trl3/app/main.py     │
                       └─────────┬──────────────────┬─────────┘
                                 │                  │
        ┌────────────────────────▼──────┐    ┌──────▼────────────────────────┐
        │   Módulo CNN (PyTorch CPU)    │    │  Módulo Tabular (scikit-learn)│
        │  MobileNetV2 (Transfer Learn.)│    │  Pipeline: Median Imputer +   │
        │  SIPaKMeD (5 clases, 224×224) │    │  StandardScaler + RF balanced │
        │  + Grad-CAM explicable (XAI)  │    │  (26 variables sin leakage)   │
        └───────────────────────────────┘    └───────────────────────────────┘
```

---

## Resumen de Resultados Experimentales (TRL 3)

Todos los experimentos fueron ejecutados de forma reproducible (`random_state=42`) con control estricto contra fuga de información (*data leakage*).

### 1. Módulo de Imágenes — CNN (`MobileNetV2` sobre SIPaKMeD)
* **Dataset:** `4,049` células recortadas (`CROPPED`) en 5 clases (`im_Dyskeratotic`, `im_Koilocytotic`, `im_Metaplastic`, `im_Parabasal`, `im_Superficial-Intermediate`) + escalado proporcional de contornos `*_cyt.dat` y `*_nuc.dat` a `224×224`.
* **Split sin fuga por lámina (`StratifiedGroupKFold`, `k=7`):** Train `2,819` (691 láminas) · Val `616` (137 láminas) · Test `614` (138 láminas) — **0 láminas compartidas entre particiones**.
* **Desempeño en Test (`n=614`):**

| Métrica (Test 5 Clases) | Valor |
|---|---:|
| Accuracy | **94.46%** |
| Macro Precision | **94.82%** |
| Macro Recall | **95.27%** |
| Macro F1-Score | **95.02%** |
| Macro ROC-AUC (OvR) | **0.9948** |
| Macro PR-AUC (OvR) | **0.9838** |
| **Tamizaje Binario (Anormal vs Normal/Benigna)** | **Sensibilidad 95.16% · Especificidad 98.46% · ROC-AUC 0.9964** |
| Latencia promedio inferencia (CPU) | **13.23 ms / imagen** |

### 2. Módulo Tabular — Factores de Riesgo (`kag_risk_factors_cervical_cancer.csv`)
* **Dataset:** `858` registros × `36` columnas → `26` variables clínicas de entrada tras excluir 8 variables post-prueba (evitando *data leakage*) y 2 columnas con >50% valores faltantes (`?`). Objetivo principal: `Biopsy` (`55/858`, `6.41%`).
* **Validación:** Cruzada estratificada (`StratifiedKFold`, `k=5`) comparando `LogisticRegression`, `RandomForest` y `ExtraTrees` dentro de `Pipeline`.
* **Modelo seleccionado:** `RandomForest (balanced)` → **ROC-AUC CV:** `0.6437 ± 0.1025`, **PR-AUC CV:** `0.1695` (2.6× sobre prevalencia base `0.0641`), umbral de tamizaje `0.44` (`Recall OOF=30.91%`, `Especificidad=85.93%`), latencia promedio **`35.34 ms`**.

---

## Estructura del Repositorio

```text
ML_cancer_utero/
├── .gitignore
├── README.md                              # Resumen general del repositorio
├── GUIA_EJECUCION.md                      # Manual completo de instalacion y ejecucion
├── SEGUIMIENTO_TAREAS.md                  # Control de tareas TRL 3 y roadmap
├── Para evidenciar TRL 3.docx             # Lineamientos base de evidencia TRL 3
└── prototipo_trl3/
    ├── README.md                          # Ficha tecnica TRL 3 detallada
    ├── requirements.txt
    ├── start_web.bat                      # Lanzador rapido del servidor FastAPI en Windows
    ├── train_eval_cnn.py                  # Entrenamiento y evaluacion CNN (SIPaKMeD)
    ├── train_eval_tabular.py              # Entrenamiento y evaluacion tabular (CV 5-fold)
    ├── gradcam.py                         # Explicabilidad Grad-CAM en PyTorch
    ├── infer.py                           # Inferencia verificable por CLI
    ├── app/                               # Servidor FastAPI + interfaz web estatica
    ├── data/                              # CSV tabular + 5 imagenes de muestra del conjunto Test
    ├── models/                            # Modelos entrenados listos para inferencia (~9.5 MB)
    └── outputs/                           # evidencia_trl3.json y graficas PNG verificables
```

*(Nota: El dataset completo SIPaKMeD `archive_base1/` de ~9 GB se excluye en `.gitignore` por tamaño; el repositorio incluye los modelos ya entrenados y 5 imágenes de prueba reales para ejecutar y demostrar la aplicación de inmediato).*

---

## Inicio Rápido

```powershell
# 1. Instalar dependencias
pip install -r prototipo_trl3/requirements.txt

# 2. Iniciar la aplicacion web
cd prototipo_trl3
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Abre **http://127.0.0.1:8000** en tu navegador. Para detalles completos, consulta **[GUIA_EJECUCION.md](GUIA_EJECUCION.md)**.
