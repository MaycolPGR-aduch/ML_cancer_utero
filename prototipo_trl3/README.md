# Prototipo IA-Tamizaje Cervical v0.1 (TRL 3)

Prototipo multimodal reproducible en entorno controlado (**TRL 3 — Prueba de concepto analítica y experimental**) para apoyo temprano al tamizaje de cáncer cervical, compuesto por:

1. **Módulo de Imágenes (CNN + Grad-CAM):** Clasificación citológica en 5 clases sobre **SIPaKMeD** (`4,049` imágenes celulares recortadas) con Transfer Learning en `MobileNetV2`, partición agrupada por lámina sin fuga de datos (*no leakage*), escalado de coordenadas `*_cyt.dat`/`*_nuc.dat` y mapas de calor explicables **Grad-CAM**.
2. **Módulo Tabular (Factores de Riesgo):** Estratificación de riesgo de biopsia positiva sobre `kag_risk_factors_cervical_cancer.csv` (`858` registros × `36` columnas → `26` variables clínicas sin fuga), con imputación por mediana dentro de pipeline y validación cruzada estratificada ($k=5$).
3. **Aplicación Web FastAPI + Interfaz Web:** Subida de imágenes citológicas, evaluación de perfil clínico tabular y panel de evidencia experimental TRL 3 con gráficas verificables y salida JSON estructurada.

> **Aviso Clínico y Regulatorio:** Software experimental en nivel TRL 3 para investigación y apoyo al tamizaje. **No constituye un diagnóstico médico ni reemplaza el criterio citopatológico o clínico.**

---

## Estructura del Repositorio

```text
prototipo_trl3/
├── README.md
├── requirements.txt
├── start_web.bat                          # Lanzador rapido en Windows (FastAPI en puerto 8000)
├── train_eval_tabular.py                  # Entrenamiento CV 5-fold + baseline + modelo + evidencia tabular
├── train_eval_cnn.py                      # Entrenamiento CNN 2 etapas + evaluacion Test + Grad-CAM + evidencia
├── gradcam.py                             # Explicabilidad Grad-CAM nativa en PyTorch (MobileNetV2)
├── infer.py                               # Inferencia CLI reproducible (imagen y/o tabular -> JSON)
├── data/
│   ├── kag_risk_factors_cervical_cancer.csv
│   └── sample_images/                     # 5 celulas reales del conjunto Test (1 por clase) + archivos .dat
├── models/
│   ├── cnn_sipakmed_trl3_v01.pt           # Modelo MobileNetV2 entrenado + metadatos
│   └── tabular_trl3_v01.joblib            # Pipeline RandomForest + metadatos + umbral
├── outputs/
│   ├── evidencia_trl3.json                # Evidencia experimental TRL 3 consolidada (CNN + Tabular)
│   ├── evidencia_cnn_trl3.json
│   ├── evidencia_tabular_trl3.json
│   ├── cnn_confusion_matrix.png
│   ├── cnn_roc_pr_curves.png
│   ├── cnn_training_history.png
│   ├── cnn_sample_annotations_224.png     # Verificacion de escalado *_cyt.dat / *_nuc.dat a 224x224
│   ├── cnn_sample_gradcam.png             # Mapas de calor Grad-CAM sobre muestras de Test
│   ├── tabular_confusion_matrix.png
│   └── tabular_roc_pr_curves.png
└── app/
    ├── main.py                            # API REST FastAPI + servidor de interfaz web y evidencias
    └── static/
        ├── index.html
        ├── styles.css
        └── app.js
```

---

## Cómo Ejecutar el Prototipo

### 1. Iniciar la Aplicación Web (FastAPI)
Haz doble clic en `start_web.bat` o ejecuta en PowerShell/CMD:

```powershell
cd C:\ML_img_can\prototipo_trl3
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Luego abre en tu navegador: **http://127.0.0.1:8000**

### 2. Inferencia Reproducible por Línea de Comandos (`infer.py`)

```powershell
cd C:\ML_img_can\prototipo_trl3

# Inferencia combinada (Imagen citologica + Registro tabular #0)
python infer.py --image data/sample_images/im_Dyskeratotic_011_01.bmp --tabular-index 0 --save-gradcam outputs/prueba_gradcam.png
```

### 3. Reproducir Entrenamientos (`random_state=42`)

```powershell
python train_eval_tabular.py
python train_eval_cnn.py --epochs-head 2 --epochs-finetune 6 --batch-size 32
```

---

## Ficha Técnica de Evidencia Experimental (TRL 3)

| Sección | Evidencia Registrada |
|---|---|
| **Identificación** | Prototipo IA-Tamizaje Cervical (`v0.1-trl3-cnn` / `v0.1-trl3-tabular`), fecha 2026-11-19 (UTC), semilla fija `random_state=42`. |
| **Entorno de Ejecución** | Windows 10/11 (AMD64), Intel Core i7-1260P (CPU, 8 hilos), Python 3.10.0, PyTorch `2.14.1+cpu`, scikit-learn `1.7.2`, FastAPI `0.141.1`. |
| **Dataset Imágenes (CNN)** | **SIPaKMeD (`CROPPED`):** 4,049 imágenes en 5 clases (`im_Dyskeratotic`: 813; `im_Koilocytotic`: 825; `im_Metaplastic`: 793; `im_Parabasal`: 787; `im_Superficial-Intermediate`: 831) + anotaciones `*_cyt.dat`/`*_nuc.dat` escaladas a `224×224`. |
| **Control Anti-Leakage (CNN)** | Partición `StratifiedGroupKFold (k=7)` agrupada por `(clase, id_lámina)`: Train 2,819 (691 láminas), Val 616 (137 láminas), Test 614 (138 láminas). **0 láminas compartidas entre splits.** |
| **Arquitectura y Entrenamiento (CNN)** | `MobileNetV2` (pesos ImageNet), 2 etapas: 2 épocas cabeza (`lr=1.5e-3`) + 6 épocas fine-tuning bloques superiores (`features[10:]`, `lr=4e-4` con `CosineAnnealingLR`), `AdamW`, `CrossEntropyLoss` ponderada, `batch_size=32`, *data augmentation* exclusivo en entrenamiento. |
| **Resultados CNN (Test, n=614)** | **Accuracy:** `94.46%` · **Macro Precision:** `94.82%` · **Macro Recall:** `95.27%` · **Macro F1:** `95.02%` · **Macro ROC-AUC (OvR):** `0.9948` · **Macro PR-AUC (OvR):** `0.9838` · **Latencia CPU:** `13.23 ms/imagen`. |
| **Tamizaje Binario CNN (Anormal vs Normal/Benigna)** | Agrupando `im_Dyskeratotic + im_Koilocytotic` como Anormal: **Sensibilidad (Recall):** `95.16%`, **Especificidad:** `98.46%`, **Precision:** `98.21%`, **F1:** `96.66%`, **ROC-AUC:** `0.9964` (`TP=275, TN=320, FP=5, FN=14`). |
| **Dataset Tabular** | `kag_risk_factors_cervical_cancer.csv`: 858 filas × 36 columnas → **26 variables de entrada** tras excluir 8 desenlaces post-prueba (sin fuga) y 2 columnas con >50% faltantes (`787/858` valores `?`). Objetivo principal: `Biopsy` (`55/858`, `6.41%`). |
| **Resultados Tabular (CV 5-Fold)** | Comparación de 3 baselines (`LogisticRegression`, `RandomForest`, `ExtraTrees`) con `SimpleImputer(median)` + `StandardScaler` dentro de `Pipeline` y `class_weight='balanced'`. Mejor modelo: **`RandomForest (balanced)`** con **ROC-AUC CV:** `0.6437 ± 0.1025`, **PR-AUC CV:** `0.1695` (2.6× sobre prevalencia base `0.0641`), umbral de tamizaje `0.44` (`Recall OOF=30.91%`, `Especificidad=85.93%`), **Latencia CPU:** `35.34 ms/caso`. |
| **Evidencias Verificables** | `outputs/evidencia_trl3.json`, gráficas PNG en `outputs/`, pruebas automatizadas de API FastAPI y ejecución CLI en `infer.py`. |

### Detalle por Clase — Módulo CNN en Conjunto de Prueba (`n=614`, sin fuga por lámina)

| Clase SIPaKMeD | Categoría Tamizaje | Precision | Recall | F1-Score | ROC-AUC (OvR) | PR-AUC (OvR) | Soporte Test |
|---|---|---:|---:|---:|---:|---:|---:|
| `im_Dyskeratotic` | Anormal | 0.9403 | 0.9474 | 0.9438 | 0.9945 | 0.9826 | 133 |
| `im_Koilocytotic` | Anormal (VPH) | 0.9122 | 0.8654 | 0.8882 | 0.9835 | 0.9520 | 156 |
| `im_Metaplastic` | Benigna | 0.9134 | 0.9508 | 0.9317 | 0.9961 | 0.9861 | 122 |
| `im_Parabasal` | Normal | 0.9750 | 1.0000 | 0.9873 | 0.9998 | 0.9985 | 78 |
| `im_Superficial-Intermediate` | Normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 125 |
