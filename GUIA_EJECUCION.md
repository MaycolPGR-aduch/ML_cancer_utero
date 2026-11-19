# Guía de Ejecución — Prototipo IA-Tamizaje Cervical v0.1 (TRL 3)

Esta guía explica paso a paso cómo instalar dependencias, ejecutar la aplicación web, probar la inferencia por línea de comandos (CLI), consumir la API REST y reproducir desde cero el entrenamiento de ambos modelos.

---

## 1. Requisitos Previos

* **Sistema Operativo:** Windows 10/11 (también compatible con Linux/macOS ajustando el comando de activación).
* **Python:** Versión **3.10 o superior** (verificado en Python `3.10.0`).
* **RAM recomendada:** Mínimo 8 GB (16 GB recomendados si vas a re-entrenar la CNN).
* **Espacio en disco:**
  * Solo para ejecutar la aplicación web y pruebas de inferencia (usando los modelos ya incluidos en el repositorio): **~600 MB** (dependencias de Python + modelos).
  * Para re-entrenar la CNN desde cero: requiere además la carpeta del dataset SIPaKMeD (`archive_base1/`, ~9 GB).

---

## 2. Instalación de Dependencias

Desde la raíz del repositorio (`ML_cancer_utero/`):

### Opción A — Entorno virtual dedicado (recomendado)

```powershell
# Crear y activar entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Instalar PyTorch (version CPU, ligera ~125 MB)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Instalar el resto de dependencias del prototipo
pip install -r prototipo_trl3\requirements.txt
```

### Opción B — Instalación directa en tu Python global

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r prototipo_trl3\requirements.txt
```

---

## 3. Ejecutar la Aplicación Web (FastAPI + Frontend)

Los modelos entrenados (`models/cnn_sipakmed_trl3_v01.pt` y `models/tabular_trl3_v01.joblib`) ya están incluidos en el repositorio, por lo que puedes iniciar la aplicación web inmediatamente sin re-entrenar.

### Forma rápida (Windows)
Haz doble clic en el archivo:
```text
prototipo_trl3\start_web.bat
```

### Forma manual (PowerShell / Terminal)

```powershell
cd prototipo_trl3
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Cuando veas el mensaje `Uvicorn running on http://127.0.0.1:8000`, abre tu navegador en:

* **Interfaz Web principal:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Documentación interactiva Swagger (OpenAPI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Cómo usar las 3 pestañas de la interfaz web
1. **Citología Cervical (CNN + Grad-CAM):**
   * Arrastra y suelta cualquier imagen celular (`.bmp`, `.png`, `.jpg`), **o** haz clic en cualquiera de los **5 botones de muestras reales del conjunto Test** (una por cada clase SIPaKMeD).
   * Verás la clasificación, nivel de prioridad de tamizaje (`ALTO` / `MODERADO` / `BAJO`), probabilidad anormal agregada, comparativa lado a lado **Original `224×224` vs. Mapa de calor Grad-CAM**, barras de probabilidad de las 5 clases, latencia en milisegundos y salida JSON estructurada.
2. **Factores de Riesgo (Tabular):**
   * Usa los botones rápidos **Cargar Perfil Bajo**, **Cargar Perfil Alto** o **Vaciar (Probar Imputación)**, o llena los campos manualmente y pulsa **Calcular Estratificación de Riesgo**.
   * Verás la probabilidad estimada de biopsia positiva comparada visualmente frente al umbral de tamizaje (`0.44`), número de variables imputadas automáticamente y las variables más influyentes del modelo.
3. **Evidencia Experimental TRL 3:**
   * Consulta en vivo los KPIs del experimento, descarga `evidencia_trl3.json` y visualiza todas las gráficas generadas (matrices de confusión, curvas ROC/PR, historial de entrenamiento, anotaciones `.dat` escaladas y ejemplos Grad-CAM).

Para detener el servidor, presiona `Ctrl + C` en la terminal.

---

## 4. Ejecutar Inferencia por Línea de Comandos (CLI — Evidencia TRL 3)

El script `prototipo_trl3/infer.py` permite demostrar inferencia reproducible en consola y guardar el mapa Grad-CAM:

```powershell
cd prototipo_trl3

# 1) Solo imagen citologica (con guardado de mapa de calor Grad-CAM)
python infer.py --image data/sample_images/im_Dyskeratotic_011_01.bmp --save-gradcam outputs/demo_gradcam.png

# 2) Solo registro tabular (por indice del CSV)
python infer.py --tabular-index 0

# 3) Ejecucion multimodal simultanea (CNN + Tabular)
python infer.py --image data/sample_images/im_Koilocytotic_006_01.bmp --tabular-index 0
```

También puedes pasar un registro clínico personalizado en formato JSON:
```powershell
python infer.py --tabular-json mi_caso.json
```

---

## 5. Consumo Directo de la API REST (Ejemplos)

Con el servidor corriendo en `http://127.0.0.1:8000`:

```powershell
# Verificar estado de modelos
curl.exe http://127.0.0.1:8000/api/health

# Predecir una imagen local mediante multipart/form-data
curl.exe -X POST http://127.0.0.1:8000/api/predict/image -F "file=@prototipo_trl3/data/sample_images/im_Dyskeratotic_011_01.bmp"

# Predecir un perfil tabular via JSON
curl.exe -X POST http://127.0.0.1:8000/api/predict/tabular -H "Content-Type: application/json" -d "{\"values\": {\"Age\": 38, \"Smokes\": 1, \"STDs:HPV\": 1}}"

# Obtener evidencia experimental TRL 3 consolidada
curl.exe http://127.0.0.1:8000/api/evidence
```

---

## 6. Reproducir los Entrenamientos desde Cero

Todos los entrenamientos usan semilla fija `random_state=42` para reproducibilidad.

### A. Módulo Tabular (~5 segundos en CPU)
Usa directamente `prototipo_trl3/data/kag_risk_factors_cervical_cancer.csv`:

```powershell
cd prototipo_trl3
python train_eval_tabular.py
```

Genera:
* `models/tabular_trl3_v01.joblib`
* `outputs/evidencia_tabular_trl3.json`
* `outputs/tabular_confusion_matrix.png` y `outputs/tabular_roc_pr_curves.png`

### B. Módulo CNN — SIPaKMeD (~18 minutos en CPU Intel i7 / ~1-2 min en GPU)
Requiere tener descomprimido el dataset SIPaKMeD en la carpeta raíz `archive_base1/` con la estructura:
```text
archive_base1/
├── im_Dyskeratotic/im_Dyskeratotic/CROPPED/*.bmp
├── im_Koilocytotic/im_Koilocytotic/CROPPED/*.bmp
├── im_Metaplastic/im_Metaplastic/CROPPED/*.bmp
├── im_Parabasal/im_Parabasal/CROPPED/*.bmp
└── im_Superficial-Intermediate/im_Superficial-Intermediate/CROPPED/*.bmp
```

Ejecución:
```powershell
cd prototipo_trl3
python train_eval_cnn.py --epochs-head 2 --epochs-finetune 6 --batch-size 32
```

Genera:
* `models/cnn_sipakmed_trl3_v01.pt`
* `outputs/evidencia_cnn_trl3.json` y `outputs/evidencia_trl3.json` (consolidado)
* `outputs/cnn_confusion_matrix.png`, `outputs/cnn_roc_pr_curves.png`, `outputs/cnn_training_history.png`, `outputs/cnn_sample_annotations_224.png` y `outputs/cnn_sample_gradcam.png`

---

## 7. Solución de Problemas Comunes

| Problema | Causa / Solución |
|---|---|
| `No module named 'multipart'` al subir imagen | Instalar `pip install python-multipart`. |
| `El puerto 8000 ya está en uso` | Ejecuta con otro puerto: `python -m uvicorn app.main:app --port 8001` y abre `http://127.0.0.1:8001`. |
| `No se encontro la carpeta CROPPED` al entrenar CNN | Solo ocurre al ejecutar `train_eval_cnn.py` sin `archive_base1/` descomprimido. Para usar la app web o `infer.py` **no necesitas** `archive_base1/`. |
| Error de permisos al activar `.venv` en PowerShell | Ejecuta `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y vuelve a activar el entorno virtual. |
