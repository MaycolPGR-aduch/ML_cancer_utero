"""
Prototipo IA-Tamizaje Cervical v0.1 (TRL 3) - Backend FastAPI + Interfaz Web
============================================================================
Ejecucion local:
    cd C:\ML_img_can\prototipo_trl3
    uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
"""

import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms

# Asegurar acceso a modulos de prototipo_trl3
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from gradcam import MobileNetV2GradCAM  # noqa: E402

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
SAMPLE_IMAGES_DIR = BASE_DIR / "data" / "sample_images"
STATIC_DIR = Path(__file__).resolve().parent / "static"

CNN_MODEL_PATH = MODELS_DIR / "cnn_sipakmed_trl3_v01.pt"
TABULAR_MODEL_PATH = MODELS_DIR / "tabular_trl3_v01.joblib"

app = FastAPI(
    title="Prototipo IA-Tamizaje Cervical v0.1 (TRL 3)",
    description="API de inferencia para clasificacion citologica (CNN MobileNetV2) y estratificacion de riesgo tabular.",
    version="0.1.0-trl3",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AppState:
    cnn_model: Optional[nn.Module] = None
    cnn_ckpt: Optional[dict] = None
    cnn_tf = None
    tabular_bundle: Optional[dict] = None


state = AppState()


def load_models() -> None:
    if CNN_MODEL_PATH.exists():
        ckpt = torch.load(CNN_MODEL_PATH, map_location="cpu", weights_only=False)
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_features, len(ckpt["classes"])),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        state.cnn_model = model
        state.cnn_ckpt = ckpt
        state.cnn_tf = transforms.Compose([
            transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=ckpt["mean"], std=ckpt["std"]),
        ])
    if TABULAR_MODEL_PATH.exists():
        state.tabular_bundle = joblib.load(TABULAR_MODEL_PATH)


@app.on_event("startup")
def on_startup() -> None:
    load_models()


def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def run_cnn_inference(pil_img: Image.Image, filename: str) -> dict:
    if state.cnn_model is None or state.cnn_ckpt is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo CNN aun no esta disponible. Ejecute train_eval_cnn.py primero.",
        )

    orig_w, orig_h = pil_img.size
    rgb_img = pil_img.convert("RGB")
    inp = state.cnn_tf(rgb_img).unsqueeze(0)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = state.cnn_model(inp)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    inference_ms = (time.perf_counter() - t0) * 1000.0

    ckpt = state.cnn_ckpt
    classes = ckpt["classes"]
    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    abnormal_indices = ckpt["abnormal_class_indices"]
    prob_abnormal = float(probs[abnormal_indices].sum())

    # Grad-CAM explicable
    t_cam0 = time.perf_counter()
    gcam = MobileNetV2GradCAM(state.cnn_model)
    overlay_np, _ = gcam.generate(inp, rgb_img, target_class=pred_idx, alpha=0.45)
    gradcam_ms = (time.perf_counter() - t_cam0) * 1000.0

    preview_224 = rgb_img.resize((ckpt["img_size"], ckpt["img_size"]), Image.BILINEAR)

    if prob_abnormal >= 0.50:
        triage_level = "ALTO"
        triage_text = "Sospecha de anomalia citologica (Disqueratosica / Coilocitica). Priorizar revision por especialista."
    elif prob_abnormal >= 0.25:
        triage_level = "MODERADO"
        triage_text = "Patron intermedio o probabilidad anormal moderada. Se recomienda correlacion clinica."
    else:
        triage_level = "BAJO"
        triage_text = "Patron compatible con morfologia celular normal o metaplasica benigna."

    ordered_probs = sorted(
        [
            {
                "clase": cls,
                "nombre_es": ckpt["clinical_info"][cls]["nombre_es"],
                "categoria": ckpt["clinical_info"][cls]["categoria_tamizaje"],
                "probabilidad": round(float(probs[i]), 4),
            }
            for i, cls in enumerate(classes)
        ],
        key=lambda x: x["probabilidad"],
        reverse=True,
    )

    return {
        "modulo": "cnn_citologia_sipakmed",
        "version": ckpt["version"],
        "entrada": {
            "archivo": filename,
            "tamano_original": [orig_w, orig_h],
            "tamano_modelo": [ckpt["img_size"], ckpt["img_size"]],
        },
        "prediccion": {
            "clase_predicha": pred_class,
            "nombre_clinico": ckpt["clinical_info"][pred_class]["nombre_es"],
            "categoria_tamizaje": ckpt["clinical_info"][pred_class]["categoria_tamizaje"],
            "descripcion": ckpt["clinical_info"][pred_class]["descripcion"],
            "confianza": round(float(probs[pred_idx]), 4),
            "probabilidad_anormal_agregada": round(prob_abnormal, 4),
            "nivel_tamizaje": triage_level,
            "recomendacion_apoyo": triage_text,
            "probabilidades": ordered_probs,
        },
        "tiempos_ms": {
            "inferencia_cnn": round(inference_ms, 2),
            "gradcam_xai": round(gradcam_ms, 2),
        },
        "imagenes": {
            "entrada_224": pil_to_base64_png(preview_224),
            "gradcam_overlay": pil_to_base64_png(Image.fromarray(overlay_np)),
        },
        "aviso_legal": "Prototipo experimental TRL 3 de apoyo al tamizaje. No reemplaza la evaluacion citopatologica ni constituye diagnostico medico.",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": "v0.1-trl3",
        "cnn_ready": state.cnn_model is not None,
        "tabular_ready": state.tabular_bundle is not None,
    }


@app.get("/api/samples")
def list_samples():
    if not SAMPLE_IMAGES_DIR.exists():
        return {"samples": []}
    samples = []
    for p in sorted(SAMPLE_IMAGES_DIR.glob("*.bmp")):
        cls = p.name.split("_0")[0]
        samples.append({"filename": p.name, "expected_class": cls})
    return {"samples": samples}


@app.post("/api/predict/image")
async def predict_image_upload(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Archivo de imagen vacio.")
    try:
        pil_img = Image.open(io.BytesIO(contents))
        pil_img.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo leer la imagen: {exc}")
    return JSONResponse(run_cnn_inference(pil_img, file.filename or "imagen_subida"))


@app.post("/api/predict/sample/{filename}")
def predict_sample_image(filename: str):
    safe_name = Path(filename).name
    target = SAMPLE_IMAGES_DIR / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail="Imagen de muestra no encontrada.")
    pil_img = Image.open(target)
    return JSONResponse(run_cnn_inference(pil_img, safe_name))


class TabularRequest(BaseModel):
    values: Dict[str, Optional[float]]


@app.get("/api/tabular/schema")
def tabular_schema():
    if state.tabular_bundle is None:
        raise HTTPException(status_code=503, detail="Modelo tabular no cargado.")
    b = state.tabular_bundle
    return {
        "version": b["version"],
        "model_name": b["model_name"],
        "threshold": b["threshold"],
        "features": [
            {
                "name": f,
                **b["feature_metadata"][f],
                "median": b["feature_medians"][f],
            }
            for f in b["features"]
        ],
        "top_features": b["top_features"],
    }


@app.post("/api/predict/tabular")
def predict_tabular_api(payload: TabularRequest):
    if state.tabular_bundle is None:
        raise HTTPException(status_code=503, detail="Modelo tabular no cargado.")
    b = state.tabular_bundle
    features = b["features"]
    pipeline = b["pipeline"]
    threshold = float(b["threshold"])

    row = {}
    imputed_count = 0
    for f in features:
        v = payload.values.get(f, None)
        if v is None:
            row[f] = np.nan
            imputed_count += 1
        else:
            row[f] = float(v)

    df_in = pd.DataFrame([row], columns=features)
    t0 = time.perf_counter()
    prob = float(pipeline.predict_proba(df_in)[0, 1])
    latency_ms = (time.perf_counter() - t0) * 1000.0

    if prob >= max(threshold, 0.55):
        level = "ALTO"
        text = "Riesgo estimado elevado respecto a la cohorte de referencia. Priorizar evaluacion clinica."
    elif prob >= threshold:
        level = "MODERADO"
        text = "Supera el umbral de tamizaje del prototipo. Se sugiere seguimiento preventivo."
    else:
        level = "BAJO"
        text = "Riesgo estimado por debajo del umbral de tamizaje del prototipo."

    return {
        "modulo": "tabular_factores_riesgo",
        "version": b["version"],
        "modelo": b["model_name"],
        "prediccion": {
            "probabilidad_riesgo_biopsia": round(prob, 4),
            "umbral_tamizaje": threshold,
            "alerta_tamizaje": bool(prob >= threshold),
            "nivel_tamizaje": level,
            "recomendacion_apoyo": text,
        },
        "variables_imputadas": imputed_count,
        "latencia_inferencia_ms": round(latency_ms, 2),
        "top_features": b["top_features"][:6],
        "aviso_legal": "Prototipo experimental TRL 3 de estratificacion de riesgo. No constituye diagnostico medico.",
    }


@app.get("/api/evidence")
def get_evidence():
    unified_path = OUTPUTS_DIR / "evidencia_trl3.json"
    if unified_path.exists():
        return json.loads(unified_path.read_text(encoding="utf-8"))
    result = {}
    tab_path = OUTPUTS_DIR / "evidencia_tabular_trl3.json"
    cnn_path = OUTPUTS_DIR / "evidencia_cnn_trl3.json"
    if tab_path.exists():
        result["modulo_tabular"] = json.loads(tab_path.read_text(encoding="utf-8"))
    if cnn_path.exists():
        result["modulo_cnn"] = json.loads(cnn_path.read_text(encoding="utf-8"))
    return result


# Servir archivos de evidencia (graficas PNG y JSON) y frontend estatico
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")
