"""
Prototipo IA-Tamizaje Cervical v0.1 (TRL 3) - Script de Inferencia CLI
======================================================================
Permite ejecutar inferencia verificable y reproducible desde linea de comandos:
  1) Sobre una imagen citologica (.bmp, .png, .jpg):
     python infer.py --image data/sample_images/im_Dyskeratotic_001_01.bmp
  2) Sobre un caso tabular (indice del CSV o archivo JSON de variables):
     python infer.py --tabular-index 0
     python infer.py --tabular-json caso.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from gradcam import MobileNetV2GradCAM

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

CNN_MODEL_PATH = MODELS_DIR / "cnn_sipakmed_trl3_v01.pt"
TABULAR_MODEL_PATH = MODELS_DIR / "tabular_trl3_v01.joblib"


def load_cnn_bundle(model_path: Path = CNN_MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontro el modelo CNN en {model_path}. Ejecute train_eval_cnn.py primero.")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features, len(classes)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    eval_tf = transforms.Compose([
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=ckpt["mean"], std=ckpt["std"]),
    ])
    return model, ckpt, eval_tf


def predict_image(
    image_path: Path,
    model: nn.Module,
    ckpt: dict,
    eval_tf,
    save_gradcam_path: Optional[Path] = None,
) -> dict:
    pil_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_img.size

    t0 = time.perf_counter()
    inp = eval_tf(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    classes = ckpt["classes"]
    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    abnormal_idx = ckpt["abnormal_class_indices"]
    prob_abnormal = float(probs[abnormal_idx].sum())

    if prob_abnormal >= 0.50:
        nivel_riesgo = "ALTO (Sospecha de celula anormal - requiere revision citopatologica)"
    elif prob_abnormal >= 0.25:
        nivel_riesgo = "MODERADO (Hallazgo indeterminado - seguimiento recomendado)"
    else:
        nivel_riesgo = "BAJO (Patron compatible con celula normal o benigna)"

    gradcam_saved = None
    if save_gradcam_path is not None:
        gcam = MobileNetV2GradCAM(model)
        overlay, _ = gcam.generate(inp, pil_img, target_class=pred_idx, alpha=0.45)
        Image.fromarray(overlay).save(save_gradcam_path)
        gradcam_saved = str(save_gradcam_path)

    return {
        "modulo": "cnn_citologia_sipakmed",
        "version": ckpt["version"],
        "entrada": {
            "archivo": str(image_path.name),
            "tamano_original": [orig_w, orig_h],
            "tamano_modelo": [ckpt["img_size"], ckpt["img_size"]],
        },
        "prediccion": {
            "clase_predicha": pred_class,
            "nombre_clinico": ckpt["clinical_info"][pred_class]["nombre_es"],
            "categoria_tamizaje": ckpt["clinical_info"][pred_class]["categoria_tamizaje"],
            "confianza_clase_predicha": round(float(probs[pred_idx]), 4),
            "probabilidad_anormal_agregada": round(prob_abnormal, 4),
            "nivel_prioridad_tamizaje": nivel_riesgo,
            "probabilidades_por_clase": {
                cls: round(float(probs[i]), 4) for i, cls in enumerate(classes)
            },
        },
        "latencia_inferencia_ms": round(latency_ms, 2),
        "gradcam_guardado": gradcam_saved,
        "aviso": "Herramienta experimental TRL 3 de apoyo al tamizaje. No constituye diagnostico medico definitivo.",
    }


def predict_tabular(record: Dict[str, Optional[float]], bundle: dict) -> dict:
    features = bundle["features"]
    pipeline = bundle["pipeline"]
    threshold = float(bundle["threshold"])

    row_dict = {}
    missing_imputed = []
    for f in features:
        val = record.get(f, None)
        if val is None or val == "" or val == "?":
            row_dict[f] = np.nan
            missing_imputed.append(f)
        else:
            row_dict[f] = float(val)

    df_in = pd.DataFrame([row_dict], columns=features)

    t0 = time.perf_counter()
    prob = float(pipeline.predict_proba(df_in)[0, 1])
    latency_ms = (time.perf_counter() - t0) * 1000.0

    is_high = prob >= threshold
    if prob >= max(threshold, 0.55):
        riesgo = "ALTO (Priorizar evaluacion clinica / colposcopia)"
    elif is_high:
        riesgo = "MODERADO-ALTO (Supera umbral de tamizaje - seguimiento recomendado)"
    else:
        riesgo = "BAJO (Riesgo estimado por debajo del umbral de tamizaje)"

    return {
        "modulo": "tabular_factores_riesgo",
        "version": bundle["version"],
        "modelo": bundle["model_name"],
        "prediccion": {
            "probabilidad_riesgo_biopsia": round(prob, 4),
            "umbral_tamizaje": threshold,
            "alerta_tamizaje": bool(is_high),
            "estrato_riesgo": riesgo,
        },
        "variables_imputadas_automaticamente": len(missing_imputed),
        "latencia_inferencia_ms": round(latency_ms, 2),
        "aviso": "Herramienta experimental TRL 3 de estratificacion de riesgo. No constituye diagnostico medico definitivo.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia reproducible TRL 3 (CNN y/o Tabular)")
    parser.add_argument("--image", type=str, help="Ruta a una imagen citologica (.bmp, .png, .jpg)")
    parser.add_argument("--save-gradcam", type=str, default=None, help="Ruta opcional para guardar overlay Grad-CAM")
    parser.add_argument("--tabular-index", type=int, default=None, help="Indice de fila del CSV tabular para inferencia")
    parser.add_argument("--tabular-json", type=str, default=None, help="Archivo JSON con variables clinicas")
    args = parser.parse_args()

    if not args.image and args.tabular_index is None and not args.tabular_json:
        parser.print_help()
        return

    results = {}

    if args.image:
        model, ckpt, eval_tf = load_cnn_bundle()
        gcam_out = Path(args.save_gradcam) if args.save_gradcam else None
        results["cnn"] = predict_image(Path(args.image), model, ckpt, eval_tf, gcam_out)

    if args.tabular_index is not None or args.tabular_json:
        if not TABULAR_MODEL_PATH.exists():
            raise FileNotFoundError(f"No se encontro {TABULAR_MODEL_PATH}. Ejecute train_eval_tabular.py primero.")
        bundle = joblib.load(TABULAR_MODEL_PATH)
        if args.tabular_json:
            rec = json.loads(Path(args.tabular_json).read_text(encoding="utf-8"))
        else:
            csv_path = DATA_DIR / "kag_risk_factors_cervical_cancer.csv"
            df = pd.read_csv(csv_path).replace("?", np.nan)
            rec = df.iloc[args.tabular_index].to_dict()
        results["tabular"] = predict_tabular(rec, bundle)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
