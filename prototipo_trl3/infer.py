"""
Prototipo IA-Tamizaje Cervical v0.1 (TRL 3) - Script de Inferencia CLI
======================================================================
Permite ejecutar inferencia verificable y reproducible desde linea de comandos:
  1) Sobre una imagen citologica (.bmp, .png, .jpg) con Filtro de Validez OOD:
     python infer.py --image data/sample_images/im_Dyskeratotic_011_01.bmp
  2) Sobre una imagen fuera de dominio (para verificar rechazo OOD):
     python infer.py --image data/sample_ood_images/ood_brain_glioma_brain_glioma_0001.jpg
  3) Sobre un caso tabular (indice del CSV o archivo JSON de variables):
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
OOD_MODEL_PATH = MODELS_DIR / "ood_domain_gate_v01.joblib"
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
    ood_bundle = joblib.load(OOD_MODEL_PATH) if OOD_MODEL_PATH.exists() else None
    return model, ckpt, eval_tf, ood_bundle


def evaluate_ood_embedding(emb: np.ndarray, ood_bundle: Optional[dict]) -> dict:
    if ood_bundle is None:
        return {"habilitado": False, "es_citologia_cervical": True}

    t0 = time.perf_counter()
    prob_cerv = float(ood_bundle["pipeline"].predict_proba(emb)[0, 1])
    norm_emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    max_cos = float((norm_emb @ ood_bundle["class_centroids"].T).max())
    ood_ms = (time.perf_counter() - t0) * 1000.0

    prob_ok = prob_cerv >= float(ood_bundle["prob_threshold"])
    cos_ok = max_cos >= float(ood_bundle["cosine_threshold"])
    is_valid = bool(prob_ok and cos_ok)

    return {
        "habilitado": True,
        "version_ood": ood_bundle["version"],
        "es_citologia_cervical": is_valid,
        "probabilidad_dominio_cervical": round(prob_cerv, 4),
        "umbral_probabilidad": float(ood_bundle["prob_threshold"]),
        "similitud_coseno_centroides": round(max_cos, 4),
        "umbral_coseno": float(ood_bundle["cosine_threshold"]),
        "latencia_ood_ms": round(ood_ms, 3),
        "motivo": (
            "Imagen validada dentro del dominio de citologia cervical."
            if is_valid
            else "Imagen rechazada por el filtro OOD: no corresponde al dominio de citologia cervical (SIPaKMeD)."
        ),
    }


def predict_image(
    image_path: Path,
    model: nn.Module,
    ckpt: dict,
    eval_tf,
    ood_bundle: Optional[dict] = None,
    save_gradcam_path: Optional[Path] = None,
) -> dict:
    pil_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_img.size

    t0 = time.perf_counter()
    inp = eval_tf(pil_img).unsqueeze(0)
    with torch.no_grad():
        feats = model.features(inp)
        pooled = feats.mean(dim=[2, 3])
        logits = model.classifier(pooled)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        emb_np = pooled.cpu().numpy()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    ood_info = evaluate_ood_embedding(emb_np, ood_bundle)

    if not ood_info["es_citologia_cervical"]:
        return {
            "modulo": "cnn_citologia_sipakmed",
            "version": ckpt["version"],
            "entrada": {
                "archivo": str(image_path.name),
                "tamano_original": [orig_w, orig_h],
                "tamano_modelo": [ckpt["img_size"], ckpt["img_size"]],
            },
            "validacion_dominio_ood": ood_info,
            "prediccion": {
                "estado": "RECHAZADA_FUERA_DE_DOMINIO",
                "mensaje": "No se emite clasificacion citologica porque la imagen fue identificada como fuera de dominio.",
            },
            "latencia_inferencia_ms": round(latency_ms, 2),
            "aviso": "Herramienta experimental TRL 3 con filtro de validez de dominio OOD.",
        }

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
        "validacion_dominio_ood": ood_info,
        "prediccion": {
            "estado": "CLASIFICADA",
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
    parser = argparse.ArgumentParser(description="Inferencia reproducible TRL 3 (CNN + OOD y/o Tabular)")
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
        model, ckpt, eval_tf, ood_bundle = load_cnn_bundle()
        gcam_out = Path(args.save_gradcam) if args.save_gradcam else None
        results["cnn"] = predict_image(Path(args.image), model, ckpt, eval_tf, ood_bundle, gcam_out)

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
