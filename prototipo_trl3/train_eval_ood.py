"""
Prototipo IA-Tamizaje Cervical v0.1 (TRL 3/4) - Filtro de Validez de Dominio (OOD Gate)
=======================================================================================
Aprovecha las 7 categorias medicas NO cervicales de archivebase2 (Multi Cancer Dataset:
ALL, Brain, Breast, Kidney, Lung & Colon, Lymphoma, Oral) junto con los embeddings
profundos (1,280 dim) de MobileNetV2 para rechazar automaticamente imagenes que NO
corresponden a citologia cervical antes de emitir una clasificacion.

Rigor metodologico (cero fuga):
- Positivos (Citologia Cervical): usa exactamente la misma particion agrupada por lamina
  (StratifiedGroupKFold k=7) que train_eval_cnn.py -> Train (2,819), Val (616), Test (614).
- Negativos (No Cervical / OOD): muestra estratificada de las 21 subclases de los 7 organos
  no cervicales de archivebase2 (excluyendo estrictamente Cervical Cancer de archivebase2).
- Doble evaluacion verificable:
  1) Test retenido (SIPaKMeD Test sin fuga vs. 7 organos no cervicales + controles sinteticos).
  2) Validacion Leave-One-Organ-Out (LOOO): demuestra rechazo de organos COMPLETAMENTE
     no vistos durante el entrenamiento del filtro OOD.
- Doble compuerta en inferencia: clasificador lineal calibrado sobre embeddings + similitud
  coseno minima contra los 5 centroides citologicos de entrenamiento.
"""

import json
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from train_eval_cnn import (
    CLASSES,
    IMG_SIZE,
    SIPAKMED_ROOT,
    build_unified_trl3_evidence,
    create_grouped_splits,
    discover_sipakmed_cropped,
)

RANDOM_STATE = 42
PROTOTYPE_VERSION = "v0.1-trl3-ood"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ARCHIVE2_ROOT = PROJECT_ROOT / "archivebase2" / "Multi Cancer" / "Multi Cancer"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
SAMPLE_OOD_DIR = BASE_DIR / "data" / "sample_ood_images"

CNN_MODEL_PATH = MODELS_DIR / "cnn_sipakmed_trl3_v01.pt"

NON_CERVICAL_ORGANS = [
    "ALL",
    "Brain Cancer",
    "Breast Cancer",
    "Kidney Cancer",
    "Lung and Colon Cancer",
    "Lymphoma",
    "Oral Cancer",
]


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PathImageDataset(Dataset):
    def __init__(self, paths: List[str], tf):
        self.paths = paths
        self.tf = tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img)


def sample_non_cervical_dataset(
    root: Path,
    train_per_sub: int = 120,
    val_per_sub: int = 20,
    test_per_sub: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Muestrea de forma deterministica las 21 subclases no cervicales de archivebase2."""
    rng = random.Random(RANDOM_STATE)
    tr_rows, va_rows, te_rows = [], [], []

    for organ in NON_CERVICAL_ORGANS:
        organ_dir = root / organ
        if not organ_dir.exists():
            raise FileNotFoundError(f"No se encontro la carpeta {organ_dir}")
        for sub_dir in sorted(p for p in organ_dir.iterdir() if p.is_dir()):
            imgs = sorted(sub_dir.glob("*.jpg"))
            needed = train_per_sub + val_per_sub + test_per_sub
            chosen = rng.sample(imgs, min(needed, len(imgs)))
            tr = chosen[:train_per_sub]
            va = chosen[train_per_sub : train_per_sub + val_per_sub]
            te = chosen[train_per_sub + val_per_sub : needed]
            for p in tr:
                tr_rows.append({"image_path": str(p), "organ": organ, "subclass": sub_dir.name, "is_cervical": 0})
            for p in va:
                va_rows.append({"image_path": str(p), "organ": organ, "subclass": sub_dir.name, "is_cervical": 0})
            for p in te:
                te_rows.append({"image_path": str(p), "organ": organ, "subclass": sub_dir.name, "is_cervical": 0})

    return pd.DataFrame(tr_rows), pd.DataFrame(va_rows), pd.DataFrame(te_rows)


def generate_synthetic_ood_tensors(tf, n_samples: int = 60) -> torch.Tensor:
    """Genera imagenes sinteticas no medicas (ruido, colores solidos, gradientes, cuadriculas) para stress-test."""
    rng = np.random.default_rng(RANDOM_STATE)
    tensors = []
    for i in range(n_samples):
        mode = i % 4
        if mode == 0:
            arr = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        elif mode == 1:
            color = rng.integers(0, 256, size=(1, 1, 3), dtype=np.uint8)
            arr = np.broadcast_to(color, (224, 224, 3)).copy()
        elif mode == 2:
            x = np.linspace(0, 255, 224, dtype=np.uint8)
            g = np.tile(x, (224, 1))
            arr = np.stack([g, np.flip(g, axis=1), np.full_like(g, (i * 37) % 255)], axis=-1)
        else:
            tile = np.indices((224, 224)).sum(axis=0) // 16 % 2
            arr = np.stack([tile * 240, (1 - tile) * 200, tile * 120], axis=-1).astype(np.uint8)
        tensors.append(tf(Image.fromarray(arr)))
    return torch.stack(tensors, dim=0)


def load_trained_mobilenet_extractor(model_path: Path):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.25), nn.Linear(in_features, len(ckpt["classes"])))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    eval_tf = transforms.Compose([
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=ckpt["mean"], std=ckpt["std"]),
    ])
    return model, ckpt, eval_tf


@torch.no_grad()
def extract_embeddings(model: nn.Module, paths: List[str], tf, batch_size: int = 64) -> np.ndarray:
    loader = DataLoader(PathImageDataset(paths, tf), batch_size=batch_size, shuffle=False, num_workers=0)
    feats = []
    for batch in loader:
        x = model.features(batch)
        pooled = x.mean(dim=[2, 3])
        feats.append(pooled.cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def extract_embeddings_from_tensor(model: nn.Module, tensor_batch: torch.Tensor) -> np.ndarray:
    x = model.features(tensor_batch)
    return x.mean(dim=[2, 3]).cpu().numpy()


def compute_max_cosine_similarity(embs: np.ndarray, norm_centroids: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    norm_embs = embs / norms
    sims = norm_embs @ norm_centroids.T
    return sims.max(axis=1)


def export_sample_ood_images(te_ood_df: pd.DataFrame) -> List[dict]:
    SAMPLE_OOD_DIR.mkdir(parents=True, exist_ok=True)
    picks = ["Brain Cancer", "Kidney Cancer", "Lung and Colon Cancer"]
    exported = []
    for organ in picks:
        row = te_ood_df[te_ood_df["organ"] == organ].iloc[0]
        src = Path(row["image_path"])
        dst = SAMPLE_OOD_DIR / f"ood_{row['subclass']}_{src.name}"
        shutil.copy2(src, dst)
        exported.append({"filename": dst.name, "organ": organ, "subclass": row["subclass"]})
    return exported


def plot_ood_confusion_and_organs(cm: np.ndarray, looo_results: dict, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), dpi=150)

    # Panel 1: Matriz de confusion
    im = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    labels = ["Fuera de Dominio (0)", "Citologia Cervical (1)"]
    axes[0].set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Dominio Real",
        xlabel="Decision del Filtro OOD",
        title="Matriz de Confusion - Filtro OOD (Test Retenido)",
    )
    thresh = cm.max() / 2.0
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            axes[0].text(
                j, i, f"{cm[i, j]}\n({pct:.1f}%)",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "#0f172a",
                fontsize=10.5, fontweight="bold",
            )

    # Panel 2: Tasa de rechazo Leave-One-Organ-Out (organos nunca vistos)
    organs = list(looo_results.keys())
    rates = [looo_results[o]["tasa_rechazo_ood"] * 100 for o in organs]
    short_organs = [o.replace(" Cancer", "").replace("Lung and Colon", "Lung/Colon") for o in organs]
    bars = axes[1].barh(short_organs, rates, color="#0ea5e9", edgecolor="#0369a1")
    axes[1].set_xlim(0, 108)
    axes[1].set_xlabel("Tasa de Rechazo Correcto (%)")
    axes[1].set_title("Generalizacion OOD: Leave-One-Organ-Out\n(Organos NUNCA vistos al entrenar el filtro)")
    axes[1].grid(axis="x", alpha=0.25)
    for bar, r in zip(bars, rates):
        axes[1].text(r + 1.0, bar.get_y() + bar.get_height() / 2, f"{r:.1f}%", va="center", fontsize=8.8, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_ood_distributions(
    cerv_probs: np.ndarray,
    ood_probs: np.ndarray,
    synth_probs: np.ndarray,
    cerv_cos: np.ndarray,
    ood_cos: np.ndarray,
    synth_cos: np.ndarray,
    prob_thr: float,
    cos_thr: float,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), dpi=150)

    axes[0].hist(ood_probs, bins=25, alpha=0.65, color="#ef4444", label="7 Organos No Cervicales (Test)")
    axes[0].hist(synth_probs, bins=15, alpha=0.65, color="#f59e0b", label="Imagenes Sinteticas No Medicas")
    axes[0].hist(cerv_probs, bins=25, alpha=0.70, color="#10b981", label="Citologia Cervical SIPaKMeD (Test)")
    axes[0].axvline(prob_thr, color="#0f172a", linestyle="--", lw=1.6, label=f"Umbral prob = {prob_thr:.2f}")
    axes[0].set_xlabel("Probabilidad de Dominio Cervical (Filtro OOD)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Separacion por Probabilidad del Filtro OOD")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].hist(ood_cos, bins=25, alpha=0.65, color="#ef4444", label="7 Organos No Cervicales (Test)")
    axes[1].hist(synth_cos, bins=15, alpha=0.65, color="#f59e0b", label="Imagenes Sinteticas No Medicas")
    axes[1].hist(cerv_cos, bins=25, alpha=0.70, color="#10b981", label="Citologia Cervical SIPaKMeD (Test)")
    axes[1].axvline(cos_thr, color="#0f172a", linestyle="--", lw=1.6, label=f"Umbral coseno = {cos_thr:.3f}")
    axes[1].set_xlabel("Similitud Coseno Maxima a Centroides Cervicales")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Separacion Geometrica en Espacio de Embeddings (1,280 dim)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    seed_everything(RANDOM_STATE)
    torch.set_num_threads(min(8, os.cpu_count() or 4))

    print("[INFO] Cargando MobileNetV2 entrenado como extractor de embeddings (1,280 dim)...")
    model, ckpt, eval_tf = load_trained_mobilenet_extractor(CNN_MODEL_PATH)

    # 1. Positivos (Citologia Cervical): exactamente el mismo split agrupado por lamina
    sipak_df = discover_sipakmed_cropped(SIPAKMED_ROOT)
    tr_cerv_df, va_cerv_df, te_cerv_df = create_grouped_splits(sipak_df)

    # 2. Negativos (7 organos no cervicales de archivebase2)
    tr_ood_df, va_ood_df, te_ood_df = sample_non_cervical_dataset(
        ARCHIVE2_ROOT, train_per_sub=120, val_per_sub=20, test_per_sub=30
    )
    print(
        f"[INFO] Positivos SIPaKMeD -> Train={len(tr_cerv_df)}, Val={len(va_cerv_df)}, Test={len(te_cerv_df)}\n"
        f"[INFO] Negativos archivebase2 (7 organos, 21 subclases) -> Train={len(tr_ood_df)}, Val={len(va_ood_df)}, Test={len(te_ood_df)}"
    )

    sample_ood_meta = export_sample_ood_images(te_ood_df)

    # 3. Extraccion de embeddings (un solo pase forward por imagen)
    t0 = time.perf_counter()
    print("[INFO] Extrayendo embeddings de entrenamiento, validacion y prueba...")
    emb_tr_cerv = extract_embeddings(model, tr_cerv_df["image_path"].tolist(), eval_tf)
    emb_va_cerv = extract_embeddings(model, va_cerv_df["image_path"].tolist(), eval_tf)
    emb_te_cerv = extract_embeddings(model, te_cerv_df["image_path"].tolist(), eval_tf)

    emb_tr_ood = extract_embeddings(model, tr_ood_df["image_path"].tolist(), eval_tf)
    emb_va_ood = extract_embeddings(model, va_ood_df["image_path"].tolist(), eval_tf)
    emb_te_ood = extract_embeddings(model, te_ood_df["image_path"].tolist(), eval_tf)

    synth_tensors = generate_synthetic_ood_tensors(eval_tf, n_samples=60)
    emb_synth = extract_embeddings_from_tensor(model, synth_tensors)
    ext_sec = time.perf_counter() - t0
    print(f"[OK] Embeddings extraidos en {ext_sec:.1f} s")

    # 4. Centroides de las 5 clases cervicales (calculados SOLO sobre SIPaKMeD Train)
    centroids = []
    tr_labels = tr_cerv_df["label"].values
    for idx in range(len(CLASSES)):
        c_vec = emb_tr_cerv[tr_labels == idx].mean(axis=0)
        c_vec = c_vec / (np.linalg.norm(c_vec) + 1e-8)
        centroids.append(c_vec)
    norm_centroids = np.vstack(centroids)

    # Calibrar umbral de similitud coseno usando percentil 0.5 de SIPaKMeD Val
    va_cerv_cos = compute_max_cosine_similarity(emb_va_cerv, norm_centroids)
    va_ood_cos = compute_max_cosine_similarity(emb_va_ood, norm_centroids)
    cos_threshold = float(max(0.45, np.percentile(va_cerv_cos, 0.5) - 0.03))
    print(
        f"[INFO] Similitud coseno Val -> Cervical p0.5={np.percentile(va_cerv_cos, 0.5):.4f}, "
        f"mediana={np.median(va_cerv_cos):.4f} | OOD mediana={np.median(va_ood_cos):.4f} | Umbral={cos_threshold:.4f}"
    )

    # 5. Clasificador OOD calibrado sobre embeddings
    X_tr = np.vstack([emb_tr_cerv, emb_tr_ood])
    y_tr = np.concatenate([np.ones(len(emb_tr_cerv), dtype=int), np.zeros(len(emb_tr_ood), dtype=int)])

    ood_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])
    ood_pipe.fit(X_tr, y_tr)
    prob_threshold = 0.50

    # 6. Evaluacion en conjunto de prueba retenido (SIPaKMeD Test vs 7 organos Test)
    X_te = np.vstack([emb_te_cerv, emb_te_ood])
    y_te = np.concatenate([np.ones(len(emb_te_cerv), dtype=int), np.zeros(len(emb_te_ood), dtype=int)])
    te_probs = ood_pipe.predict_proba(X_te)[:, 1]
    te_cos = compute_max_cosine_similarity(X_te, norm_centroids)
    te_preds = ((te_probs >= prob_threshold) & (te_cos >= cos_threshold)).astype(int)

    cm = confusion_matrix(y_te, te_preds)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    te_recall = recall_score(y_te, te_preds)
    te_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    te_prec = precision_score(y_te, te_preds)
    te_f1 = f1_score(y_te, te_preds)
    te_roc = roc_auc_score(y_te, te_probs)
    te_pr = average_precision_score(y_te, te_probs)

    # Stress-test con imagenes sinteticas no medicas
    synth_probs = ood_pipe.predict_proba(emb_synth)[:, 1]
    synth_cos = compute_max_cosine_similarity(emb_synth, norm_centroids)
    synth_rejected = int(((synth_probs < prob_threshold) | (synth_cos < cos_threshold)).sum())

    # 7. Evaluacion rigurosa Leave-One-Organ-Out (LOOO): rechazo de organos NUNCA vistos
    print("\n=== Evaluacion Leave-One-Organ-Out (organos nunca vistos al entrenar OOD) ===")
    looo_results = {}
    all_ood_embs = np.vstack([emb_tr_ood, emb_te_ood])
    all_ood_organs = np.concatenate([tr_ood_df["organ"].values, te_ood_df["organ"].values])
    for held_organ in NON_CERVICAL_ORGANS:
        mask_held = all_ood_organs == held_organ
        X_looo_tr = np.vstack([emb_tr_cerv, all_ood_embs[~mask_held]])
        y_looo_tr = np.concatenate([np.ones(len(emb_tr_cerv), dtype=int), np.zeros(int((~mask_held).sum()), dtype=int)])
        looo_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
        ])
        looo_pipe.fit(X_looo_tr, y_looo_tr)
        held_embs = all_ood_embs[mask_held]
        held_probs = looo_pipe.predict_proba(held_embs)[:, 1]
        held_cos = compute_max_cosine_similarity(held_embs, norm_centroids)
        rejected = int(((held_probs < prob_threshold) | (held_cos < cos_threshold)).sum())
        rej_rate = rejected / len(held_embs)
        looo_results[held_organ] = {
            "imagenes_evaluadas": int(len(held_embs)),
            "rechazadas_como_ood": rejected,
            "tasa_rechazo_ood": round(float(rej_rate), 4),
        }
        print(f"  - Organo no visto [{held_organ}]: rechazo OOD = {rejected}/{len(held_embs)} ({rej_rate*100:.2f}%)")

    # 8. Benchmark de latencia adicional del filtro OOD sobre un embedding ya calculado
    sample_emb = emb_te_cerv[:1]
    latencies = []
    for _ in range(200):
        ts = time.perf_counter()
        _p = float(ood_pipe.predict_proba(sample_emb)[0, 1])
        _c = float(compute_max_cosine_similarity(sample_emb, norm_centroids)[0])
        latencies.append((time.perf_counter() - ts) * 1000.0)
    ood_overhead_ms = float(np.mean(latencies))

    print("\n=== RESULTADOS FILTRO DE VALIDEZ DE DOMINIO (TEST RETENIDO) ===")
    print(f"  Aceptacion citologia valida (Recall):   {te_recall*100:.2f}% ({tp}/{tp+fn})")
    print(f"  Rechazo no cervical (Especificidad):    {te_spec*100:.2f}% ({tn}/{tn+fp})")
    print(f"  Precision: {te_prec:.4f} | F1: {te_f1:.4f} | ROC-AUC: {te_roc:.4f} | PR-AUC: {te_pr:.4f}")
    print(f"  Rechazo sinteticas no medicas:          {synth_rejected}/{len(emb_synth)} ({synth_rejected/len(emb_synth)*100:.1f}%)")
    print(f"  Sobrecosto de latencia OOD (CPU):       {ood_overhead_ms:.3f} ms")

    # 9. Graficas
    cm_png = OUTPUTS_DIR / "ood_confusion_matrix.png"
    dist_png = OUTPUTS_DIR / "ood_score_distribution.png"
    plot_ood_confusion_and_organs(cm, looo_results, cm_png)
    plot_ood_distributions(
        te_probs[: len(emb_te_cerv)],
        te_probs[len(emb_te_cerv) :],
        synth_probs,
        te_cos[: len(emb_te_cerv)],
        te_cos[len(emb_te_cerv) :],
        synth_cos,
        prob_threshold,
        cos_threshold,
        dist_png,
    )

    # 10. Guardar modelo OOD ligero (~30 KB)
    ood_model_path = MODELS_DIR / "ood_domain_gate_v01.joblib"
    joblib.dump(
        {
            "version": PROTOTYPE_VERSION,
            "pipeline": ood_pipe,
            "class_centroids": norm_centroids,
            "classes": CLASSES,
            "prob_threshold": prob_threshold,
            "cosine_threshold": round(cos_threshold, 4),
            "embedding_dim": int(emb_tr_cerv.shape[1]),
        },
        ood_model_path,
    )

    # 11. Evidencia JSON
    evidencia_ood = {
        "identificacion": {
            "prototipo": "Prototipo IA-Tamizaje Cervical",
            "modulo": "ood_validez_dominio_citologico",
            "version": PROTOTYPE_VERSION,
            "fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(),
            "random_state": RANDOM_STATE,
            "entorno": {
                "os": platform.platform(),
                "python": sys.version.split()[0],
                "pytorch": torch.__version__,
            },
        },
        "datos": {
            "positivos_in_domain": "SIPaKMeD CROPPED (misma particion sin fuga por lamina que modulo CNN)",
            "negativos_out_of_domain": "archivebase2 (Multi Cancer Dataset) — 7 organos NO cervicales (21 subclases); Cervical Cancer de archivebase2 excluido explicitamente para evitar data leakage",
            "organos_negativos_usados": NON_CERVICAL_ORGANS,
            "conteos": {
                "train_cervical": int(len(tr_cerv_df)),
                "train_no_cervical": int(len(tr_ood_df)),
                "val_cervical": int(len(va_cerv_df)),
                "val_no_cervical": int(len(va_ood_df)),
                "test_cervical": int(len(te_cerv_df)),
                "test_no_cervical": int(len(te_ood_df)),
                "test_sinteticas_no_medicas": int(len(emb_synth)),
            },
            "muestras_ood_exportadas": sample_ood_meta,
        },
        "arquitectura": {
            "extractor_compartido": "MobileNetV2 (mismo pase forward del clasificador CNN, 1,280 dim, cero pase CNN adicional)",
            "doble_compuerta": [
                f"Clasificador calibrado StandardScaler + LogisticRegression(C=0.5, balanced) >= {prob_threshold}",
                f"Similitud coseno maxima a los 5 centroides citologicos de entrenamiento >= {round(cos_threshold, 4)}",
            ],
        },
        "metricas_test": {
            "aceptacion_citologia_valida_recall": round(float(te_recall), 4),
            "rechazo_no_cervical_especificidad": round(float(te_spec), 4),
            "precision": round(float(te_prec), 4),
            "f1_score": round(float(te_f1), 4),
            "roc_auc": round(float(te_roc), 4),
            "pr_auc": round(float(te_pr), 4),
            "matriz_confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "stress_test_sinteticas_rechazo": f"{synth_rejected}/{len(emb_synth)} ({round(synth_rejected/len(emb_synth)*100, 1)}%)",
            "validacion_leave_one_organ_out": looo_results,
            "sobrecosto_latencia_ms": round(ood_overhead_ms, 3),
        },
        "artefactos_generados": {
            "modelo_ood": str(ood_model_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "matriz_confusion_looo_png": str(cm_png.relative_to(BASE_DIR)).replace("\\", "/"),
            "distribucion_scores_png": str(dist_png.relative_to(BASE_DIR)).replace("\\", "/"),
        },
    }

    out_json = OUTPUTS_DIR / "evidencia_ood_trl3.json"
    out_json.write_text(json.dumps(evidencia_ood, indent=2, ensure_ascii=False), encoding="utf-8")

    # Actualizar evidencia_trl3.json consolidada incluyendo modulo OOD
    unified_path = OUTPUTS_DIR / "evidencia_trl3.json"
    if unified_path.exists():
        unified = json.loads(unified_path.read_text(encoding="utf-8"))
    else:
        build_unified_trl3_evidence()
        unified = json.loads(unified_path.read_text(encoding="utf-8"))

    if "ood_validez_dominio_citologico" not in unified["modulos_disponibles"]:
        unified["modulos_disponibles"].append("ood_validez_dominio_citologico")
    unified["modulo_ood"] = evidencia_ood
    unified["fecha_consolidacion_utc"] = datetime.now(timezone.utc).isoformat()
    unified_path.write_text(json.dumps(unified, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[OK] Modelo OOD guardado: {ood_model_path} ({ood_model_path.stat().st_size/1024:.1f} KB)")
    print(f"[OK] Evidencia OOD guardada: {out_json}")
    print(f"[OK] Evidencia consolidada actualizada: {unified_path}")


if __name__ == "__main__":
    main()
