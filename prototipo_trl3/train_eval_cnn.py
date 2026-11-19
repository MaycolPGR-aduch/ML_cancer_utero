"""
Prototipo IA-Tamizaje Cervical v0.1 (TRL 3) - Modulo CNN (Citologia SIPaKMeD)
=============================================================================
Cumple todos los requisitos de evidencia experimental TRL 3:
- Carga las 4,049 imagenes recortadas (CROPPED) de SIPaKMeD en 5 clases celulares
- Particion estratificada agrupada por lamina (StratifiedGroupKFold) SIN fuga de datos
  (celulas provenientes de la misma lamina microscopica nunca se mezclan entre train/val/test)
- Preprocesamiento a 224x224, normalizacion ImageNet y data augmentation exclusivo en train
- Soporte y verificacion visual de escalado de coordenadas auxiliares (*_cyt.dat y *_nuc.dat)
- Transfer Learning en 2 etapas sobre MobileNetV2 (ImageNet) con semilla fija (42)
- Evaluacion rigurosa en conjunto de prueba: Precision, Recall, F1, ROC-AUC, PR-AUC,
  Matriz de Confusion 5x5, metricas binarias de tamizaje (Anormal vs Normal/Benigno),
  explicabilidad Grad-CAM y benchmark de latencia por imagen
- Genera artefactos trazables: models/cnn_sipakmed_trl3_v01.pt, graficas PNG,
  outputs/evidencia_cnn_trl3.json y evidencia unificada outputs/evidencia_trl3.json.
"""

import argparse
import json
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from gradcam import MobileNetV2GradCAM

RANDOM_STATE = 42
PROTOTYPE_VERSION = "v0.1-trl3-cnn"
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SIPAKMED_ROOT = PROJECT_ROOT / "archive_base1"
DATA_DIR = BASE_DIR / "data"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

CLASSES = [
    "im_Dyskeratotic",
    "im_Koilocytotic",
    "im_Metaplastic",
    "im_Parabasal",
    "im_Superficial-Intermediate",
]

CLASS_CLINICAL_INFO = {
    "im_Dyskeratotic": {
        "nombre_es": "Disqueratosica (Anormal)",
        "categoria_tamizaje": "Anormal",
        "descripcion": "Celula escamosa anormal con queratinizacion atipica y alteracion nuclear; requiere revision citopatologica prioritaria.",
    },
    "im_Koilocytotic": {
        "nombre_es": "Coilocitica (Anormal / VPH)",
        "categoria_tamizaje": "Anormal",
        "descripcion": "Celula con halo perinuclear caracteristico asociado a infeccion por VPH y lesion intraepitelial escamosa de bajo grado (LSIL).",
    },
    "im_Metaplastic": {
        "nombre_es": "Metaplasica (Benigna)",
        "categoria_tamizaje": "Benigna",
        "descripcion": "Celula de metaplasia escamosa benigna, comun en la zona de transformacion cervical.",
    },
    "im_Parabasal": {
        "nombre_es": "Parabasal (Normal)",
        "categoria_tamizaje": "Normal",
        "descripcion": "Celula epitelial normal de capas profundas del epitelio escamoso.",
    },
    "im_Superficial-Intermediate": {
        "nombre_es": "Superficial-Intermedia (Normal)",
        "categoria_tamizaje": "Normal",
        "descripcion": "Celula epitelial escamosa madura normal de capas superficiales o intermedias.",
    },
}

ABNORMAL_CLASS_INDICES = [0, 1]  # im_Dyskeratotic, im_Koilocytotic


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scale_dat_coordinates(dat_path: Path, orig_w: int, orig_h: int, target_size: int = IMG_SIZE) -> np.ndarray:
    """Lee un archivo *_cyt.dat o *_nuc.dat y escala sus coordenadas (x, y) a target_size x target_size."""
    if not dat_path.exists():
        return np.empty((0, 2), dtype=np.float32)
    coords = []
    for line in dat_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        parts = line.split(",")
        try:
            x, y = float(parts[0]), float(parts[1])
            coords.append([x * (target_size / float(orig_w)), y * (target_size / float(orig_h))])
        except ValueError:
            continue
    return np.array(coords, dtype=np.float32) if coords else np.empty((0, 2), dtype=np.float32)


def discover_sipakmed_cropped(root: Path) -> pd.DataFrame:
    rows = []
    for idx, cls in enumerate(CLASSES):
        cropped_dir = root / cls / cls / "CROPPED"
        if not cropped_dir.exists():
            # Fallback por si el usuario descomprimio sin nivel duplicado
            alt_dir = root / cls / "CROPPED"
            if alt_dir.exists():
                cropped_dir = alt_dir
            else:
                raise FileNotFoundError(f"No se encontro la carpeta CROPPED para {cls} en {root}")

        bmps = sorted(cropped_dir.glob("*.bmp"))
        for bmp_path in bmps:
            stem = bmp_path.stem  # ej. 001_01
            slide_id = stem.split("_")[0]
            cyt_path = cropped_dir / f"{stem}_cyt.dat"
            nuc_path = cropped_dir / f"{stem}_nuc.dat"
            rows.append(
                {
                    "image_path": str(bmp_path),
                    "class_name": cls,
                    "label": idx,
                    "slide_id": slide_id,
                    "slide_group": f"{cls}__{slide_id}",
                    "cyt_dat": str(cyt_path) if cyt_path.exists() else "",
                    "nuc_dat": str(nuc_path) if nuc_path.exists() else "",
                }
            )
    return pd.DataFrame(rows)


def create_grouped_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide en ~71.4% train / ~14.3% val / ~14.3% test usando StratifiedGroupKFold(n_splits=7)
    agrupado por slide_group (clase + id de lamina), garantizando CERO fuga entre splits.
    """
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=RANDOM_STATE)
    folds = np.zeros(len(df), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(df, df["label"], groups=df["slide_group"])):
        folds[val_idx] = fold_idx

    df = df.copy()
    df["fold"] = folds
    test_df = df[df["fold"] == 0].reset_index(drop=True)
    val_df = df[df["fold"] == 1].reset_index(drop=True)
    train_df = df[df["fold"] >= 2].reset_index(drop=True)

    # Verificacion estricta anti-leakage
    g_tr = set(train_df["slide_group"])
    g_va = set(val_df["slide_group"])
    g_te = set(test_df["slide_group"])
    assert len(g_tr & g_va) == 0, "Fuga detectada entre train y val!"
    assert len(g_tr & g_te) == 0, "Fuga detectada entre train y test!"
    assert len(g_va & g_te) == 0, "Fuga detectada entre val y test!"
    return train_df, val_df, test_df


class SipakmedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.paths = df["image_path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def build_mobilenet_v2(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features, num_classes),
    )
    return model


def save_sample_annotations_and_files(test_df: pd.DataFrame, out_png: Path) -> List[dict]:
    """
    Exporta 1 imagen de prueba por clase a data/sample_images/ (para la app web y CLI)
    y genera una figura demostrando el escalado geometrico de *_cyt.dat y *_nuc.dat a 224x224.
    """
    SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(15, 3.4), dpi=150)
    exported = []

    for i, cls in enumerate(CLASSES):
        row = test_df[test_df["class_name"] == cls].iloc[0]
        src_img = Path(row["image_path"])
        orig_pil = Image.open(src_img).convert("RGB")
        orig_w, orig_h = orig_pil.size
        resized = orig_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

        # Guardar copia PNG/BMP ligera en sample_images para pruebas en la app web
        dst_img = SAMPLE_IMAGES_DIR / f"{cls}_{src_img.name}"
        shutil.copy2(src_img, dst_img)
        if row["cyt_dat"]:
            shutil.copy2(row["cyt_dat"], SAMPLE_IMAGES_DIR / f"{cls}_{Path(row['cyt_dat']).name}")
        if row["nuc_dat"]:
            shutil.copy2(row["nuc_dat"], SAMPLE_IMAGES_DIR / f"{cls}_{Path(row['nuc_dat']).name}")

        cyt_pts = scale_dat_coordinates(Path(row["cyt_dat"]), orig_w, orig_h, IMG_SIZE) if row["cyt_dat"] else np.empty((0, 2))
        nuc_pts = scale_dat_coordinates(Path(row["nuc_dat"]), orig_w, orig_h, IMG_SIZE) if row["nuc_dat"] else np.empty((0, 2))

        ax = axes[i]
        ax.imshow(resized)
        if len(cyt_pts) > 2:
            closed_cyt = np.vstack([cyt_pts, cyt_pts[0]])
            ax.plot(closed_cyt[:, 0], closed_cyt[:, 1], color="#22d3ee", lw=1.6, label="Citoplasma (_cyt.dat)")
        if len(nuc_pts) > 2:
            closed_nuc = np.vstack([nuc_pts, nuc_pts[0]])
            ax.plot(closed_nuc[:, 0], closed_nuc[:, 1], color="#facc15", lw=1.6, label="Nucleo (_nuc.dat)")
        ax.set_title(f"{cls.replace('im_', '')}\n({orig_w}x{orig_h} -> 224x224)", fontsize=8.5, fontweight="bold")
        ax.axis("off")
        if i == 0:
            ax.legend(loc="lower center", fontsize=7, framealpha=0.8)

        exported.append(
            {
                "filename": dst_img.name,
                "class_name": cls,
                "orig_size": [orig_w, orig_h],
                "cyt_points_scaled": int(len(cyt_pts)),
                "nuc_points_scaled": int(len(nuc_pts)),
            }
        )

    fig.suptitle("Verificacion de Preprocesamiento Geometrico (224x224) y Escalado de Anotaciones .dat", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return exported


def plot_training_history(history: List[dict], out_png: Path) -> None:
    epochs = [h["epoch"] for h in history]
    tr_loss = [h["train_loss"] for h in history]
    va_loss = [h["val_loss"] for h in history]
    va_f1 = [h["val_macro_f1"] for h in history]
    va_acc = [h["val_accuracy"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150)
    axes[0].plot(epochs, tr_loss, marker="o", label="Train Loss", color="#0284c7")
    axes[0].plot(epochs, va_loss, marker="s", label="Val Loss", color="#f97316")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Curva de Perdida (Entrenamiento vs Validacion)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, va_f1, marker="o", label="Val Macro-F1", color="#10b981", lw=2)
    axes[1].plot(epochs, va_acc, marker="^", label="Val Accuracy", color="#6366f1", lw=1.8)
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Puntaje")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Desempeno en Validacion por Epoca")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_cnn_confusion_matrix(cm: np.ndarray, out_png: Path) -> None:
    short_names = [c.replace("im_", "").replace("Superficial-Intermediate", "Superf.-Interm.") for c in CLASSES]
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(short_names)),
        yticks=np.arange(len(short_names)),
        xticklabels=short_names,
        yticklabels=short_names,
        ylabel="Clase Real (Test sin fuga por lamina)",
        xlabel="Clase Predicha (MobileNetV2)",
        title="Matriz de Confusion - Conjunto de Prueba SIPaKMeD",
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")

    row_sums = cm.sum(axis=1, keepdims=True)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = (cm[i, j] / row_sums[i, 0] * 100) if row_sums[i, 0] > 0 else 0
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "#0f172a",
                fontsize=8.5,
                fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_cnn_roc_pr(y_true: np.ndarray, y_probs: np.ndarray, out_png: Path) -> None:
    y_bin = label_binarize(y_true, classes=list(range(len(CLASSES))))
    colors = ["#ef4444", "#f59e0b", "#0ea5e9", "#8b5cf6", "#10b981"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    for i, cls in enumerate(CLASSES):
        short = cls.replace("im_", "")
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        auc_val = roc_auc_score(y_bin[:, i], y_probs[:, i])
        axes[0].plot(fpr, tpr, lw=1.9, color=colors[i], label=f"{short} (AUC={auc_val:.4f})")

        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap_val = average_precision_score(y_bin[:, i], y_probs[:, i])
        axes[1].plot(rec, prec, lw=1.9, color=colors[i], label=f"{short} (PR-AUC={ap_val:.4f})")

    axes[0].plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1)
    axes[0].set_title("Curvas ROC Multiclase (One-vs-Rest) - Test")
    axes[0].set_xlabel("Tasa de Falsos Positivos")
    axes[0].set_ylabel("Sensibilidad (Recall)")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Curvas Precision-Recall Multiclase - Test")
    axes[1].set_xlabel("Sensibilidad (Recall)")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def save_gradcam_showcase(model: nn.Module, test_df: pd.DataFrame, eval_tf, device: torch.device, out_png: Path) -> None:
    gradcam = MobileNetV2GradCAM(model)
    fig, axes = plt.subplots(2, len(CLASSES), figsize=(14.5, 6.0), dpi=150)
    for i, cls in enumerate(CLASSES):
        row = test_df[test_df["class_name"] == cls].iloc[0]
        pil_img = Image.open(row["image_path"]).convert("RGB")
        inp = eval_tf(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(inp), dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        overlay, _ = gradcam.generate(inp, pil_img, target_class=pred_idx, alpha=0.45)

        axes[0, i].imshow(pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        axes[0, i].set_title(f"Real: {cls.replace('im_', '')}", fontsize=8.5, fontweight="bold")
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay)
        axes[1, i].set_title(
            f"Pred: {CLASSES[pred_idx].replace('im_', '')}\nConf: {probs[pred_idx]*100:.1f}%",
            fontsize=8.5,
            color="#059669" if pred_idx == row["label"] else "#dc2626",
        )
        axes[1, i].axis("off")

    fig.suptitle("Explicabilidad Visual (Grad-CAM) sobre Imagenes del Conjunto de Prueba", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * len(labels)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_probs = np.concatenate(all_probs)
    y_pred = np.argmax(y_probs, axis=1)
    avg_loss = total_loss / len(y_true)
    return avg_loss, y_true, y_pred, y_probs


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        total_n += len(labels)
    return total_loss / max(1, total_n)


def build_unified_trl3_evidence() -> None:
    """Combina evidencia tabular + CNN en un unico archivo outputs/evidencia_trl3.json."""
    tab_path = OUTPUTS_DIR / "evidencia_tabular_trl3.json"
    cnn_path = OUTPUTS_DIR / "evidencia_cnn_trl3.json"
    unified = {
        "prototipo": "Prototipo IA-Tamizaje Cervical v0.1 (TRL 3)",
        "nivel_madurez": "TRL 3 - Prueba de concepto analitica y experimental en entorno controlado",
        "fecha_consolidacion_utc": datetime.now(timezone.utc).isoformat(),
        "modulos_disponibles": [],
    }
    if cnn_path.exists():
        unified["modulos_disponibles"].append("cnn_citologia_sipakmed")
        unified["modulo_cnn"] = json.loads(cnn_path.read_text(encoding="utf-8"))
    if tab_path.exists():
        unified["modulos_disponibles"].append("tabular_factores_riesgo")
        unified["modulo_tabular"] = json.loads(tab_path.read_text(encoding="utf-8"))

    out_path = OUTPUTS_DIR / "evidencia_trl3.json"
    out_path.write_text(json.dumps(unified, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Evidencia TRL 3 consolidada guardada en: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento y evaluacion TRL 3 - CNN SIPaKMeD")
    parser.add_argument("--epochs-head", type=int, default=2, help="Epocas Fase 1 (backbone congelado)")
    parser.add_argument("--epochs-finetune", type=int, default=6, help="Epocas Fase 2 (fine-tuning bloques superiores)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    seed_everything(RANDOM_STATE)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Optimizar hilos CPU en Intel hibrido (12 cores / 16 threads)
    if not torch.cuda.is_available():
        torch.set_num_threads(min(8, os.cpu_count() or 4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo de ejecucion: {device} | PyTorch {torch.__version__} | Hilos CPU: {torch.get_num_threads()}")

    # 1. Descubrimiento del dataset SIPaKMeD CROPPED
    df = discover_sipakmed_cropped(SIPAKMED_ROOT)
    class_counts = df["class_name"].value_counts().reindex(CLASSES).to_dict()
    print(f"[INFO] Total imagenes CROPPED encontradas: {len(df)}")
    for c, n in class_counts.items():
        print(f"   - {c}: {n}")

    # 2. Split agrupado por lamina (cero leakage)
    train_df, val_df, test_df = create_grouped_splits(df)
    print(
        f"[INFO] Split sin fuga por lamina -> Train: {len(train_df)} ({train_df['slide_group'].nunique()} laminas), "
        f"Val: {len(val_df)} ({val_df['slide_group'].nunique()} laminas), "
        f"Test: {len(test_df)} ({test_df['slide_group'].nunique()} laminas)"
    )

    # 3. Verificacion visual de escalado de coordenadas *_cyt.dat y *_nuc.dat + copia de muestras
    ann_png = OUTPUTS_DIR / "cnn_sample_annotations_224.png"
    sample_meta = save_sample_annotations_and_files(test_df, ann_png)
    print(f"[OK] Grafica de escalado de anotaciones .dat guardada: {ann_png}")

    # 4. Transformaciones (augmentation SOLO en entrenamiento)
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_loader = DataLoader(SipakmedDataset(train_df, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(SipakmedDataset(val_df, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(SipakmedDataset(test_df, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Pesos de clase inversamente proporcionales a frecuencia en entrenamiento
    train_counts = train_df["label"].value_counts().sort_index().values.astype(np.float32)
    class_weights = torch.tensor(train_counts.sum() / (len(CLASSES) * train_counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 5. Modelo MobileNetV2 (Transfer Learning 2 etapas)
    model = build_mobilenet_v2(num_classes=len(CLASSES), pretrained=True).to(device)

    history = []
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    total_train_t0 = time.perf_counter()

    # Fase 1: Backbone congelado
    for p in model.features.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1.5e-3, weight_decay=1e-4)

    print(f"\n=== FASE 1: Entrenamiento de cabeza clasificadora ({args.epochs_head} epocas) ===")
    epoch_counter = 0
    for ep in range(1, args.epochs_head + 1):
        epoch_counter += 1
        t0 = time.perf_counter()
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_true, va_pred, _ = evaluate_loader(model, val_loader, criterion, device)
        va_acc = accuracy_score(va_true, va_pred)
        va_f1 = f1_score(va_true, va_pred, average="macro")
        elapsed = time.perf_counter() - t0
        history.append({
            "epoch": epoch_counter,
            "phase": "head_only",
            "train_loss": round(tr_loss, 4),
            "val_loss": round(va_loss, 4),
            "val_accuracy": round(float(va_acc), 4),
            "val_macro_f1": round(float(va_f1), 4),
            "seconds": round(elapsed, 1),
        })
        print(f"  Epoca {epoch_counter:02d} [Fase 1] | tr_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_acc={va_acc:.4f} | val_F1={va_f1:.4f} ({elapsed:.1f}s)")
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch_counter
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Fase 2: Fine-tuning de bloques convolucionales superiores (features[10:]) + classifier
    for idx_block, child in enumerate(model.features.children()):
        if idx_block >= 10:
            for p in child.parameters():
                p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=4e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs_finetune), eta_min=2e-5)

    print(f"\n=== FASE 2: Fine-tuning de bloques superiores ({args.epochs_finetune} epocas) ===")
    for ep in range(1, args.epochs_finetune + 1):
        epoch_counter += 1
        t0 = time.perf_counter()
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        va_loss, va_true, va_pred, _ = evaluate_loader(model, val_loader, criterion, device)
        va_acc = accuracy_score(va_true, va_pred)
        va_f1 = f1_score(va_true, va_pred, average="macro")
        elapsed = time.perf_counter() - t0
        history.append({
            "epoch": epoch_counter,
            "phase": "finetune_top_blocks",
            "train_loss": round(tr_loss, 4),
            "val_loss": round(va_loss, 4),
            "val_accuracy": round(float(va_acc), 4),
            "val_macro_f1": round(float(va_f1), 4),
            "seconds": round(elapsed, 1),
        })
        print(f"  Epoca {epoch_counter:02d} [Fase 2] | tr_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_acc={va_acc:.4f} | val_F1={va_f1:.4f} ({elapsed:.1f}s)")
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch_counter
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    total_train_sec = time.perf_counter() - total_train_t0

    # Cargar el mejor modelo segun val_macro_f1
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # 6. Evaluacion final en el conjunto de prueba (nunca visto)
    test_loss, y_true, y_pred, y_probs = evaluate_loader(model, test_loader, criterion, device)
    y_bin = label_binarize(y_true, classes=list(range(len(CLASSES))))

    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_roc_auc = float(roc_auc_score(y_bin, y_probs, average="macro", multi_class="ovr"))
    macro_pr_auc = float(average_precision_score(y_bin, y_probs, average="macro"))
    cm = confusion_matrix(y_true, y_pred)

    cls_rep = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    per_class_metrics = {}
    for i, cls in enumerate(CLASSES):
        per_class_metrics[cls] = {
            "precision": round(float(cls_rep[cls]["precision"]), 4),
            "recall": round(float(cls_rep[cls]["recall"]), 4),
            "f1_score": round(float(cls_rep[cls]["f1-score"]), 4),
            "roc_auc_ovr": round(float(roc_auc_score(y_bin[:, i], y_probs[:, i])), 4),
            "pr_auc_ovr": round(float(average_precision_score(y_bin[:, i], y_probs[:, i])), 4),
            "soporte_test": int(cls_rep[cls]["support"]),
        }

    # Agregacion clinica binaria de tamizaje: Anormal (Dyskeratotic + Koilocytotic) vs Normal/Benigna
    y_true_abn = np.isin(y_true, ABNORMAL_CLASS_INDICES).astype(int)
    y_prob_abn = y_probs[:, ABNORMAL_CLASS_INDICES].sum(axis=1)
    y_pred_abn = (y_prob_abn >= 0.5).astype(int)
    cm_bin = confusion_matrix(y_true_abn, y_pred_abn)
    b_tn, b_fp, b_fn, b_tp = [int(v) for v in cm_bin.ravel()]
    clinical_screening_metrics = {
        "definicion_anormal": ["im_Dyskeratotic", "im_Koilocytotic"],
        "definicion_normal_benigna": ["im_Metaplastic", "im_Parabasal", "im_Superficial-Intermediate"],
        "umbral_probabilidad_anormal": 0.5,
        "sensibilidad_recall": round(float(recall_score(y_true_abn, y_pred_abn, zero_division=0)), 4),
        "especificidad": round(float(b_tn / (b_tn + b_fp)) if (b_tn + b_fp) > 0 else 0.0, 4),
        "precision": round(float(precision_score(y_true_abn, y_pred_abn, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true_abn, y_pred_abn, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true_abn, y_prob_abn)), 4),
        "pr_auc": round(float(average_precision_score(y_true_abn, y_prob_abn)), 4),
        "matriz_confusion_binaria": {"tn": b_tn, "fp": b_fp, "fn": b_fn, "tp": b_tp},
    }

    print("\n=== RESULTADOS EN CONJUNTO DE PRUEBA (SIN FUGA POR LAMINA) ===")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  Macro Precision:{macro_prec:.4f}")
    print(f"  Macro Recall:   {macro_rec:.4f}")
    print(f"  Macro F1:       {macro_f1:.4f}")
    print(f"  Macro ROC-AUC:  {macro_roc_auc:.4f}")
    print(f"  Macro PR-AUC:   {macro_pr_auc:.4f}")
    print(
        f"  Tamizaje Binario (Anormal vs Normal/Benigna) -> "
        f"Sensibilidad={clinical_screening_metrics['sensibilidad_recall']:.4f}, "
        f"Especificidad={clinical_screening_metrics['especificidad']:.4f}, "
        f"ROC-AUC={clinical_screening_metrics['roc_auc']:.4f}"
    )

    # 7. Graficas de evidencia TRL 3
    hist_png = OUTPUTS_DIR / "cnn_training_history.png"
    cm_png = OUTPUTS_DIR / "cnn_confusion_matrix.png"
    roc_png = OUTPUTS_DIR / "cnn_roc_pr_curves.png"
    gradcam_png = OUTPUTS_DIR / "cnn_sample_gradcam.png"

    plot_training_history(history, hist_png)
    plot_cnn_confusion_matrix(cm, cm_png)
    plot_cnn_roc_pr(y_true, y_probs, roc_png)
    save_gradcam_showcase(model, test_df, eval_tf, device, gradcam_png)

    # 8. Guardar modelo PyTorch (.pt)
    pt_path = MODELS_DIR / "cnn_sipakmed_trl3_v01.pt"
    torch.save(
        {
            "version": PROTOTYPE_VERSION,
            "architecture": "MobileNetV2",
            "img_size": IMG_SIZE,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "classes": CLASSES,
            "clinical_info": CLASS_CLINICAL_INFO,
            "abnormal_class_indices": ABNORMAL_CLASS_INDICES,
            "best_epoch": best_epoch,
            "best_val_macro_f1": round(float(best_val_f1), 4),
            "state_dict": best_state,
        },
        pt_path,
    )

    # 9. Benchmark de latencia de inferencia por imagen en CPU
    sample_row = test_df.iloc[0]
    sample_img = Image.open(sample_row["image_path"]).convert("RGB")
    sample_tensor = eval_tf(sample_img).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(5):
            _ = model(sample_tensor)
        latencies = []
        for _ in range(30):
            t0 = time.perf_counter()
            out_logits = model(sample_tensor)
            out_probs = torch.softmax(out_logits, dim=1)[0].cpu().numpy()
            latencies.append((time.perf_counter() - t0) * 1000.0)
    avg_latency_ms = float(np.mean(latencies))
    pred_sample_idx = int(np.argmax(out_probs))

    # 10. Evidencia TRL 3 estructurada (JSON)
    evidencia_cnn = {
        "identificacion": {
            "prototipo": "Prototipo IA-Tamizaje Cervical",
            "modulo": "cnn_citologia_sipakmed",
            "version": PROTOTYPE_VERSION,
            "nivel_madurez": "TRL 3 (Prueba de concepto analitica y experimental en entorno controlado)",
            "fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(),
            "random_state": RANDOM_STATE,
            "entorno": {
                "os": platform.platform(),
                "python": sys.version.split()[0],
                "dispositivo": str(device),
                "cpu_hilos_usados": torch.get_num_threads(),
                "pytorch": torch.__version__,
            },
        },
        "datos": {
            "fuente": "SIPaKMeD (imagenes celulares recortadas CROPPED + anotaciones *_cyt.dat / *_nuc.dat)",
            "total_imagenes": int(len(df)),
            "clases": CLASSES,
            "distribucion_por_clase": {k: int(v) for k, v in class_counts.items()},
            "particion_sin_fuga": {
                "metodo": "StratifiedGroupKFold (n_splits=7) agrupado por (clase, id_lamina)",
                "laminas_compartidas_entre_splits": 0,
                "train": {
                    "imagenes": int(len(train_df)),
                    "laminas_unicas": int(train_df["slide_group"].nunique()),
                    "por_clase": {k: int(v) for k, v in train_df["class_name"].value_counts().reindex(CLASSES).items()},
                },
                "val": {
                    "imagenes": int(len(val_df)),
                    "laminas_unicas": int(val_df["slide_group"].nunique()),
                    "por_clase": {k: int(v) for k, v in val_df["class_name"].value_counts().reindex(CLASSES).items()},
                },
                "test": {
                    "imagenes": int(len(test_df)),
                    "laminas_unicas": int(test_df["slide_group"].nunique()),
                    "por_clase": {k: int(v) for k, v in test_df["class_name"].value_counts().reindex(CLASSES).items()},
                },
            },
        },
        "preprocesamiento": {
            "tamano_entrada": f"{IMG_SIZE}x{IMG_SIZE}",
            "normalizacion": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            "data_augmentation_solo_train": [
                "RandomHorizontalFlip(p=0.5)",
                "RandomVerticalFlip(p=0.5)",
                "RandomRotation(25 deg)",
                "ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10)",
            ],
            "escalado_anotaciones_dat": "Coordenadas *_cyt.dat y *_nuc.dat escaladas proporcionalmente a 224x224 (ver cnn_sample_annotations_224.png)",
            "muestras_exportadas": sample_meta,
        },
        "entrenamiento": {
            "arquitectura": "MobileNetV2 (Transfer Learning, pesos ImageNet)",
            "estrategia": f"2 etapas: {args.epochs_head} epocas solo clasificador + {args.epochs_finetune} epocas fine-tuning (features[10:] + clasificador)",
            "funcion_perdida": "CrossEntropyLoss ponderada por frecuencia de clase",
            "optimizador": "AdamW (lr_fase1=1.5e-3, lr_fase2=4e-4 + CosineAnnealingLR, weight_decay=1e-4)",
            "batch_size": args.batch_size,
            "epocas_totales": args.epochs_head + args.epochs_finetune,
            "criterio_seleccion": "Mejor Macro-F1 en conjunto de validacion",
            "mejor_epoca": best_epoch,
            "mejor_val_macro_f1": round(float(best_val_f1), 4),
            "tiempo_entrenamiento_segundos": round(float(total_train_sec), 1),
            "historial": history,
        },
        "metricas_test": {
            "test_loss": round(float(test_loss), 4),
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "macro_roc_auc_ovr": round(macro_roc_auc, 4),
            "macro_pr_auc_ovr": round(macro_pr_auc, 4),
            "por_clase": per_class_metrics,
            "matriz_confusion_5x5": cm.tolist(),
            "tamizaje_clinico_binario": clinical_screening_metrics,
        },
        "prueba_inferencia": {
            "imagen_prueba": Path(sample_row["image_path"]).name,
            "clase_real": sample_row["class_name"],
            "clase_predicha": CLASSES[pred_sample_idx],
            "probabilidades_por_clase": {cls: round(float(out_probs[i]), 4) for i, cls in enumerate(CLASSES)},
            "probabilidad_anormal_agregada": round(float(out_probs[ABNORMAL_CLASS_INDICES].sum()), 4),
            "latencia_promedio_ms_30_runs": round(avg_latency_ms, 2),
        },
        "artefactos_generados": {
            "modelo_pytorch": str(pt_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "matriz_confusion_png": str(cm_png.relative_to(BASE_DIR)).replace("\\", "/"),
            "curvas_roc_pr_png": str(roc_png.relative_to(BASE_DIR)).replace("\\", "/"),
            "historial_entrenamiento_png": str(hist_png.relative_to(BASE_DIR)).replace("\\", "/"),
            "anotaciones_dat_224_png": str(ann_png.relative_to(BASE_DIR)).replace("\\", "/"),
            "gradcam_ejemplos_png": str(gradcam_png.relative_to(BASE_DIR)).replace("\\", "/"),
        },
    }

    out_json = OUTPUTS_DIR / "evidencia_cnn_trl3.json"
    out_json.write_text(json.dumps(evidencia_cnn, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Modelo CNN guardado en: {pt_path}")
    print(f"[OK] Evidencia TRL 3 CNN guardada en: {out_json}")
    print(f"[OK] Latencia promedio inferencia CPU: {avg_latency_ms:.2f} ms")

    build_unified_trl3_evidence()


if __name__ == "__main__":
    main()
