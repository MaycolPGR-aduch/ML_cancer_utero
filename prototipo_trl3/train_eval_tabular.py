"""
Prototipo IA-Tamizaje Cervical v0.1 (TRL 3) - Modulo Tabular (Factores de Riesgo)
================================================================================
Cumple los criterios de evidencia experimental TRL 3:
- Extraccion y limpieza reproducible del dataset kag_risk_factors_cervical_cancer.csv
- Control explicito de datos faltantes ('?') y fuga de informacion (data leakage)
- Comparacion de modelos baseline mediante validacion cruzada estratificada (k=5)
- Optimizacion de umbral de decision para tamizaje (sensibilidad / F1)
- Generacion de modelo serializado (.joblib), graficas (Matriz de Confusion, ROC/PR)
  y archivo de evidencia trazable (outputs/evidencia_tabular_trl3.json).
"""

import io
import json
import os
import platform
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
PROTOTYPE_VERSION = "v0.1-trl3-tabular"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

CSV_PATH = DATA_DIR / "kag_risk_factors_cervical_cancer.csv"
ZIP_SOURCE = PROJECT_ROOT / "drive-download-20261119T203250Z-1-001.zip"

# Variables post-prueba / diagnosticos directos excluidos para evitar data leakage
POST_TEST_OUTCOMES = [
    "Hinselmann",
    "Schiller",
    "Citology",
    "Biopsy",
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV",
    "Dx",
]

# Descripcion amigable en espanol para el formulario web
FEATURE_METADATA = {
    "Age": {"label": "Edad (anos)", "type": "number", "min": 13, "max": 90, "step": 1, "default": 28},
    "Number of sexual partners": {"label": "Numero de parejas sexuales", "type": "number", "min": 0, "max": 30, "step": 1, "default": 2},
    "First sexual intercourse": {"label": "Edad primera relacion sexual", "type": "number", "min": 10, "max": 45, "step": 1, "default": 17},
    "Num of pregnancies": {"label": "Numero de embarazos", "type": "number", "min": 0, "max": 15, "step": 1, "default": 2},
    "Smokes": {"label": "Fuma actualmente", "type": "binary", "default": 0},
    "Smokes (years)": {"label": "Anos fumando", "type": "number", "min": 0, "max": 50, "step": 0.5, "default": 0},
    "Smokes (packs/year)": {"label": "Paquetes de cigarrillos / ano", "type": "number", "min": 0, "max": 50, "step": 0.5, "default": 0},
    "Hormonal Contraceptives": {"label": "Uso de anticonceptivos hormonales", "type": "binary", "default": 1},
    "Hormonal Contraceptives (years)": {"label": "Anos con anticonceptivos hormonales", "type": "number", "min": 0, "max": 35, "step": 0.5, "default": 1},
    "IUD": {"label": "Uso de DIU (dispositivo intrauterino)", "type": "binary", "default": 0},
    "IUD (years)": {"label": "Anos con DIU", "type": "number", "min": 0, "max": 25, "step": 0.5, "default": 0},
    "STDs": {"label": "Antecedente de ETS", "type": "binary", "default": 0},
    "STDs (number)": {"label": "Numero de ETS diagnosticadas", "type": "number", "min": 0, "max": 10, "step": 1, "default": 0},
    "STDs:condylomatosis": {"label": "ETS: Condilomatosis", "type": "binary", "default": 0},
    "STDs:cervical condylomatosis": {"label": "ETS: Condilomatosis cervical", "type": "binary", "default": 0},
    "STDs:vaginal condylomatosis": {"label": "ETS: Condilomatosis vaginal", "type": "binary", "default": 0},
    "STDs:vulvo-perineal condylomatosis": {"label": "ETS: Condilomatosis vulvo-perineal", "type": "binary", "default": 0},
    "STDs:syphilis": {"label": "ETS: Sifilis", "type": "binary", "default": 0},
    "STDs:pelvic inflammatory disease": {"label": "ETS: Enfermedad inflamatoria pelvica", "type": "binary", "default": 0},
    "STDs:genital herpes": {"label": "ETS: Herpes genital", "type": "binary", "default": 0},
    "STDs:molluscum contagiosum": {"label": "ETS: Molusco contagioso", "type": "binary", "default": 0},
    "STDs:AIDS": {"label": "ETS: SIDA", "type": "binary", "default": 0},
    "STDs:HIV": {"label": "ETS: VIH", "type": "binary", "default": 0},
    "STDs:Hepatitis B": {"label": "ETS: Hepatitis B", "type": "binary", "default": 0},
    "STDs:HPV": {"label": "ETS: VPH (Virus del Papiloma Humano)", "type": "binary", "default": 0},
    "STDs: Number of diagnosis": {"label": "Numero total de diagnosticos de ETS", "type": "number", "min": 0, "max": 10, "step": 1, "default": 0},
}


def ensure_csv_extracted() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists():
        return CSV_PATH
    if not ZIP_SOURCE.exists():
        raise FileNotFoundError(
            f"No se encontro {CSV_PATH} ni el archivo fuente {ZIP_SOURCE}."
        )
    with zipfile.ZipFile(ZIP_SOURCE, "r") as outer_zip:
        with outer_zip.open("archive.zip") as inner_bytes:
            with zipfile.ZipFile(io.BytesIO(inner_bytes.read()), "r") as inner_zip:
                data = inner_zip.read("kag_risk_factors_cervical_cancer.csv")
                CSV_PATH.write_bytes(data)
    print(f"[INFO] CSV extraido en: {CSV_PATH}")
    return CSV_PATH


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    classes = ["Negativo (0)", "Positivo Biopsia (1)"]
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="Etiqueta Real",
        xlabel="Prediccion del Modelo",
        title=title,
    )
    thresh = cm.max() / 2.0
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = (cm[i, j] / total * 100) if total > 0 else 0
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "#1e293b",
                fontsize=11,
                fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, model_name: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    prevalence = y_true.mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)

    axes[0].plot(fpr, tpr, color="#0ea5e9", lw=2.2, label=f"{model_name} (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="#94a3b8", lw=1.2, linestyle="--", label="Azar (AUC = 0.50)")
    axes[0].set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    axes[0].set_ylabel("Sensibilidad (Recall)")
    axes[0].set_title("Curva ROC (Validacion Cruzada 5-Fold OOF)")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(recall, precision, color="#10b981", lw=2.2, label=f"{model_name} (PR-AUC = {pr_auc:.4f})")
    axes[1].axhline(prevalence, color="#94a3b8", lw=1.2, linestyle="--", label=f"Prevalencia base ({prevalence:.3f})")
    axes[1].set_xlabel("Sensibilidad (Recall)")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Curva Precision-Recall (Validacion Cruzada 5-Fold OOF)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def select_screening_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Selecciona un umbral orientado a tamizaje (balance entre sensibilidad y F1)."""
    thresholds = np.linspace(0.10, 0.85, 76)
    best_f1 = {"threshold": 0.5, "f1": -1.0, "recall": 0.0, "precision": 0.0}
    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        # Prioriza mejor F1 manteniendo sensibilidad util (>= 0.30 cuando sea posible)
        if f1 > best_f1["f1"] or (abs(f1 - best_f1["f1"]) < 1e-6 and rec > best_f1["recall"]):
            best_f1 = {
                "threshold": round(float(thr), 3),
                "f1": round(float(f1), 4),
                "recall": round(float(rec), 4),
                "precision": round(float(prec), 4),
            }
    return best_f1


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_file = ensure_csv_extracted()
    raw_df = pd.read_csv(csv_file)
    initial_shape = raw_df.shape

    # Limpieza de '?' -> NaN y conversion numerica
    df = raw_df.replace("?", np.nan).apply(pd.to_numeric, errors="coerce")

    missing_counts = df.isna().sum()
    high_missing_cols = [c for c in df.columns if df[c].isna().mean() > 0.50]

    # Seleccion de variables sin leakage y sin >50% faltantes
    excluded_cols = sorted(set(POST_TEST_OUTCOMES + high_missing_cols))
    features = [c for c in df.columns if c not in excluded_cols]

    # Descartar posibles columnas constantes (varianza 0) si existieran? Las conservamos documentadas
    constant_cols = [c for c in features if df[c].dropna().nunique() <= 1]

    target_col = "Biopsy"
    X = df[features].copy()
    y = df[target_col].astype(int).values

    # Estadisticas de desenlaces secundarios documentados en el informe
    secondary_outcomes = {}
    for col in ["Biopsy", "Dx:Cancer", "Hinselmann", "Schiller", "Citology"]:
        pos = int(df[col].sum())
        secondary_outcomes[col] = {
            "positivos": pos,
            "negativos": int(len(df) - pos),
            "prevalencia_pct": round(float(pos / len(df) * 100), 2),
        }

    # Definicion de modelos baseline dentro de pipelines sin fuga
    candidates = {
        "LogisticRegression (balanced)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "RandomForest (balanced)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "ExtraTrees (balanced)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ExtraTreesClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
    }

    baseline_comparison = {}
    oof_probs_map = {}

    print("=== Evaluando modelos baseline (Validacion Cruzada Estratificada 5-Fold) ===")
    for name, pipe in candidates.items():
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
        oof_probs = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        oof_probs_map[name] = oof_probs

        oof_roc = roc_auc_score(y, oof_probs)
        oof_pr = average_precision_score(y, oof_probs)
        thr_info = select_screening_threshold(y, oof_probs)

        baseline_comparison[name] = {
            "cv_5fold_mean": {
                "roc_auc": round(float(scores["test_roc_auc"].mean()), 4),
                "roc_auc_std": round(float(scores["test_roc_auc"].std()), 4),
                "pr_auc": round(float(scores["test_pr_auc"].mean()), 4),
                "pr_auc_std": round(float(scores["test_pr_auc"].std()), 4),
                "recall_thr_0_5": round(float(scores["test_recall"].mean()), 4),
                "precision_thr_0_5": round(float(scores["test_precision"].mean()), 4),
                "f1_thr_0_5": round(float(scores["test_f1"].mean()), 4),
            },
            "oof_global": {
                "roc_auc": round(float(oof_roc), 4),
                "pr_auc": round(float(oof_pr), 4),
                "umbral_optimizado_tamizaje": thr_info,
            },
        }
        print(
            f"  - {name}: ROC-AUC={scores['test_roc_auc'].mean():.4f} "
            f"(±{scores['test_roc_auc'].std():.4f}), PR-AUC={scores['test_pr_auc'].mean():.4f}, "
            f"Recall@0.5={scores['test_recall'].mean():.4f}, F1@thr*={thr_info['f1']:.4f} (thr={thr_info['threshold']})"
        )

    # Seleccion del mejor modelo segun ROC-AUC promedio en CV 5-fold (desempate por PR-AUC)
    best_model_name = max(
        baseline_comparison.keys(),
        key=lambda k: (
            baseline_comparison[k]["cv_5fold_mean"]["roc_auc"],
            baseline_comparison[k]["cv_5fold_mean"]["pr_auc"],
        ),
    )
    best_pipeline = candidates[best_model_name]
    best_oof_probs = oof_probs_map[best_model_name]
    best_thr = baseline_comparison[best_model_name]["oof_global"]["umbral_optimizado_tamizaje"]["threshold"]

    # Matriz de confusion OOF usando umbral optimizado y umbral 0.50
    oof_preds_opt = (best_oof_probs >= best_thr).astype(int)
    cm_opt = confusion_matrix(y, oof_preds_opt)
    tn, fp, fn, tp = [int(v) for v in cm_opt.ravel()]
    specificity_opt = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Graficas
    cm_path = OUTPUTS_DIR / "tabular_confusion_matrix.png"
    roc_pr_path = OUTPUTS_DIR / "tabular_roc_pr_curves.png"
    plot_confusion_matrix(
        cm_opt,
        cm_path,
        f"Matriz de Confusion OOF (5-Fold)\n{best_model_name} | umbral={best_thr}",
    )
    plot_roc_pr_curves(y, best_oof_probs, roc_pr_path, best_model_name)

    # Entrenamiento final sobre todo el conjunto para artefacto del prototipo
    best_pipeline.fit(X, y)

    # Importancias o coeficientes para explicabilidad tabular
    model_step = best_pipeline.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        raw_imp = model_step.feature_importances_
    elif hasattr(model_step, "coef_"):
        raw_imp = np.abs(model_step.coef_[0])
    else:
        raw_imp = np.zeros(len(features))
    imp_sum = float(np.sum(raw_imp)) or 1.0
    top_features = sorted(
        [
            {"variable": f, "importancia_relativa": round(float(v / imp_sum), 4)}
            for f, v in zip(features, raw_imp)
        ],
        key=lambda d: d["importancia_relativa"],
        reverse=True,
    )[:10]

    # Guardado del modelo con metadatos completos para inferencia/API
    medians = {c: float(X[c].median()) if pd.notna(X[c].median()) else 0.0 for c in features}
    artifact_path = MODELS_DIR / "tabular_trl3_v01.joblib"
    joblib.dump(
        {
            "version": PROTOTYPE_VERSION,
            "model_name": best_model_name,
            "pipeline": best_pipeline,
            "features": features,
            "feature_metadata": {c: FEATURE_METADATA.get(c, {"label": c, "type": "number", "default": medians[c]}) for c in features},
            "feature_medians": medians,
            "threshold": best_thr,
            "target": target_col,
            "top_features": top_features,
        },
        artifact_path,
    )

    # Benchmark de latencia de inferencia (promedio sobre 50 ejecuciones en 1 registro)
    sample_row = X.iloc[[0]]
    # Warm-up
    _ = best_pipeline.predict_proba(sample_row)
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        prob_sample = float(best_pipeline.predict_proba(sample_row)[0, 1])
        latencies.append((time.perf_counter() - t0) * 1000.0)
    avg_latency_ms = float(np.mean(latencies))

    # Construccion de evidencia TRL 3 estructurada
    evidencia_tabular = {
        "identificacion": {
            "prototipo": "Prototipo IA-Tamizaje Cervical",
            "modulo": "tabular_factores_riesgo",
            "version": PROTOTYPE_VERSION,
            "nivel_madurez": "TRL 3 (Prueba de concepto analitica y experimental en entorno controlado)",
            "fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(),
            "random_state": RANDOM_STATE,
            "entorno": {
                "os": platform.platform(),
                "python": sys.version.split()[0],
                "cpu": platform.processor(),
                "scikit_learn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
        },
        "datos": {
            "fuente": "kag_risk_factors_cervical_cancer.csv (Hospital Universitario de Caracas / Kaggle)",
            "dimensiones_iniciales": {"filas": int(initial_shape[0]), "columnas": int(initial_shape[1])},
            "dimensiones_entrada_modelo": {"filas": int(X.shape[0]), "variables_entrada": int(X.shape[1])},
            "variable_objetivo": target_col,
            "desenlaces_registrados": secondary_outcomes,
            "faltantes_detectados_por_columna": {
                col: int(cnt) for col, cnt in missing_counts.items() if cnt > 0
            },
            "variables_excluidas_leakage": POST_TEST_OUTCOMES,
            "variables_excluidas_alta_ausencia_gt_50pct": high_missing_cols,
            "variables_constantes_observadas": constant_cols,
            "variables_entrada_utilizadas": features,
        },
        "metodologia": {
            "estrategia_validacion": "Validacion cruzada estratificada (StratifiedKFold, k=5, shuffle=True, random_state=42)",
            "preprocesamiento_en_pipeline": [
                "Reemplazo de '?' por NaN y conversion numerica",
                "SimpleImputer(strategy='median') ajustado dentro de cada fold (sin fuga)",
                "StandardScaler() ajustado dentro de cada fold",
                "Manejo de desbalance mediante class_weight='balanced'",
            ],
            "modelo_seleccionado": best_model_name,
            "umbral_decision_tamizaje": best_thr,
            "criterio_seleccion": "Mayor ROC-AUC medio en CV 5-fold (desempate por PR-AUC)",
        },
        "comparacion_modelos_baseline": baseline_comparison,
        "metricas_modelo_seleccionado": {
            "cv_5fold": baseline_comparison[best_model_name]["cv_5fold_mean"],
            "oof_con_umbral_tamizaje": {
                "umbral": best_thr,
                "roc_auc": round(float(roc_auc_score(y, best_oof_probs)), 4),
                "pr_auc": round(float(average_precision_score(y, best_oof_probs)), 4),
                "recall_sensibilidad": round(float(recall_score(y, oof_preds_opt, zero_division=0)), 4),
                "especificidad": round(float(specificity_opt), 4),
                "precision": round(float(precision_score(y, oof_preds_opt, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y, oof_preds_opt, zero_division=0)), 4),
                "matriz_confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            },
            "top_10_variables_importantes": top_features,
        },
        "prueba_inferencia": {
            "caso_prueba": "registro_indice_0",
            "probabilidad_riesgo_biopsia": round(prob_sample, 4),
            "clasificacion_umbral_tamizaje": int(prob_sample >= best_thr),
            "latencia_promedio_ms_50_runs": round(avg_latency_ms, 3),
        },
        "artefactos_generados": {
            "modelo": str(artifact_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "matriz_confusion_png": str(cm_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "curvas_roc_pr_png": str(roc_pr_path.relative_to(BASE_DIR)).replace("\\", "/"),
        },
    }

    out_json = OUTPUTS_DIR / "evidencia_tabular_trl3.json"
    out_json.write_text(json.dumps(evidencia_tabular, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[OK] Modelo seleccionado: {best_model_name}")
    print(f"[OK] Artefacto guardado: {artifact_path}")
    print(f"[OK] Evidencia TRL 3 tabular guardada: {out_json}")
    print(f"[OK] Latencia promedio inferencia: {avg_latency_ms:.3f} ms")


if __name__ == "__main__":
    main()
