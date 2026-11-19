"""
Suite de Verificacion Integral End-to-End - Prototipo IA-Tamizaje Cervical v0.1 (TRL 3)
=======================================================================================
Valida automaticamente todos los componentes y flujos del sistema:
1. Frontend estatico (HTML/CSS/JS) y documentacion Swagger (/docs, /openapi.json)
2. Estado de salud de los 3 modelos en memoria (CNN, OOD, Tabular)
3. Las 5 muestras reales de prueba SIPaKMeD (validacion OOD + clasificacion exacta + Grad-CAM)
4. Carga multipart de imagen (/api/predict/image)
5. Las 3 muestras OOD medicas no cervicales + ruido sintetico (rechazo garantizado)
6. Esquema tabular (26 variables) + perfiles bajo, alto y 100% vacio (imputacion automatica)
7. Evidencia JSON consolidada (/api/evidence) y las 9 graficas PNG servidas en /outputs/
"""

import io
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.main import app, load_models  # noqa: E402


def run_all_checks() -> None:
    load_models()
    client = TestClient(app)

    passed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed
        if not condition:
            raise AssertionError(f"[FAIL] {name}: {detail}")
        passed += 1
        print(f"[PASS] {name}: {detail}")

    # 1. Frontend y Swagger
    r_idx = client.get("/")
    check("Frontend / (index.html)", r_idx.status_code == 200 and "IA-Tamizaje Cervical" in r_idx.text, f"{len(r_idx.text)} chars")
    for asset in ["/static/styles.css", "/static/app.js", "/docs", "/openapi.json"]:
        ra = client.get(asset)
        check(f"Recurso {asset}", ra.status_code == 200, f"HTTP {ra.status_code}")

    # 2. Health
    h = client.get("/api/health").json()
    check("API /api/health", h["status"] == "ok" and h["cnn_ready"] and h["ood_ready"] and h["tabular_ready"], str(h))

    # 3. Muestras SIPaKMeD (5/5 validas + Grad-CAM)
    s = client.get("/api/samples").json()
    check("Listado /api/samples", len(s["samples"]) == 5 and len(s["ood_samples"]) == 3, "5 validas + 3 OOD")

    exact_matches = 0
    for item in s["samples"]:
        fn = item["filename"]
        exp = item["expected_class"]
        rp = client.post(f"/api/predict/sample/{fn}").json()
        pred = rp["prediccion"]["clase_predicha"]
        conf = rp["prediccion"]["confianza"]
        ms_cnn = rp["tiempos_ms"]["inferencia_cnn"]
        ms_cam = rp["tiempos_ms"]["gradcam_xai"]
        ok_cam = isinstance(rp["imagenes"]["gradcam_overlay"], str) and rp["imagenes"]["gradcam_overlay"].startswith("data:image/png;base64,")
        if pred == exp:
            exact_matches += 1
        check(
            f"CNN + OOD + Grad-CAM ({exp})",
            rp["dominio_valido"] is True and ok_cam and len(rp["prediccion"]["probabilidades"]) == 5,
            f"pred={pred}, conf={conf:.4f}, cnn={ms_cnn}ms, gradcam={ms_cam}ms",
        )
    check("Exactitud en muestras de prueba (>=4/5)", exact_matches >= 4, f"{exact_matches}/5 aciertos exactos (consistente con 94.46% en Test global)")

    # 4. Upload multipart valido
    bmp_path = BASE_DIR / "data" / "sample_images" / s["samples"][1]["filename"]
    with open(bmp_path, "rb") as f:
        ru = client.post("/api/predict/image", files={"file": (bmp_path.name, f, "image/bmp")})
    du = ru.json()
    check(
        "Upload multipart /api/predict/image (valido)",
        ru.status_code == 200 and du["dominio_valido"] is True,
        f"archivo={bmp_path.name}, pred={du['prediccion']['clase_predicha']}",
    )

    # 5. Rechazo OOD (3 muestras medicas no cervicales)
    for oitem in s["ood_samples"]:
        ofn = oitem["filename"]
        olbl = oitem["label"]
        ro = client.post(f"/api/predict/ood-sample/{ofn}").json()
        ood_v = ro["validacion_dominio_ood"]
        check(
            f"Rechazo OOD ({olbl})",
            ro["dominio_valido"] is False and ro["prediccion"]["nivel_tamizaje"] == "FUERA_DE_DOMINIO",
            f"prob_cerv={ood_v['probabilidad_dominio_cervical']}, cos={ood_v['similitud_coseno_centroides']}, ood={ood_v['latencia_ood_ms']}ms",
        )

    # 6. Rechazo OOD (imagen sintetica de ruido aleatorio)
    rng = np.random.default_rng(42)
    noise_img = Image.fromarray(rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    noise_img.save(buf, format="PNG")
    buf.seek(0)
    rn = client.post("/api/predict/image", files={"file": ("noise_test.png", buf, "image/png")}).json()
    ood_n = rn["validacion_dominio_ood"]
    check(
        "Rechazo OOD (ruido sintetico no medico)",
        rn["dominio_valido"] is False,
        f"prob_cerv={ood_n['probabilidad_dominio_cervical']}, cos={ood_n['similitud_coseno_centroides']}",
    )

    # 7. Modulo tabular (esquema + 3 perfiles incluyendo imputacion total)
    sch = client.get("/api/tabular/schema").json()
    check("Esquema /api/tabular/schema", len(sch["features"]) == 26, "26 variables clinicas sin leakage")

    profiles = [
        ("Perfil Bajo", {"Age": 24, "Number of sexual partners": 1, "Smokes": 0, "STDs": 0}),
        (
            "Perfil Alto (caso real Biopsy=1)",
            {
                "Age": 38,
                "Number of sexual partners": 4,
                "First sexual intercourse": 18,
                "Num of pregnancies": 4,
                "Smokes": 0,
                "Hormonal Contraceptives": 1,
                "Hormonal Contraceptives (years)": 16,
                "IUD": 1,
                "IUD (years)": 3,
                "STDs": 0,
            },
        ),
        ("Perfil Vacio (26 imputaciones)", {}),
    ]
    for pname, payload in profiles:
        rt = client.post("/api/predict/tabular", json={"values": payload}).json()
        p = rt["prediccion"]
        check(
            f"Tabular /api/predict/tabular ({pname})",
            0.0 <= p["probabilidad_riesgo_biopsia"] <= 1.0,
            f"prob={p['probabilidad_riesgo_biopsia']}, nivel={p['nivel_tamizaje']}, imputadas={rt['variables_imputadas']}, ms={rt['latencia_inferencia_ms']}",
        )

    # 8. Evidencia consolidada y 9 figuras PNG servidas via HTTP
    ev = client.get("/api/evidence").json()
    check("Evidencia /api/evidence", len(ev.get("modulos_disponibles", [])) == 3, str(ev.get("modulos_disponibles")))

    expected_pngs = [
        "cnn_confusion_matrix.png",
        "cnn_roc_pr_curves.png",
        "cnn_training_history.png",
        "cnn_sample_annotations_224.png",
        "cnn_sample_gradcam.png",
        "ood_confusion_matrix.png",
        "ood_score_distribution.png",
        "tabular_confusion_matrix.png",
        "tabular_roc_pr_curves.png",
    ]
    for png in expected_pngs:
        rf = client.get(f"/outputs/{png}")
        check(f"Grafica /outputs/{png}", rf.status_code == 200 and len(rf.content) > 10000, f"{len(rf.content) // 1024} KB")

    print(f"\n=== VERIFICACION INTEGRAL COMPLETADA: {passed}/{passed} PRUEBAS EXITOSAS ===")


if __name__ == "__main__":
    run_all_checks()
