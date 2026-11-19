# Guía de Despliegue Gratuito en Hugging Face Spaces (SDK Docker)

El proyecto ya cuenta con **todo lo necesario** para publicarse gratis (\$0/mes) con **2 vCPU y 16 GB de RAM** en **Hugging Face Spaces** con acceso HTTPS público desde cualquier navegador o celular:

* `README.md` ya incluye la cabecera YAML de configuración (`sdk: docker`, `app_port: 7860`).
* `Dockerfile` está optimizado para Hugging Face Spaces (usuario no-root UID `1000`, `opencv-python-headless`, PyTorch CPU y puerto `7860`).
* `.dockerignore` excluye automáticamente los datasets pesados (`archive_base1/`, `archivebase2/`, `*.zip`).
* El endpoint `/api/predict/image` procesa imágenes **solo en memoria** y limita el tamaño máximo a **10 MB**.

---

## Paso 1: Crear el Space en Hugging Face (2 minutos)

1. Inicia sesión (o crea una cuenta gratuita) en **https://huggingface.co/spaces**.
2. Haz clic en **+ New Space**.
3. Configura:
   * **Space name:** `ia-tamizaje-cervical-trl3` (o el nombre que prefieras)
   * **License:** `MIT` (o la licencia de tu proyecto)
   * **Select the Space SDK:** elige **Docker** → plantilla **Blank**
   * **Space hardware:** **CPU basic · 2 vCPU · 16 GB · FREE**
   * **Visibility:** **Public** (para que cualquiera pueda abrir la URL)
4. Haz clic en **Create Space**.

---

## Paso 2: Crear un Access Token de escritura (1 minuto)

Hugging Face usa un token en lugar de tu contraseña para `git push`:
1. Ve a **https://huggingface.co/settings/tokens**.
2. Haz clic en **+ Create new token** → tipo **Write** → ponle un nombre (ej. `deploy-cervical`) y copia el token (`hf_...`).

---

## Paso 3: Subir el repositorio al Space desde PowerShell (2 minutos)

Abre PowerShell en `C:\ML_img_can` y ejecuta (reemplazando `TU_USUARIO_HF` por tu usuario real de Hugging Face):

```powershell
cd C:\ML_img_can

# 1. Agregar el Space como remoto adicional llamado 'hf'
git remote add hf https://huggingface.co/spaces/TU_USUARIO_HF/ia-tamizaje-cervical-trl3

# 2. Subir la rama main al Space (cuando pida credenciales: usuario=TU_USUARIO_HF, clave=tu token hf_...)
git push --force hf main
```

> **Nota si Hugging Face solicita Git LFS para archivos `.pt` / `.joblib`:**
> Todos los archivos pesan menos de 10 MB (`cnn_sipakmed_trl3_v01.pt` pesa `8.73 MB`), pero si Hugging Face solicita rastrear binarios con LFS, ejecuta:
> ```powershell
> git lfs install
> git lfs track "*.pt" "*.joblib" "*.docx"
> git add .gitattributes
> git commit -m "chore: habilitar Git LFS para modelos en Hugging Face Spaces"
> git push --force hf main
> ```

---

## Paso 4: Verificar la aplicación publicada

1. Abre la página de tu Space:
   `https://huggingface.co/spaces/TU_USUARIO_HF/ia-tamizaje-cervical-trl3`
2. Verás la pestaña **Logs** construyendo la imagen Docker (toma ~2–4 minutos la primera vez).
3. Cuando el estado cambie a **Running**, tu aplicación web quedará disponible para cualquier persona en:
   * **Enlace embebido:** `https://huggingface.co/spaces/TU_USUARIO_HF/ia-tamizaje-cervical-trl3`
   * **Enlace directo pantalla completa:** `https://TU_USUARIO_HF-ia-tamizaje-cervical-trl3.hf.space`
   * **Swagger API:** `https://TU_USUARIO_HF-ia-tamizaje-cervical-trl3.hf.space/docs`

---

## (Opcional) Sincronización automática desde GitHub

Si más adelante quieres que cada `git push origin main` en GitHub actualice automáticamente tu Hugging Face Space:
1. En tu repositorio de GitHub (`Settings → Secrets and variables → Actions`), crea un secreto llamado `HF_TOKEN` con tu token `hf_...`.
2. Agrega un workflow en `.github/workflows/sync_hf.yml` que ejecute `git push --force https://TU_USUARIO_HF:${{ secrets.HF_TOKEN }}@huggingface.co/spaces/TU_USUARIO_HF/ia-tamizaje-cervical-trl3 main`.
