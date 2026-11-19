# Dockerfile optimizado para Hugging Face Spaces (SDK Docker, UID 1000, puerto 7860)
# Tambien compatible con Google Cloud Run, Railway y despliegue Docker local.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    MPLCONFIGDIR=/tmp/matplotlib

# Dependencias minimas de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root UID 1000 (requisito de Hugging Face Spaces)
RUN useradd -m -u 1000 user

WORKDIR /app

# Instalar PyTorch CPU ligero + dependencias (opencv-python-headless para Linux cloud)
COPY prototipo_trl3/requirements.txt /app/requirements.txt
RUN sed 's/^opencv-python/opencv-python-headless/' /app/requirements.txt > /app/requirements_cloud.txt \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/requirements_cloud.txt

# Copiar solo el prototipo ligero (~13 MB) con permisos para UID 1000
COPY --chown=user:user prototipo_trl3 /app/prototipo_trl3

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app/prototipo_trl3

EXPOSE 7860

# Puerto 7860 por defecto para Hugging Face Spaces (o $PORT en Cloud Run / Railway)
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
