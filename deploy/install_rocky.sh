#!/usr/bin/env bash
# ==============================================================================
# Instalador Nativo (sin Docker) para Rocky Linux 8 / 9 / 10
# Prototipo IA-Tamizaje Cervical v0.1 (TRL 3)
# ==============================================================================
set -euo pipefail

APP_DIR="/opt/ia-cervical"
SERVICE_NAME="ia-cervical"
PORT="${PORT:-8000}"
RUN_USER="${SUDO_USER:-$USER}"

echo "=== [1/6] Detectando Python >= 3.10 e instalando paquetes base ==="
sudo dnf install -y git glib2 python3 python3-pip

PYTHON_BIN=""
for candidate in python3 python3.12 python3.11; do
    if command -v "${candidate}" >/dev/null 2>&1; then
        if "${candidate}" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"; then
            PYTHON_BIN="${candidate}"
            break
        fi
    fi
done

if [ -z "${PYTHON_BIN}" ]; then
    echo "[INFO] python3 del sistema es < 3.10; instalando python3.11..."
    sudo dnf install -y python3.11 python3.11-pip
    PYTHON_BIN="python3.11"
fi

echo "[OK] Usando interprete: ${PYTHON_BIN} ($(${PYTHON_BIN} --version))"

echo "=== [2/6] Preparando entorno virtual en ${APP_DIR}/.venv ==="
cd "${APP_DIR}"
"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "=== [3/6] Instalando PyTorch CPU y dependencias Linux ==="
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -r prototipo_trl3/requirements-linux.txt

echo "=== [4/6] Ejecutando verificacion integral E2E (32 pruebas) ==="
python prototipo_trl3/test_e2e.py

echo "=== [5/6] Configurando servicio systemd (${SERVICE_NAME}.service) ==="
sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" > /dev/null <<EOF
[Unit]
Description=Prototipo IA-Tamizaje Cervical v0.1 (FastAPI)
After=network.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${APP_DIR}/prototipo_trl3
Environment="MPLCONFIGDIR=/tmp/matplotlib"
ExecStart=${APP_DIR}/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"

echo "=== [6/6] Estado del servicio ==="
sudo systemctl status "${SERVICE_NAME}" --no-pager -l || true

echo ""
echo "========================================================================"
echo " DESPLIEGUE COMPLETADO EN PUERTO ${PORT}"
echo "------------------------------------------------------------------------"
echo " Si accedes directo por IP:${PORT}, abre el firewall con:"
echo "   sudo firewall-cmd --permanent --add-port=${PORT}/tcp"
echo "   sudo firewall-cmd --reload"
echo ""
echo " Ver logs en vivo:"
echo "   sudo journalctl -u ${SERVICE_NAME} -f"
echo "========================================================================"
