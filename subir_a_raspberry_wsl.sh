#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

RPI_USER="${RPI_USER:-rp5}"
RPI_HOST="${RPI_HOST:-192.168.50.7}"
RPI_DIR="${RPI_DIR:-~/sc2-hailo8-transformer}"

echo "=========================================================="
echo "  Transfiriendo SC2-Hailo8 V2 a Raspberry Pi 5 (desde WSL)"
echo "  Destino: ${RPI_USER}@${RPI_HOST}:${RPI_DIR}/"
echo "=========================================================="

ssh "${RPI_USER}@${RPI_HOST}" "mkdir -p ${RPI_DIR}/models ${RPI_DIR}/src"

if [ -f "models/sc2_v2_agent.hef" ]; then
    scp "models/sc2_v2_agent.hef" "${RPI_USER}@${RPI_HOST}:${RPI_DIR}/models/sc2_v2_agent.hef"
else
    echo "[Aviso] models/sc2_v2_agent.hef no encontrado aún; omitiendo .hef."
fi

if [ -f "models/sc2_v2_actor.onnx" ]; then
    scp "models/sc2_v2_actor.onnx" "${RPI_USER}@${RPI_HOST}:${RPI_DIR}/models/sc2_v2_actor.onnx"
fi

scp "src/config.py" "${RPI_USER}@${RPI_HOST}:${RPI_DIR}/src/config.py"
scp "src/rpi_hailo_server.py" "${RPI_USER}@${RPI_HOST}:${RPI_DIR}/src/rpi_hailo_server.py"
scp "src/__init__.py" "${RPI_USER}@${RPI_HOST}:${RPI_DIR}/src/__init__.py"

echo "=========================================================="
echo "✅ Transferencia V2 finalizada."
echo "   En la Raspberry Pi inicia el servidor con:"
echo "   cd ${RPI_DIR} && python3 -m src.rpi_hailo_server --port 5555"
echo "=========================================================="
