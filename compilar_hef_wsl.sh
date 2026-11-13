#!/usr/bin/env bash
set -e

# Directorio base del repositorio en WSL
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=========================================================="
echo "  Hailo Dataflow Compiler (DFC) - Compilador V2 en WSL    "
echo "  Modelo: SC2HailoV2Actor -> models/sc2_v2_agent.hef      "
echo "=========================================================="

# 1. Activar entorno virtual si existe
if [ -f "$HOME/hailo_dfc_venv/bin/activate" ]; then
    echo "[WSL] Activando entorno virtual: $HOME/hailo_dfc_venv"
    source "$HOME/hailo_dfc_venv/bin/activate"
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "[WSL] Usando entorno virtual activo: $VIRTUAL_ENV"
else
    echo "[AVISO] No se detectó un entorno virtual activo. Usando python3 del sistema."
fi

# 2. Verificar que el paquete hailo_sdk_client esté instalado
python3 -c "import hailo_sdk_client" 2>/dev/null || {
    echo "❌ Error: hailo_sdk_client no está instalado en este entorno."
    echo "Instala el wheel de Hailo DFC con:"
    echo "  pip install hailo_dataflow_compiler-*.whl"
    exit 1
}

# 3. Ejecutar la compilación
echo "[WSL] Ejecutando compilación ONNX -> HAR -> HEF..."
python3 src/compile_hef.py "$@"

echo "=========================================================="
echo "✅ Compilación V2 completada con éxito."
echo "   Archivo generado: models/sc2_v2_agent.hef"
echo "   Sube a la RPi5 con: ./subir_a_raspberry_wsl.sh (o .\\subir_a_raspberry.bat)"
echo "=========================================================="
