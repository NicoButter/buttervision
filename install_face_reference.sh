#!/bin/bash
# Instala dependencias opcionales para Face Reference / InstantID.

set -e

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "No existe el entorno virtual venv."
    echo "Primero ejecuta: bash install.sh cuda121"
    exit 1
fi

source venv/bin/activate

VENV_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
for nvidia_lib_dir in "$VENV_SITE"/nvidia/*/lib; do
    if [ -d "$nvidia_lib_dir" ]; then
        export LD_LIBRARY_PATH="$nvidia_lib_dir:${LD_LIBRARY_PATH:-}"
    fi
done

echo "📦 Instalando dependencias opcionales de Face Reference..."
pip install -r requirements-instantid.txt
pip install onnxruntime==1.23.2
pip install --force-reinstall --no-deps onnxruntime-gpu==1.23.2

echo ""
echo "🔍 Verificando dependencias de InstantID..."
python -c "import cv2; import numpy; import insightface; import onnxruntime as ort; print('Face Reference runtime OK'); print('ONNXRuntime providers:', ort.get_available_providers())"

echo ""
echo "✅ Face Reference listo."
echo "Inicia ButterVision con: ./run.sh"
