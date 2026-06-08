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

echo "📦 Instalando dependencias opcionales de Face Reference..."
pip install -r requirements-instantid.txt

echo ""
echo "🔍 Verificando dependencias de InstantID..."
python -c "import cv2; import numpy; import insightface; import onnxruntime; print('Face Reference runtime OK')"

echo ""
echo "✅ Face Reference listo."
echo "Inicia ButterVision con: ./run.sh"
