#!/bin/bash
# Setup script para ButterVision con entorno virtual
# Ejecutar: bash setup.sh

echo "🚀 Configurando ButterVision con entorno virtual..."

# Verificar que estamos en el directorio correcto
if [ ! -f "requirements.txt" ] || [ ! -f "main.py" ]; then
    echo "❌ Error: Ejecutar desde el directorio de ButterVision"
    exit 1
fi

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual con Python 3.10..."
    python3.10 -m venv venv
else
    echo "✅ Entorno virtual ya existe"
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Actualizar pip
echo "⬆️  Actualizando pip..."
pip install --upgrade pip

# Instalar PyTorch con CUDA
echo "🔥 Instalando PyTorch con CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instalar resto de dependencias
echo "📚 Instalando dependencias del proyecto..."
pip install -r requirements.txt

# Verificar instalación
echo "✅ Verificando instalación..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || { echo "❌ Error con PyTorch"; exit 1; }
python -c "import gradio; print(f'Gradio: {gradio.__version__}')" || { echo "❌ Error con Gradio"; exit 1; }
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')" || { echo "❌ Error con Diffusers"; exit 1; }

# Actualizar script run.sh
echo '#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "No existe el entorno virtual venv."
    echo "Primero ejecuta: bash install.sh cuda121"
    exit 1
fi

source venv/bin/activate
python main.py "$@"' > run.sh
chmod +x run.sh

echo ""
echo "🎉 ¡Configuración completada!"
echo ""
echo "Para ejecutar ButterVision:"
echo "  ./run.sh"
echo ""
echo "Puedes pasar opciones al lanzador:"
echo "  ./run.sh --port 7861"
