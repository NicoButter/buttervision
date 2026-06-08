#!/bin/bash
# Script de instalación rápida para ButterVision
# Uso: bash install.sh [cuda118|cuda121|cpu] [--instantid]

set -e

CUDA_VERSION=${1:-cuda118}
INSTALL_INSTANTID=${2:-}

echo "🎨 ButterVision - Script de instalación"
echo "========================================"
echo ""

# Detectar sistema operativo
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
else
    OS="Windows/Other"
fi

echo "Sistema operativo: $OS"
echo "Versión CUDA: $CUDA_VERSION"
if [[ "$INSTALL_INSTANTID" == "--instantid" ]]; then
    echo "Face Reference / InstantID: incluido"
else
    echo "Face Reference / InstantID: opcional"
fi
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no encontrado. Por favor instala Python 3.10 o superior."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python encontrado: $PYTHON_VERSION"

# Crear entorno virtual
echo ""
echo "📦 Creando entorno virtual..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Entorno virtual creado"
else
    echo "ℹ️  Entorno virtual ya existe"
fi

# Activar entorno virtual
echo ""
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Actualizar pip
echo ""
echo "⬆️  Actualizando pip..."
pip install --upgrade pip

# Instalar PyTorch según versión CUDA
echo ""
echo "🔥 Instalando PyTorch..."

case $CUDA_VERSION in
    cuda118)
        echo "Instalando PyTorch con CUDA 11.8..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        ;;
    cuda121)
        echo "Instalando PyTorch con CUDA 12.1..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        ;;
    cpu)
        echo "Instalando PyTorch para CPU..."
        pip install torch torchvision
        ;;
    *)
        echo "❌ Versión CUDA inválida. Usa: cuda118, cuda121, o cpu"
        exit 1
        ;;
esac

# Instalar dependencias
echo ""
echo "📚 Instalando dependencias..."
pip install -r requirements.txt

if [[ "$INSTALL_INSTANTID" == "--instantid" ]]; then
    echo ""
    echo "🧑 Instalando dependencias opcionales de Face Reference / InstantID..."
    pip install -r requirements-instantid.txt
fi

# Instalar xformers (opcional pero recomendado)
if [[ "$CUDA_VERSION" != "cpu" ]]; then
    echo ""
    echo "⚡ Instalando xformers (optimización de memoria)..."
    pip install xformers || echo "⚠️  xformers no se pudo instalar (no crítico)"
fi

# Crear directorios necesarios
echo ""
echo "📁 Creando directorios..."
mkdir -p models/lora
mkdir -p models/controlnet
mkdir -p models/embeddings
mkdir -p outputs
mkdir -p cache
mkdir -p extensions
chmod +x run.sh
chmod +x install_face_reference.sh

# Verificar instalación
echo ""
echo "🔍 Verificando instalación..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"

echo ""
echo "✅ ¡Instalación completada!"
echo ""
echo "Para iniciar ButterVision:"
echo "  ./run.sh"
echo ""
echo "Opciones útiles:"
echo "  ./run.sh --port 7861"
echo "  ./run.sh --share"
echo "  ./run.sh --skip-model-download"
echo "  ./install_face_reference.sh  # habilitar Face Reference / InstantID"
echo ""
echo "Para más información: cat README.md"
