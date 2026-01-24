#!/bin/bash
# Downgrade a stack ESTABLE para GTX 1650
# PyTorch 2.1.2 + CUDA 11.8

echo "🔄 DOWNGRADE A STACK ESTABLE - GTX 1650"
echo "PyTorch 2.1.2 + CUDA 11.8"
echo "=" * 50

# Confirmar
read -p "¿Estás seguro de hacer downgrade? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelado."
    exit 1
fi

echo "🧹 Desinstalando PyTorch actual..."
pip uninstall torch torchvision torchaudio -y

echo "📦 Instalando PyTorch 2.1.2 + CUDA 11.8..."
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

echo "🔍 Verificando instalación..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

echo "✅ Downgrade completado"
echo "Ahora puedes usar fp16 normalmente sin NaNs"