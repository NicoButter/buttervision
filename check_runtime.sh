#!/bin/bash
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

echo "ButterVision runtime check"
echo "=========================="

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
    echo "nvidia-smi no encontrado"
fi

python - <<'PY'
import config
import torch
import onnxruntime as ort

print(f"Profile: {config.model_config.hardware_profile}")
print(f"Low VRAM defaults: {config.model_config.default_width}x{config.model_config.default_height}, batch={config.model_config.default_batch_size}, steps={config.model_config.default_steps}")
print(f"Face Reference defaults: {config.model_config.face_default_width}x{config.model_config.face_default_height}, steps={config.model_config.face_default_steps}")
print(f"Face Reference fp16/offload: {config.model_config.face_use_fp16}/{config.model_config.face_enable_cpu_offload}")
print(f"Torch: {torch.__version__}")
print(f"Torch CUDA build: {torch.version.cuda}")
print(f"Torch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Torch GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"Torch VRAM: {props.total_memory / 1024**3:.2f} GB")
print(f"ONNXRuntime: {ort.__version__}")
print(f"ONNXRuntime providers: {ort.get_available_providers()}")
print(f"ONNXRuntime device: {ort.get_device()}")
PY
