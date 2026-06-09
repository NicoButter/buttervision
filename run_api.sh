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

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-7860}

python -m uvicorn api.app:app --host "$HOST" --port "$PORT"
