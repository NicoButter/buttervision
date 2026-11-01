#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# Local settings live in .env.local (KEY=VALUE, no shell syntax). Process
# environment wins over the file; parsing never evaluates the file as code.
if [ -f .env.local ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|'#'*) continue ;; esac
        key=${line%%=*}
        value=${line#*=}
        case "$key" in BUTTERVISION_*) ;; *) continue ;; esac
        if [ "$key" != "$line" ] && ! [[ -v $key ]]; then
            export "$key=$value"
        fi
    done < .env.local
fi

if [ "$#" -gt 0 ]; then
    echo "El inicio principal no recibe opciones: ./run.sh"
    echo "Para ejecutar Gradio con opciones, usa: ./run_gradio.sh [opciones]"
    exit 2
fi

if [ ! -d "venv" ]; then
    echo "No existe el entorno virtual venv."
    echo "Primero ejecuta: bash install.sh cuda121"
    exit 1
fi
if [ ! -d "web/node_modules" ]; then
    echo "Faltan dependencias web. Ejecuta: cd web && npm install"
    exit 1
fi

source venv/bin/activate
VENV_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
for nvidia_lib_dir in "$VENV_SITE"/nvidia/*/lib; do
    if [ -d "$nvidia_lib_dir" ]; then
        export LD_LIBRARY_PATH="$nvidia_lib_dir:${LD_LIBRARY_PATH:-}"
    fi
done

cleanup() {
    if [ -n "${WEB_PID:-}" ]; then kill "$WEB_PID" 2>/dev/null || true; fi
    if [ -n "${API_PID:-}" ]; then kill "$API_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

python -m uvicorn api.app:app --host 127.0.0.1 --port 7861 --workers 1 &
API_PID=$!
(cd web && ./node_modules/.bin/vite) &
WEB_PID=$!

echo "FastAPI: http://127.0.0.1:7861"
echo "React:   http://127.0.0.1:5173"
wait -n "$API_PID" "$WEB_PID"
