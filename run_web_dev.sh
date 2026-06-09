#!/bin/bash
set -e

cd "$(dirname "$0")/web"

if [ ! -d "node_modules" ]; then
    echo "Faltan dependencias web. Ejecuta: cd web && npm install"
    exit 1
fi

VITE_API_BASE=${VITE_API_BASE:-http://localhost:7860} npm run dev
