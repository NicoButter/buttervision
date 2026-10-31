#!/bin/bash
set -e

cd "$(dirname "$0")/web"

if [ ! -d "node_modules" ]; then
    echo "Faltan dependencias web. Ejecuta: cd web && npm install"
    exit 1
fi

npm run dev
