@echo off
cd /d "%~dp0web"

if not exist "node_modules\.bin\vite.cmd" (
    echo Faltan dependencias web. Ejecuta: npm install
    exit /b 1
)

call "node_modules\.bin\vite.cmd"
