@echo off
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo No existe el entorno virtual venv. Ejecuta install.bat primero.
    exit /b 1
)

call "venv\Scripts\activate.bat"
python -m uvicorn api.app:app --host 127.0.0.1 --port 7861 --workers 1
