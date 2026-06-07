@echo off
cd /d "%~dp0"

if not exist venv\Scripts\activate.bat (
    echo No existe el entorno virtual venv.
    echo Primero ejecuta: install.bat cuda121
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python main.py %*
