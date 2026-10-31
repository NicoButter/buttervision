@echo off
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo No existe el entorno virtual venv. Ejecuta install.bat primero.
    exit /b 1
)
if not exist "web\node_modules\.bin\vite.cmd" (
    echo Faltan dependencias web. Ejecuta: cd web ^&^& npm install
    exit /b 1
)

start "Buttervision FastAPI" "%ComSpec%" /k ""%~dp0run_api.bat""
start "Buttervision React" "%ComSpec%" /k ""%~dp0run_web_dev.bat""
echo React:   http://127.0.0.1:5173
echo FastAPI: http://127.0.0.1:7861
