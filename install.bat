@echo off
REM Script de instalación para Windows
REM Uso: install.bat [cuda118|cuda121|cpu]

setlocal
set CUDA_VERSION=%1
if "%CUDA_VERSION%"=="" set CUDA_VERSION=cuda118

echo.
echo 🎨 ButterVision - Script de instalacion
echo ========================================
echo.
echo Version CUDA: %CUDA_VERSION%
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python no encontrado. Instala Python 3.10 o superior.
    pause
    exit /b 1
)

echo ✅ Python encontrado
echo.

REM Crear entorno virtual
echo 📦 Creando entorno virtual...
if not exist venv (
    python -m venv venv
    echo ✅ Entorno virtual creado
) else (
    echo ℹ️  Entorno virtual ya existe
)

REM Activar entorno virtual
echo.
echo 🔧 Activando entorno virtual...
call venv\Scripts\activate.bat

REM Actualizar pip
echo.
echo ⬆️  Actualizando pip...
python -m pip install --upgrade pip

REM Instalar PyTorch según versión CUDA
echo.
echo 🔥 Instalando PyTorch...

if "%CUDA_VERSION%"=="cuda118" (
    echo Instalando PyTorch con CUDA 11.8...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else if "%CUDA_VERSION%"=="cuda121" (
    echo Instalando PyTorch con CUDA 12.1...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) else if "%CUDA_VERSION%"=="cpu" (
    echo Instalando PyTorch para CPU...
    pip install torch torchvision
) else (
    echo ❌ Version CUDA invalida. Usa: cuda118, cuda121, o cpu
    pause
    exit /b 1
)

REM Instalar dependencias
echo.
echo 📚 Instalando dependencias...
pip install -r requirements.txt

REM Instalar xformers (opcional)
if not "%CUDA_VERSION%"=="cpu" (
    echo.
    echo ⚡ Instalando xformers...
    pip install xformers || echo ⚠️  xformers no se pudo instalar (no crítico)
)

REM Crear directorios
echo.
echo 📁 Creando directorios...
if not exist models\lora mkdir models\lora
if not exist models\controlnet mkdir models\controlnet
if not exist models\embeddings mkdir models\embeddings
if not exist outputs mkdir outputs
if not exist cache mkdir cache
if not exist extensions mkdir extensions

REM Verificar instalación
echo.
echo 🔍 Verificando instalacion...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"

echo.
echo ✅ ¡Instalacion completada!
echo.
echo Para iniciar ButterVision:
echo   1. Activa el entorno: venv\Scripts\activate.bat
echo   2. Ejecuta: python main.py
echo.
echo Opciones utiles:
echo   python main.py --lowvram    REM Para GPUs con ^< 4GB VRAM
echo   python main.py --medvram    REM Para GPUs con 4-6GB VRAM
echo   python main.py --share      REM Crear link publico
echo.
pause
