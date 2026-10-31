# ButterVision - Stable Diffusion WebUI

Una interfaz minimalista para generar imágenes con Stable Diffusion desde texto.

## 🚀 Instalación Rápida

1. **Configurar entorno virtual:**
   ```bash
   bash setup.sh
   ```

2. **Ejecutar la aplicación:**
   ```bash
   ./run.sh
   ```

## 📋 Requisitos del Sistema

- **Python 3.10+**
- **GPU con CUDA** (GTX 1650 o superior recomendado)
- **8GB RAM mínimo**

## 🎨 Características MVP

- ✅ **Interfaz Text-to-Image**
- ✅ **Descarga/verificación del modelo base al primer arranque**
- ✅ **Optimizaciones para baja VRAM**
- ✅ **Parámetros básicos**: prompt, negative prompt, steps, CFG, tamaño y seed

## 🛠️ Uso Manual

Si prefieres configurar manualmente:

```bash
# Crear entorno virtual
python3.10 -m venv venv

# Activar
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
./run.sh
```

## 📁 Estructura del Proyecto

```
buttervision/
├── main.py              # Punto de entrada
├── core/                # Lógica del pipeline
├── ui/                  # Interfaz Gradio
├── models/              # Modelos y LoRAs
├── outputs/             # Imágenes generadas
├── requirements.txt     # Dependencias
├── setup.sh            # Script de instalación
├── run.bat             # Inicio principal en Windows
├── run.sh              # Inicio principal: React + FastAPI
└── run_gradio.sh       # Interfaz Gradio anterior
```

## 🎯 Funcionalidad Disponible

Por ahora ButterVision expone solo **Text to Image**. El resto de módulos se agregará después de estabilizar este flujo.

## 🚀 Lanzadores locales

El comando principal inicia React y FastAPI juntos en `127.0.0.1:5173` y `127.0.0.1:7861`:

Linux/Mac:

```bash
./run.sh
```

Gradio continúa disponible por separado en `127.0.0.1:7860`:

```bash
./run_gradio.sh
```

`run_api.sh` y `run_web_dev.sh` se conservan solo para iniciar componentes individualmente en tareas de desarrollo. No se enlazan a la red por defecto.

Windows: `run.bat` inicia la interfaz principal y `run_gradio.bat` inicia Gradio. `run_api.bat` y `run_web_dev.bat` quedan disponibles para desarrollo.

**Privacidad:** `/outputs` sirve imágenes sin autenticación; archivos nombrados `reference_face.png`, prompts y metadatos no se descargan por esa ruta. Las fotos faciales ya existentes se conservan localmente, pero quedan bloqueadas desde el servidor estático. No cambies los hosts a direcciones públicas en una red no confiable.
