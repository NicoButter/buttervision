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
├── run.bat             # Lanzador Windows
└── run.sh              # Script de ejecución
```

## 🎯 Funcionalidad Disponible

Por ahora ButterVision expone solo **Text to Image**. El resto de módulos se agregará después de estabilizar este flujo.

## 🚀 Lanzador

Linux/Mac:

```bash
./run.sh
./run.sh --port 7861
./run.sh --share
```

Windows:

```bat
run.bat
run.bat --port 7861
run.bat --share
```
