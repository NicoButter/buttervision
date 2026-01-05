# ButterVision - Stable Diffusion WebUI

Una interfaz minimalista y personalizada para Stable Diffusion con LoRA de mejora automática.

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

## 🎨 Características

- ✅ **Interfaz minimalista** con 4 pestañas
- ✅ **LoRA de mejora automática** (descarga automática)
- ✅ **Optimizaciones para baja VRAM**
- ✅ **Entrenamiento de LoRA** (interfaz preparada)
- ✅ **Controles de calidad** ajustables

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
python main.py
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
└── run.sh              # Script de ejecución
```

## 🎯 Pestañas Disponibles

1. **Text to Image** - Generación básica
2. **Image to Image** - Transformación de imágenes
3. **Train LoRA** - Entrenamiento personalizado
4. **Settings** - Configuración y gestión de modelos

¡Disfruta generando imágenes con tu ButterVision personalizado! 🎨