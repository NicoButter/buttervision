# 🎨 ButterVision - Stable Diffusion WebUI

WebUI ligero y personalizado para Stable Diffusion, similar a Automatic1111 pero optimizado y modular.

## ✨ Características

- **Interfaz web moderna** con Gradio
- **Text-to-Image**: Genera imágenes desde prompts de texto
- **Image-to-Image**: Transforma imágenes existentes
- **Soporte para LoRAs**: Carga dinámicamente múltiples LoRAs
- **Optimizado para baja VRAM**: Funciona con GPUs de 4GB+
- **Múltiples schedulers**: DPM++, Euler, DDIM, etc.
- **Sistema extensible**: Arquitectura modular para añadir plugins

## 📋 Requisitos

- Python 3.10 o superior
- GPU NVIDIA con CUDA (recomendado 4GB+ VRAM)
- 10GB+ de espacio en disco

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tuusuario/buttervision.git
cd buttervision
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Instalar dependencias

**Para CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Para CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Para CPU (no recomendado, muy lento):**
```bash
pip install torch torchvision
pip install -r requirements.txt
```

### 4. (Opcional) Instalar xformers

xformers proporciona optimizaciones de memoria significativas:

```bash
pip install xformers
```

## 🎮 Uso

### Inicio básico

```bash
python main.py
```

La interfaz se abrirá automáticamente en: `http://localhost:7860`

### Opciones de línea de comandos

#### Configuración del servidor

```bash
# Cambiar puerto
python main.py --port 7861

# Crear share link público (Gradio)
python main.py --share

# Añadir autenticación
python main.py --auth usuario:contraseña

# Cambiar host
python main.py --host 127.0.0.1
```

#### Optimizaciones de VRAM

```bash
# GPU con poca VRAM (< 4GB) - Activa todas las optimizaciones + CPU offload
python main.py --lowvram

# GPU con VRAM media (4-6GB) - Optimizaciones sin CPU offload
python main.py --medvram

# Desactivar todas las optimizaciones (para debugging)
python main.py --no-optimizations
```

#### Configuración del modelo

```bash
# Usar un modelo diferente
python main.py --model "stabilityai/stable-diffusion-2-1"

# Desactivar float16 (usa más VRAM)
python main.py --no-fp16

# Desactivar xformers
python main.py --no-xformers
```

#### Combinaciones útiles

```bash
# Para GPU de 4GB (ej: GTX 1650)
python main.py --lowvram --share

# Para GPU de 6GB (ej: RTX 3060)
python main.py --medvram

# Para GPU de 8GB+ (ej: RTX 3070)
python main.py

# Usar modelo SD 2.1 con optimizaciones
python main.py --model "stabilityai/stable-diffusion-2-1" --medvram
```

## 📁 Estructura del proyecto

```
buttervision/
├── main.py                 # Punto de entrada principal
├── config.py              # Configuración centralizada
├── requirements.txt       # Dependencias Python
├── README.md             # Este archivo
├── LICENSE               # Licencia del proyecto
│
├── core/                 # Núcleo del sistema
│   ├── __init__.py
│   ├── pipeline.py       # StableDiffusionManager
│   └── lora_manager.py   # Gestor de LoRAs
│
├── ui/                   # Interfaz de usuario
│   ├── __init__.py
│   └── interface.py      # Interfaz Gradio
│
├── models/               # Modelos y recursos
│   ├── lora/            # Archivos .safetensors de LoRAs
│   ├── controlnet/      # Modelos de ControlNet
│   └── embeddings/      # Textual inversions
│
├── extensions/           # Plugins/extensiones personalizadas
├── outputs/             # Imágenes generadas
└── cache/               # Cache de modelos de HuggingFace
```

## 🎨 Uso de la interfaz

### Text-to-Image

1. Escribe tu prompt en el campo de texto
2. (Opcional) Añade un negative prompt
3. Ajusta los parámetros:
   - **Steps**: 20-50 para calidad (más = más lento)
   - **CFG Scale**: 7-9 para seguir el prompt
   - **Width/Height**: 512x512 por defecto
   - **Seed**: -1 para aleatorio
4. (Opcional) Carga LoRAs desde el acordeón
5. Haz clic en "Generate"

### Image-to-Image

1. Carga una imagen inicial
2. Escribe el prompt de transformación
3. Ajusta **Strength**: 
   - 0.3-0.5: Cambios sutiles
   - 0.6-0.8: Transformación moderada
   - 0.9-1.0: Cambio completo
4. Haz clic en "Transform"

### LoRAs

1. Coloca archivos `.safetensors` de LoRAs en `models/lora/`
2. Haz clic en "🔄 Refrescar LoRAs"
3. Selecciona hasta 2 LoRAs simultáneos
4. Ajusta sus pesos (0.0 a 2.0, típico 0.5-1.0)

## 🔧 Configuración avanzada

### Cambiar modelo base

Edita [config.py](config.py):

```python
@dataclass
class ModelConfig:
    model_id: str = "stabilityai/stable-diffusion-2-1"  # Cambia aquí
    # ... resto de configuración
```

Modelos populares:
- `runwayml/stable-diffusion-v1-5` (ligero, rápido)
- `stabilityai/stable-diffusion-2-1` (mejor calidad)
- `stabilityai/stable-diffusion-xl-base-1.0` (SDXL, requiere más VRAM)

### Añadir nuevos schedulers

Los schedulers disponibles están en [config.py](config.py). Puedes añadir más editando la función `get_available_schedulers()`.

### Extender con plugins

Crea scripts Python en la carpeta `extensions/` para añadir funcionalidades personalizadas. (Sistema de plugins en desarrollo)

## 📊 Consumo de VRAM estimado

| Configuración | VRAM | Velocidad |
|--------------|------|-----------|
| SD 1.5 + lowvram | ~3GB | Lento |
| SD 1.5 + medvram | ~4GB | Moderado |
| SD 1.5 estándar | ~5GB | Rápido |
| SD 2.1 + medvram | ~5GB | Moderado |
| SD 2.1 estándar | ~6GB | Rápido |
| SDXL + lowvram | ~6GB | Muy lento |
| SDXL estándar | ~10GB | Rápido |

## 🐛 Solución de problemas

### "CUDA out of memory"

```bash
# Prueba con optimizaciones más agresivas
python main.py --lowvram

# O reduce la resolución de generación
# Usa 384x384 o 448x448 en lugar de 512x512
```

### "xformers not available"

```bash
# Instala xformers (mejora significativa)
pip install xformers

# O desactívalo si da problemas
python main.py --no-xformers
```

### El modelo se descarga muy lento

Los modelos se descargan de HuggingFace la primera vez. Para SD 1.5 son ~4GB.

Puedes pre-descargarlos:

```bash
python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')"
```

### La generación es muy lenta

1. Asegúrate de tener GPU CUDA disponible
2. Verifica que xformers esté instalado
3. Usa schedulers más rápidos: DPM++ (2M, 2M Karras) con menos steps

## 🛣️ Roadmap

- [ ] ControlNet integration
- [ ] Inpainting/Outpainting completo
- [ ] Batch processing
- [ ] Upscaling (ESRGAN, RealESRGAN)
- [ ] Training de LoRAs
- [ ] API REST con FastAPI
- [ ] Sistema de extensiones completo
- [ ] Soporte para Stable Diffusion XL
- [ ] UI más avanzada con React (opcional)

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- [Stability AI](https://stability.ai/) por Stable Diffusion
- [Hugging Face](https://huggingface.co/) por Diffusers
- [Gradio](https://gradio.app/) por la interfaz web
- [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) por la inspiración

## 📞 Soporte

¿Problemas o preguntas? Abre un issue en GitHub o contacta al desarrollador.

---

**¡Disfruta generando arte con ButterVision! 🎨✨**
Sistema de generacion de imagenes
