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
./run.sh
```

En el primer arranque, ButterVision verifica el modelo base configurado y lo descarga automáticamente si no está disponible localmente. Por defecto usa `cyberrealistic_final.safetensors`, descargado desde el backup público de Hugging Face de CyberRealistic.

Durante la descarga verás progreso en la terminal con porcentaje, tamaño descargado, tamaño total, velocidad y ETA. Si la descarga se interrumpe, ButterVision intenta reanudar el archivo `.part` en el siguiente arranque.

La página de CivitAI puede pedir autenticación, por eso el launcher usa Hugging Face como fuente principal. Si aun así quieres usar CivitAI, crea un API token en tu cuenta y ejecútalo así:

```bash
export CIVITAI_API_TOKEN="tu_token"
./run.sh
```

También puedes descargarlo manualmente y colocarlo en:

```text
models/stable-diffusion/cyberrealistic_final.safetensors
```

La interfaz se abrirá automáticamente en: `http://localhost:7860`

En Windows, usa:

```bat
run.bat
```

### Opciones de línea de comandos

#### Configuración del servidor

```bash
# Cambiar puerto
./run.sh --port 7861

# Crear share link público (Gradio)
./run.sh --share

# Añadir autenticación
./run.sh --auth usuario:contraseña

# Cambiar host
./run.sh --host 127.0.0.1
```

#### Optimizaciones de VRAM

```bash
# GPU con poca VRAM (< 4GB) - Activa todas las optimizaciones + CPU offload
./run.sh --lowvram

# GPU con VRAM media (4-6GB) - Optimizaciones sin CPU offload
./run.sh --medvram

# Desactivar todas las optimizaciones (para debugging)
./run.sh --no-optimizations
```

#### Configuración del modelo

```bash
# Usar un modelo diferente
./run.sh --model "stabilityai/stable-diffusion-2-1"

# Modo offline estricto: no descargar modelos al arrancar
./run.sh --skip-model-download

# Desactivar float16 (usa más VRAM)
./run.sh --no-fp16

# Desactivar xformers
./run.sh --no-xformers
```

#### Combinaciones útiles

```bash
# Para GPU de 4GB (ej: GTX 1650)
./run.sh --lowvram --share

# Para GPU de 6GB (ej: RTX 3060)
./run.sh --medvram

# Para GPU de 8GB+ (ej: RTX 3070)
./run.sh

# Usar modelo SD 2.1 con optimizaciones
./run.sh --model "stabilityai/stable-diffusion-2-1" --medvram
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

### 🎭 LoRA de Cara Personal

ButterVision incluye soporte especial para entrenar y usar un LoRA de tu propia cara:

#### Preparar Dataset
```bash
# Crear directorio para fotos
mkdir -p data/mi_cara

# Coloca 15-30 fotos de tu cara (formatos: .jpg, .png)
# Variedad: ángulos, iluminación, expresiones
```

#### Entrenar LoRA
```bash
# Verificar configuración
python check_lora_setup.py

# Entrenar LoRA (30-60 minutos en GTX 1650)
python train_lora_mi_cara.py

# O con parámetros personalizados
python train_lora_mi_cara.py --steps 2000 --lr 5e-5
```

#### Usar en Generación
El LoRA se carga automáticamente. Usa prompts como:
- `foto de [tu nombre], cara realista, sonrisa, fondo neutro`
- `[tu nombre] en un parque, iluminación natural`

Ver [LORA_TRAINING_README.md](LORA_TRAINING_README.md) para guía completa.

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
./run.sh --lowvram

# O reduce la resolución de generación
# Usa 384x384 o 448x448 en lugar de 512x512
```

### "xformers not available"

```bash
# Instala xformers (mejora significativa)
pip install xformers

# O desactívalo si da problemas
./run.sh --no-xformers
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
