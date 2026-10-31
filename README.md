# 🎨 Buttervision

Aplicación local para crear imágenes mediante inteligencia artificial. React es la interfaz principal de desarrollo; la interfaz Gradio se conserva para compatibilidad y diagnóstico.

## ✨ Características

- **Interfaz React en español** con modo sencillo y modo experto
- **Crear imagen**: generación conectada a la API local existente, con o sin referencia facial
- **Mis creaciones**: galería de imágenes devuelta por la API
- **Crear video y editar medios**: pantallas preparadas, todavía sin procesamiento conectado
- **Gradio**: interfaz anterior disponible para compatibilidad y diagnóstico
- **Soporte para LoRAs**: Carga dinámicamente múltiples LoRAs
- **Optimizado para baja VRAM**: defaults conservadores para GTX 1650 / 4GB VRAM
- **Múltiples schedulers**: DPM++, Euler, DDIM, etc.
- **Sistema extensible**: Arquitectura modular para añadir plugins

## 📋 Requisitos

- Python 3.10 o superior
- GPU NVIDIA con CUDA (recomendado 4GB+ VRAM)
- 10GB+ de espacio en disco

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/nicobutter/buttervision.git
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

Esto inicia la interfaz principal React en `http://127.0.0.1:5173` y FastAPI en `http://127.0.0.1:7861`. En Windows, usa `run.bat`.

La interfaz Gradio anterior sigue disponible con `./run_gradio.sh` en `http://127.0.0.1:7860`. Sus opciones de servidor y VRAM se pasan a ese lanzador; por ejemplo:

```bash
./run_gradio.sh --lowvram
./run_gradio.sh --share
```

Al usar Gradio, ButterVision verifica el modelo base configurado y lo descarga automáticamente si no está disponible localmente. Por defecto usa `cyberrealistic_final.safetensors`, descargado desde el backup público de Hugging Face de CyberRealistic.

Durante la descarga verás progreso en la terminal con porcentaje, tamaño descargado, tamaño total, velocidad y ETA. Si la descarga se interrumpe, ButterVision intenta reanudar el archivo `.part` en el siguiente arranque.

La página de CivitAI puede pedir autenticación, por eso el launcher usa Hugging Face como fuente principal. Si aun así quieres usar CivitAI, crea un API token en tu cuenta y ejecútalo así:

```bash
export CIVITAI_API_TOKEN="tu_token"
./run_gradio.sh
```

También puedes descargarlo manualmente y colocarlo en:

```text
models/stable-diffusion/cyberrealistic_final.safetensors
```

`run.sh` es el único lanzador principal para React + FastAPI. `run_api.sh` y `run_web_dev.sh` se conservan para desarrollo y diagnóstico. En Windows, `run.bat` inicia la interfaz principal y `run_gradio.bat` inicia la interfaz anterior.

**Privacidad y seguridad:** los lanzadores enlazan a loopback. `/outputs` sigue sirviendo imágenes generadas sin autenticación; los archivos de referencia existentes quedan en disco, pero las rutas conocidas de referencias, prompts y metadatos ya no se sirven por esa ruta. Las nuevas referencias no se guardan en `outputs/`. Protege la máquina local y no cambies el host a una interfaz pública: no hay autenticación, control de acceso por archivo ni retención automatizada.

Los límites se configuran mediante `BUTTERVISION_MAX_IMAGE_UPLOAD_BYTES` (10 MiB por defecto) y `BUTTERVISION_MAX_VIDEO_UPLOAD_BYTES` (100 MiB por defecto); el frontend los obtiene de FastAPI. Se admiten JPEG, PNG y WebP para imágenes; MP4, WebM y MOV para videos. El endpoint de referencia facial vuelve a validar tamaño, formato MIME declarado y contenido de imagen.

En Windows, usa:

```bat
run.bat
```

### Opciones de línea de comandos

#### Configuración del servidor

```bash
# Cambiar puerto de Gradio (por defecto 7860)
./run_gradio.sh --port 7862

# Crear share link público (Gradio)
./run_gradio.sh --share

# Añadir autenticación
./run_gradio.sh --auth usuario:contraseña

```

#### Optimizaciones de VRAM

```bash
# GPU de 4GB (GTX 1650) - slicing + CPU offload, sin xformers/fp16 por defecto
./run_gradio.sh --lowvram

# GPU con VRAM media (6GB+) - fp16/xformers si tu GPU los tolera bien
./run_gradio.sh --medvram

# Desactivar todas las optimizaciones (para debugging)
./run_gradio.sh --no-optimizations
```

#### Configuración del modelo

```bash
# Usar un modelo diferente
./run_gradio.sh --model "stabilityai/stable-diffusion-2-1"

# Modo offline estricto: no descargar modelos al arrancar
./run_gradio.sh --skip-model-download

# Desactivar float16 (usa más VRAM)
./run_gradio.sh --no-fp16

# Desactivar xformers
./run_gradio.sh --no-xformers
```

#### Combinaciones útiles

```bash
# Para GPU de 4GB (ej: GTX 1650)
./run_gradio.sh --lowvram --share

# Para GPU de 6GB (ej: RTX 3060)
./run_gradio.sh --medvram

# Para GPU de 8GB+ (ej: RTX 3070)
./run_gradio.sh

# Usar modelo SD 2.1 con optimizaciones
./run_gradio.sh --model "stabilityai/stable-diffusion-2-1" --medvram
```

## 📁 Estructura del proyecto

```
buttervision/
├── main.py                 # Punto de entrada principal
├── config.py              # Configuración centralizada
├── requirements.txt       # Dependencias Python
├── requirements-instantid.txt # Dependencias opcionales de Face Reference
├── README.md             # Este archivo
├── LICENSE               # Licencia del proyecto
├── docs/                 # Documentación secundaria del proyecto
│
├── core/                 # Núcleo del sistema
│   ├── __init__.py
│   ├── pipeline.py       # StableDiffusionManager
│   ├── instantid_pipeline.py # Backend Face Reference / InstantID
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

Regla de documentación: `README.md` queda en la raíz para GitHub. Todo documento adicional (`.md` o `.txt`) debe vivir en `docs/`. Archivos funcionales como `requirements*.txt` permanecen en la raíz.

## 🎨 Interfaces disponibles

### React (interfaz principal)

La aplicación presenta cuatro flujos: crear imagen, crear video, editar imagen y editar video. Solo la generación de imágenes y la galería están conectadas actualmente a la API. Crear video y editar medios muestran estados pendientes y no simulan procesamiento. El modo experto permite ajustar los parámetros que admite la API para la generación de imágenes.

La API local implementada ofrece `POST /api/generate/txt2img`, `POST /api/generate/face-reference`, `GET /api/jobs/{job_id}`, `GET /api/history`, `GET /api/models`, `POST /api/models/active` y `GET /api/runtime`.

### Gradio (compatibilidad y diagnóstico)

La interfaz Gradio previa sigue disponible con `./run_gradio.sh` en `http://127.0.0.1:7860`. Sus controles y capacidades no son la interfaz principal de React.

#### Text-to-Image en Gradio

1. Escribe tu prompt en el campo de texto
2. (Opcional) Añade un negative prompt
3. Ajusta los parámetros:
   - **Steps**: 20-50 para calidad (más = más lento)
   - **CFG Scale**: 5 por defecto para estabilidad en GTX 1650
   - **Width/Height**: 512x512 por defecto
   - **Seed**: -1 para aleatorio
   - **Batch**: 1-4 imágenes por generación
4. Haz clic en "Generate"
5. Cada generación se guarda en `outputs/DDMMYYYY-HHMMSS-generation/` con los PNG, `metadata.json` y `prompt.txt`. La seed aleatoria usada vuelve al campo Seed para poder repetir el resultado.

### Face Reference

Face Reference usa **InstantID** para preservar identidad facial desde una sola imagen de referencia. Esta técnica combina embeddings faciales de InsightFace con guía estructural tipo ControlNet sobre SDXL, por lo que es más fiel para identidad que un `img2img` simple.

1. Selecciona `Face Reference` en el dropdown del navbar
2. Sube una imagen clara de una cara
3. Escribe el prompt y el negative prompt
4. Ajusta `Identity Strength` para más parecido facial y `Structure Strength` para más guía de landmarks
5. Haz clic en "Generate Face Reference"

Cada generación guarda los PNG, `metadata.json` y `prompt.txt` en `outputs/DDMMYYYY-HHMMSS-generation/`. Las generaciones nuevas no guardan la fotografía de referencia.

El baseline de ButterVision es GTX 1650 / 4GB VRAM: Face Reference arranca en `512x512`, `15 steps` y `Batch = 1`. `768x768` queda disponible como techo conservador; resoluciones mayores con InstantID/SDXL pueden hacer que el sistema mate el proceso por falta de memoria.

Face Reference mantiene sus dependencias separadas del instalador base porque `insightface` y `onnxruntime-gpu` son paquetes pesados y específicos de esta función. Para habilitarlo:

```bash
./install_face_reference.sh
```

También puedes instalar todo junto desde cero:

```bash
bash install.sh cuda121 --instantid
```

Los modelos/pesos se descargan automáticamente la primera vez que seleccionas `Face Reference`:

```text
models/instantid/
├── ControlNetModel/
└── ip-adapter.bin

models/antelopev2/
```

Nota: los modelos de InsightFace/FaceID suelen estar restringidos a investigación o uso no comercial según sus licencias upstream.

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

Ver [LORA_TRAINING_README.md](docs/LORA_TRAINING_README.md) para guía completa.

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
./run_gradio.sh --lowvram

# O reduce la resolución de generación
# Usa 384x384 o 448x448 en lugar de 512x512
```

### "xformers not available"

```bash
# Instala xformers (mejora significativa)
pip install xformers

# O desactívalo si da problemas
./run_gradio.sh --no-xformers
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

## 👤 Autor

**Nicolas Butterfield**

- Email: [nicobutter@gmail.com](mailto:nicobutter@gmail.com)
- GitHub: [nicobutter](https://github.com/nicobutter)

## 🙏 Agradecimientos

- [Stability AI](https://stability.ai/) por Stable Diffusion
- [Hugging Face](https://huggingface.co/) por Diffusers
- [Gradio](https://gradio.app/) por la interfaz web
- [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) por la inspiración

## 📞 Soporte

¿Problemas o preguntas? Abre un issue en GitHub o contacta a Nicolas Butterfield en [nicobutter@gmail.com](mailto:nicobutter@gmail.com).

---

**¡Disfruta generando arte con ButterVision! 🎨✨**
Sistema de generacion de imagenes
