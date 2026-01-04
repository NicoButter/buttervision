# Arquitectura de ButterVision

## 📐 Diseño del sistema

ButterVision está diseñado con una arquitectura modular y extensible:

```
┌─────────────────────────────────────────────┐
│           Interfaz de Usuario               │
│              (Gradio WebUI)                 │
│  - Text-to-Image  - Image-to-Image          │
│  - Inpainting     - Extras                  │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│         Capa de Presentación (UI)           │
│          ui/interface.py                    │
│  - ButterVisionUI                           │
│  - Event handlers                           │
│  - UI components                            │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│         Capa de Lógica (Core)               │
│                                             │
│  ┌───────────────────────────────────────┐ │
│  │  StableDiffusionManager               │ │
│  │  (core/pipeline.py)                   │ │
│  │  - Pipeline management                │ │
│  │  - Optimization control               │ │
│  │  - Generation methods                 │ │
│  └───────────────────────────────────────┘ │
│                                             │
│  ┌───────────────────────────────────────┐ │
│  │  LoRAManager                          │ │
│  │  (core/lora_manager.py)               │ │
│  │  - LoRA loading/unloading             │ │
│  │  - Weight management                  │ │
│  └───────────────────────────────────────┘ │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│    Framework Layer (HuggingFace)            │
│  - Diffusers (pipelines)                    │
│  - Transformers (CLIP)                      │
│  - PyTorch (backend)                        │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│          Hardware Layer                     │
│  - CUDA/CUDNN (GPU)                         │
│  - CPU (fallback)                           │
└─────────────────────────────────────────────┘
```

## 🔧 Módulos principales

### 1. config.py
Configuración centralizada del sistema.

**Clases:**
- `ModelConfig`: Configuración del modelo y optimizaciones
- `ServerConfig`: Configuración del servidor web
- `UIConfig`: Configuración de la interfaz

**Responsabilidades:**
- Rutas del proyecto
- Parámetros por defecto
- Flags de optimización

### 2. core/pipeline.py
Gestión del pipeline de Stable Diffusion.

**Clase principal:** `StableDiffusionManager`

**Métodos clave:**
- `load_txt2img_pipeline()`: Carga pipeline de text-to-image
- `load_img2img_pipeline()`: Carga pipeline de image-to-image
- `generate_txt2img()`: Generación desde texto
- `generate_img2img()`: Transformación de imágenes
- `_apply_optimizations()`: Aplica optimizaciones de memoria
- `change_scheduler()`: Cambia el scheduler de sampling

**Optimizaciones implementadas:**
- Float16 (FP16) precision
- xformers memory efficient attention
- Attention slicing
- VAE slicing
- CPU offloading (para VRAM extremadamente baja)

### 3. core/lora_manager.py
Gestión de LoRAs (Low-Rank Adaptations).

**Clase principal:** `LoRAManager`

**Métodos clave:**
- `scan_lora_directory()`: Escanea directorio de LoRAs
- `load_lora()`: Carga un LoRA en el pipeline
- `unload_lora()`: Descarga un LoRA
- `update_lora_weight()`: Actualiza peso de un LoRA
- `load_multiple_loras()`: Carga múltiples LoRAs simultáneamente

### 4. ui/interface.py
Interfaz de usuario con Gradio.

**Clase principal:** `ButterVisionUI`

**Métodos clave:**
- `create_interface()`: Construye la interfaz completa
- `txt2img_generate()`: Handler para generación txt2img
- `img2img_generate()`: Handler para generación img2img
- `refresh_loras()`: Refresca lista de LoRAs
- `_save_images()`: Guarda imágenes con metadata

**Pestañas:**
- Text-to-Image: Generación desde texto
- Image-to-Image: Transformación de imágenes
- Extras: Herramientas adicionales

### 5. main.py
Punto de entrada principal.

**Funciones:**
- `parse_args()`: Parseo de argumentos CLI
- `apply_launch_config()`: Aplica configuración de lanzamiento
- `print_system_info()`: Muestra info del sistema
- `main()`: Función principal

## 🔄 Flujo de ejecución

### Inicio de la aplicación:

```
1. main.py ejecutado
   ↓
2. Parse argumentos CLI
   ↓
3. Aplicar configuración (config.py)
   ↓
4. Mostrar info del sistema
   ↓
5. Crear interfaz (ui/interface.py)
   ↓
6. Inicializar ButterVisionUI
   ↓
7. Crear StableDiffusionManager
   ↓
8. Escanear LoRAs disponibles
   ↓
9. Lanzar servidor Gradio
   ↓
10. Interfaz disponible en http://localhost:7860
```

### Generación de imagen (txt2img):

```
1. Usuario ingresa prompt y parámetros
   ↓
2. Clic en botón "Generate"
   ↓
3. txt2img_generate() llamado
   ↓
4. Verificar/cambiar scheduler
   ↓
5. Cargar pipeline txt2img (si no está cargado)
   ├─ Aplicar optimizaciones
   ├─ Configurar scheduler
   └─ Mover a GPU/CPU
   ↓
6. Gestionar LoRAs
   ├─ Descargar LoRAs previos
   ├─ Cargar LoRAs seleccionados
   └─ Configurar pesos
   ↓
7. Generar seed (si es -1)
   ↓
8. Llamar pipeline.generate_txt2img()
   ├─ Tokenizar prompt
   ├─ Generar embeddings
   ├─ Denoising loop (steps)
   ├─ Aplicar CFG
   └─ Decodificar VAE
   ↓
9. Guardar imágenes en outputs/
   ↓
10. Retornar imágenes y metadata
   ↓
11. Mostrar en galería UI
```

## 💾 Gestión de memoria

### Estrategia de carga de pipelines:

1. **Lazy loading**: Los pipelines se cargan bajo demanda
2. **Component sharing**: txt2img/img2img comparten componentes
3. **Explicit unloading**: Método `unload_pipeline()` para liberar VRAM

### Optimizaciones por nivel de VRAM:

| VRAM | Modo | Optimizaciones |
|------|------|----------------|
| < 4GB | `--lowvram` | FP16 + xformers + slicing + CPU offload |
| 4-6GB | `--medvram` | FP16 + xformers + slicing |
| 6GB+ | Normal | FP16 + xformers |
| 8GB+ | `--no-optimizations` | Ninguna |

## 🔌 Extensibilidad

### Añadir nuevo generador:

1. Crear método en `StableDiffusionManager`:
```python
def generate_inpaint(self, image, mask, prompt, ...):
    pipe = self.load_inpaint_pipeline()
    # ... lógica de generación
    return images
```

2. Añadir handler en `ButterVisionUI`:
```python
def inpaint_generate(self, image, mask, prompt, ...):
    images = self.sd_manager.generate_inpaint(...)
    return images, info
```

3. Añadir pestaña en `create_interface()`:
```python
with gr.Tab("🎨 Inpainting"):
    # ... componentes UI
```

### Añadir soporte para ControlNet:

1. Extender `StableDiffusionManager` con métodos ControlNet
2. Crear `core/controlnet_manager.py` similar a `lora_manager.py`
3. Añadir pestaña UI con controles específicos

### Sistema de plugins (futuro):

```python
# extensions/my_plugin/__init__.py
class MyPlugin:
    def __init__(self, sd_manager):
        self.sd_manager = sd_manager
    
    def on_generate_start(self, params):
        # Hook antes de generar
        pass
    
    def on_generate_end(self, images):
        # Hook después de generar
        return images
```

## 📊 Diagrama de dependencias

```
main.py
  ├── config.py
  ├── ui/interface.py
  │     ├── core/pipeline.py
  │     │     └── diffusers
  │     │           └── torch
  │     └── core/lora_manager.py
  │           └── config.py
  └── gradio
```

## 🔒 Consideraciones de seguridad

1. **Safety checker**: Opcional, desactivado por defecto para ahorrar VRAM
2. **Autenticación**: Soportada vía `--auth usuario:contraseña`
3. **Validación de inputs**: Límites en resolución y parámetros
4. **Sandboxing**: Extensiones futuras ejecutarán en contexto limitado

## ⚡ Optimizaciones de rendimiento

### 1. Pipeline caching
- Reutilización de componentes entre txt2img/img2img
- Avoid reloading cuando solo cambian parámetros

### 2. Attention optimization
- xformers: Reduce VRAM significativamente
- Attention slicing: Divide cálculos en chunks
- VAE slicing: Procesa VAE en batches

### 3. Precision mixing
- FP16 para UNet (ahorra VRAM)
- Opción de FP32 para mayor precisión

### 4. Batching inteligente
- Generación de múltiples imágenes en un solo pase
- Configuración de `num_images_per_prompt`

## 🚀 Roadmap técnico

### Corto plazo:
- [ ] Caché de embeddings de texto
- [ ] Compilación de modelos (torch.compile)
- [ ] WebSocket para updates en tiempo real

### Medio plazo:
- [ ] API REST con FastAPI
- [ ] Queue system para múltiples usuarios
- [ ] Model manager (descarga automática)

### Largo plazo:
- [ ] Distributed generation (multi-GPU)
- [ ] Cloud integration
- [ ] Advanced plugin system con hot-reload

## 📚 Referencias

- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Gradio Documentation](https://gradio.app/docs)
- [PyTorch Optimization](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [xformers](https://github.com/facebookresearch/xformers)
