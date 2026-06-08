# ButterVision Advanced Pipeline - Guía de Uso

## Descripción
Pipeline completo optimizado para GTX 1650 (4GB VRAM) que soporta:
- ✅ Generación de imágenes realistas
- ✅ Generación de videos cortos con AnimateDiff
- ✅ LoRA personalizado de caras humanas
- ✅ Optimizaciones agresivas de VRAM
- ✅ Modo LCM para generación ultra-rápida

## Arquitectura

### Modelo Base
- **Stable Diffusion 1.5**: `runwayml/stable-diffusion-v1-5`
- **Checkpoint realista**: Realistic Vision v5.1 (aplicado como LoRA)
- **Optimizaciones**: fp16, xFormers, attention slicing, VAE slicing

### Videos
- **AnimateDiff**: Motion adapter v1-5-2
- **Frames**: 8-16 (optimizado para VRAM)
- **Scheduler**: DDIM

### LoRA
- **Detección automática**: Escanea `models/lora/`
- **Activación**: Por nombre y peso personalizado
- **Trigger words**: Ej: "n1c0 person"
- **Fusion**: Múltiples LoRAs soportados

## Uso Básico

```python
from core import create_buttervision_pipeline

# Crear pipeline optimizado
pipeline = create_buttervision_pipeline(enable_lcm=True)

# Cargar LoRA de cara personal
pipeline.load_lora("n1c0_person", weight=0.8)

# Generar imagen
images = pipeline.generate_image(
    prompt="foto profesional en estudio",
    lora_trigger="n1c0 person",
    num_inference_steps=20
)

# Generar video
video_frames = pipeline.generate_video(
    prompt="persona hablando en conferencia",
    lora_trigger="n1c0 person",
    num_frames=16
)

# Limpiar memoria
pipeline.cleanup()
```

## Configuraciones Recomendadas

### Para GTX 1650 (4GB VRAM)
```python
pipeline = ButterVisionPipeline(
    model_id="runwayml/stable-diffusion-v1-5",
    checkpoint_path="SG161222/Realistic_Vision_V5.1_noVAE",
    device="cuda",
    enable_optimizations=True,  # Obligatorio
    enable_lcm=True            # Para velocidad
)
```

### Parámetros Óptimos
- **Imágenes**: `steps=20, guidance=7.5, size=512x512`
- **Videos**: `steps=25, frames=16, guidance=7.5`
- **LoRA**: `weight=0.6-1.0` según intensidad deseada

## Gestión de LoRA

### Estructura de Directorios
```
models/lora/
├── n1c0_person.safetensors    # LoRA de cara
├── style_realistic.safetensors # LoRA de estilo
└── defaults/
    └── lcm_lora.safetensors   # LoRA LCM (opcional)
```

### Carga Dinámica
```python
# Cargar LoRA
pipeline.load_lora("n1c0_person", weight=0.8)

# Cambiar peso
pipeline.load_lora("n1c0_person", weight=1.2)

# Descargar LoRA
pipeline.unload_lora("n1c0_person")
```

### Trigger Words
- Define un trigger único: `"n1c0 person"`
- Úsalo en prompts: `"n1c0 person, foto smiling"`
- Evita palabras comunes para no interferir

## Optimizaciones VRAM

### Activadas por Defecto
1. **Float16**: Reduce VRAM ~50%
2. **xFormers**: Atención eficiente (si disponible)
3. **Attention Slicing**: Procesa atención en chunks
4. **VAE Slicing**: VAE en batches pequeños
5. **CPU Offload**: Como último recurso

### Monitoreo
```python
# Ver uso de VRAM
print(pipeline.get_vram_usage())
# Output: "VRAM: 2.3GB / 4.0GB"
```

## Modos de Generación

### Imágenes (SD Pipeline)
- Resolución: 512x512 (óptimo para 4GB)
- Pasos: 15-30
- CFG: 7.0-9.0
- Batch size: 1 (para estabilidad)

### Videos (AnimateDiff)
- Frames: 8-16 (más = más VRAM)
- Pasos: 20-30
- Resolución: 512x512
- Duración: ~2-4 segundos a 8fps

## LCM Mode (Opcional)

### Ventajas
- ⚡ **3-5x más rápido**
- 🎯 **Menos pasos necesarios** (4-8 steps)
- 💾 **Menos VRAM**

### Configuración
```python
pipeline = create_buttervision_pipeline(enable_lcm=True)
# Requiere: models/lora/defaults/lcm_lora.safetensors
```

### Uso con LCM
```python
images = pipeline.generate_image(
    prompt="foto portrait",
    num_inference_steps=6,  # Mucho menos steps
    guidance_scale=1.5      # Menos guidance
)
```

## Troubleshooting

### VRAM Issues
- Reduce resolución a 448x448
- Usa menos frames en videos (8)
- Activa CPU offload: `pipeline.enable_sequential_cpu_offload()`

### LoRA Issues
- Verifica nombre exacto: `pipeline.available_loras`
- Check trigger word en prompt
- Ajusta peso: 0.5-1.5

### Performance
- LCM reduce tiempo de 2min → 20seg
- xFormers acelera ~20%
- CPU offload último recurso (lento)

## Integración con UI

El pipeline está diseñado para integrarse fácilmente con la interfaz web:

```python
# En interface.py
from core import create_buttervision_pipeline

class ButterVisionUI:
    def __init__(self):
        self.pipeline = create_buttervision_pipeline()
        
    def generate_image_ui(self, prompt, lora_trigger):
        return self.pipeline.generate_image(
            prompt=prompt,
            lora_trigger=lora_trigger
        )
```

## Resumen Técnico

- **Framework**: Diffusers oficial
- **GPU**: GTX 1650 (4GB VRAM)
- **Modelo**: SD 1.5 + Realistic Vision
- **Videos**: AnimateDiff con motion adapter
- **LoRA**: PEFT con adapters múltiples
- **VRAM**: < 3.5GB en uso normal
- **Velocidad**: 20-60 seg por imagen
- **Compatibilidad**: Linux + CUDA 11.8+

¡El pipeline está listo para generar imágenes y videos realistas con tu LoRA personalizado! 🎨🎬