# Ejemplos de uso de ButterVision

## 1. Inicio básico

```bash
# Iniciar con configuración por defecto
python main.py

# La interfaz se abrirá en http://localhost:7860
```

## 2. Optimizaciones para diferentes GPUs

```bash
# GPU con 4GB VRAM (ej: GTX 1650, RTX 3050)
python main.py --lowvram

# GPU con 6GB VRAM (ej: RTX 2060, RTX 3060)
python main.py --medvram

# GPU con 8GB+ VRAM (ej: RTX 3070, RTX 4070)
python main.py

# GPU con 12GB+ VRAM (sin optimizaciones)
python main.py --no-optimizations
```

## 3. Modelos alternativos

```bash
# Stable Diffusion 2.1 (mejor calidad)
python main.py --model "stabilityai/stable-diffusion-2-1" --medvram

# Stable Diffusion XL (requiere 8GB+ VRAM)
python main.py --model "stabilityai/stable-diffusion-xl-base-1.0" --medvram

# Modelo personalizado de HuggingFace
python main.py --model "username/model-name"
```

## 4. Configuración de servidor

```bash
# Compartir públicamente (Gradio share link)
python main.py --share

# Cambiar puerto
python main.py --port 8080

# Solo acceso local
python main.py --host 127.0.0.1 --port 7860

# Con autenticación
python main.py --auth usuario:micontraseña --share
```

## 5. Combinaciones útiles

```bash
# Para desarrollo local con GPU de 6GB
python main.py --medvram --port 7861

# Para servidor público con GPU de 4GB
python main.py --lowvram --share --auth admin:secret123

# Para debugging (sin optimizaciones)
python main.py --no-optimizations --no-fp16
```

## 6. Uso de LoRAs

### Preparación:
1. Descarga archivos `.safetensors` de LoRAs desde CivitAI o HuggingFace
2. Colócalos en `models/lora/`
3. Reinicia la UI o usa el botón "Refrescar LoRAs"

### En la interfaz:
1. Ve a la pestaña "Text-to-Image"
2. Expande el acordeón "🎭 LoRAs"
3. Selecciona hasta 2 LoRAs
4. Ajusta sus pesos (0.5-1.0 típico)
5. Genera normalmente

## 7. Mejores prácticas para prompts

### Text-to-Image:
```
Prompt: "a beautiful cyborg woman, cyberpunk style, neon lights, highly detailed, 8k, masterpiece"
Negative: "worst quality, low quality, blurry, artifacts, bad anatomy, deformed"
Steps: 30
CFG Scale: 7.5
```

### Image-to-Image:
```
Strength: 0.5-0.7 para mantener la composición original
Strength: 0.8-1.0 para transformación completa
```

## 8. Schedulers recomendados

```
DPMSolverMultistepScheduler  → Rápido y buena calidad (recomendado)
EulerAncestralDiscreteScheduler → Creativo, buenos detalles
DDIMScheduler → Determinista (mismo seed = misma imagen)
```

## 9. Resoluciones recomendadas

### SD 1.5:
- 512x512 (estándar, óptimo)
- 512x768 (retrato)
- 768x512 (paisaje)

### SD 2.1:
- 768x768 (estándar)
- 512x768, 768x512 (alternativas)

### SDXL:
- 1024x1024 (estándar)
- 1024x1536 (retrato)

## 10. Test rápido sin UI

```bash
# Ejecutar script de prueba
python test_generation.py

# Generará una imagen en outputs/test_image.png
```

## 11. Troubleshooting común

### Problema: "CUDA out of memory"
```bash
# Solución 1: Activar lowvram
python main.py --lowvram

# Solución 2: Reducir resolución
# Usa 384x384 o 448x448 en lugar de 512x512
```

### Problema: "xformers not available"
```bash
# Instalar xformers
pip install xformers

# O desactivar
python main.py --no-xformers
```

### Problema: Generación muy lenta
```bash
# 1. Verificar que CUDA esté activo
python -c "import torch; print(torch.cuda.is_available())"

# 2. Usar schedulers más rápidos (DPM++)
# 3. Reducir steps a 20-25
# 4. Instalar xformers
pip install xformers
```

## 12. Estructura de prompts avanzada

### Formato básico:
```
[sujeto principal], [estilo], [calidad], [detalles adicionales]
```

### Ejemplo:
```
Prompt: "portrait of a warrior princess, fantasy art style, intricate armor, 
flowing hair, dramatic lighting, highly detailed, 8k, artstation trending"

Negative: "low quality, blurry, deformed, bad anatomy, ugly, watermark, 
signature, text"
```

### Pesos en prompts (syntax CLIP):
```
(keyword:1.2)  → Enfatizar al 120%
(keyword:0.8)  → Reducir al 80%
[keyword]      → Enfatizar levemente
```

## 13. Workflow típico

1. **Inicio**: `python main.py --medvram`
2. **Experimentación**: Usar seed -1, pocos steps (20-25)
3. **Refinamiento**: Fijar seed, aumentar steps (30-50)
4. **Variaciones**: Cambiar scheduler, ajustar CFG
5. **LoRAs**: Añadir para estilos específicos
6. **Img2img**: Refinar resultado con prompts adicionales

## 14. Recursos adicionales

- **Prompts**: https://lexica.art/
- **LoRAs**: https://civitai.com/
- **Modelos**: https://huggingface.co/models?pipeline_tag=text-to-image
- **Guías**: https://stable-diffusion-art.com/

## 15. Atajos y tips

```bash
# Ver info CUDA rápido
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Limpiar cache de HuggingFace (liberar espacio)
rm -rf cache/*

# Ver VRAM en uso
nvidia-smi

# Monitor en tiempo real
watch -n 1 nvidia-smi
```
