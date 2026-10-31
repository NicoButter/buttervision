# 🚀 Quick Start - ButterVision

## Instalación en 3 pasos

```bash
# 1. Clonar/entrar al directorio
cd buttervision

# 2. Instalar (Linux/Mac)
bash install.sh cuda118    # Para CUDA 11.8
# O para Windows: install.bat cuda118

# 3. Lanzar
./run.sh
# O en Windows: run.bat
```

La interfaz principal abre React en `http://127.0.0.1:5173` y usa FastAPI en `127.0.0.1:7861`.

## Interfaz Gradio anterior

```bash
./run_gradio.sh
# Windows: run_gradio.bat
```

## Estructura rápida

```
buttervision/
├── main.py              # Punto de entrada interno
├── config.py            # Configuración
├── requirements.txt     # Dependencias
│
├── core/                # Motor
│   ├── advanced_pipeline.py # Generación txt2img
│   └── model_manager.py     # Descarga/resolución de modelos
│
├── ui/                  # Interfaz
│   └── interface.py
│
├── models/              # Recursos
│   └── stable-diffusion/
│
└── outputs/             # Imágenes generadas
```

## Flujo típico

1. **Lanzar React + FastAPI**: `./run.sh`
2. **Abrir React**: http://127.0.0.1:5173
3. **Gradio heredado** (opcional): `./run_gradio.sh` en http://127.0.0.1:7860
4. **Prompt**: "a beautiful landscape, oil painting"
5. **Generate** → Esperar → ¡Listo!

## Tips rápidos

### Mejores parámetros por defecto
- **Steps**: 25-30
- **CFG Scale**: 7-8
- **Resolution**: 512x512
- **Scheduler**: DPMSolverMultistepScheduler

### Para mejor calidad
- Aumentar steps a 40-50
- Usar negative prompt detallado
- Probar diferentes schedulers

### Para más velocidad
- Reducir steps a 20
- Mantener resolución baja
- Scheduler: DPM++ o Euler

## Solución rápida de problemas

| Problema | Solución |
|----------|----------|
| Out of memory en Gradio | `./run_gradio.sh --lowvram` |
| Muy lento | Instalar xformers: `pip install xformers` |
| No encuentra GPU | Reinstalar PyTorch con CUDA |
| Modelo no descarga | Configurar `CIVITAI_API_TOKEN` o colocar el `.safetensors` manualmente |

## Verificación de instalación

```bash
# Ver si CUDA funciona
python -c "import torch; print(torch.cuda.is_available())"

# Ver GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Test rápido
python test_generation.py
```

## Teclas importantes en UI

- **Ctrl+Enter** en prompt: Generar
- **Scroll** en galería: Ver más imágenes
- Click en imagen: Expandir

## Archivos de configuración

### config.py - Modelo por defecto
```python
model_id: str = "runwayml/stable-diffusion-v1-5"
```

### config.py - Parámetros por defecto
```python
default_steps: int = 30
default_cfg_scale: float = 7.5
default_width: int = 512
default_height: int = 512
```

## Links útiles

- **README completo**: [README.md](../README.md)
- **Ejemplos**: [EXAMPLES.md](EXAMPLES.md)
- **Arquitectura**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Prompts**: https://lexica.art/
- **LoRAs**: https://civitai.com/
- **Modelos**: https://huggingface.co/models?pipeline_tag=text-to-image

## Primera imagen de prueba

```python
# En la UI:
Prompt: "a serene mountain landscape at sunset, 
         golden hour lighting, photorealistic, 
         highly detailed, 8k"

Negative: "blurry, low quality, bad anatomy"

Steps: 30
CFG: 7.5
Size: 512x512
Seed: -1

Click → Generate
```

## Mantenimiento

```bash
# Limpiar cache
rm -rf cache/*

# Actualizar dependencias
pip install --upgrade -r requirements.txt

# Ver uso de VRAM
nvidia-smi

# Monitor continuo
watch -n 1 nvidia-smi
```

## Siguientes pasos

1. ✅ Instalar y lanzar
2. ✅ Generar primera imagen
3. 📖 Leer [EXAMPLES.md](EXAMPLES.md) para casos avanzados
4. 🎨 Descargar LoRAs de CivitAI
5. 🔧 Experimentar con schedulers
6. 📚 Aprender mejores prácticas de prompting

---

**¿Problemas?** Revisa [README.md](../README.md) sección "Solución de problemas"
