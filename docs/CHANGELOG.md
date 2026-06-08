# Changelog

Todas las versiones notables del proyecto serán documentadas en este archivo.

## [Unreleased]

### Por implementar
- ControlNet integration
- Inpainting/Outpainting completo
- Batch processing
- Upscaling (ESRGAN, RealESRGAN)
- Training de LoRAs
- API REST con FastAPI
- Sistema de extensiones completo
- Soporte para SDXL optimizado

## [0.1.0] - 2026-01-03

### Añadido
- ✨ Interfaz web completa con Gradio
- 🎨 Text-to-Image con soporte completo
- 🖼️ Image-to-Image con control de strength
- 🎭 Sistema de carga dinámica de LoRAs (hasta 2 simultáneos)
- ⚙️ Múltiples schedulers (DPM++, Euler, DDIM, etc.)
- ⚡ Optimizaciones para baja VRAM:
  - Float16 (FP16) precision
  - xformers memory efficient attention
  - Attention slicing
  - VAE slicing
  - CPU offloading opcional
- 🔧 Configuración centralizada (config.py)
- 📦 StableDiffusionManager con lazy loading
- 📚 LoRAManager con gestión completa
- 🌐 Servidor con opciones de host/port/share/auth
- 💾 Auto-guardado de imágenes con metadata
- 🎮 Modos predefinidos: --lowvram, --medvram
- 📝 Documentación completa:
  - README.md con guía de instalación
  - EXAMPLES.md con casos de uso
  - ARCHITECTURE.md con diseño técnico
  - QUICKSTART.md para inicio rápido
- 🔨 Scripts de instalación para Linux/Mac y Windows
- 🧪 Script de test (test_generation.py)

### Características
- Soporte para múltiples modelos de HuggingFace
- Sliders para control fino de parámetros
- Galería de imágenes interactiva
- Sistema de seeds para reproducibilidad
- Negative prompts
- Generación de múltiples imágenes simultáneas
- Limpieza de VRAM on-demand
- Descarga de pipelines para liberar memoria

### Optimizaciones
- Pipeline component sharing (txt2img/img2img)
- Lazy loading de modelos
- Garbage collection automático
- Soporte para GPUs de 4GB+

## Versiones futuras planificadas

### [0.2.0] - Planeado
- ControlNet completo
- Inpainting con modelo específico
- Upscaling integrado
- Batch processing
- Más opciones de scheduler
- Mejoras de UI

### [0.3.0] - Planeado
- API REST con FastAPI
- WebSocket para updates en tiempo real
- Queue system
- Multi-user support
- Model manager con descarga automática

### [0.4.0] - Planeado
- Sistema de extensiones/plugins completo
- Training de LoRAs integrado
- SDXL optimizado
- Textual Inversion support
- Prompt library

### [1.0.0] - Futuro
- UI avanzada con React (opcional)
- Distributed generation (multi-GPU)
- Cloud integration
- Advanced caching
- Production-ready API

---

## Formato del Changelog

Este proyecto sigue [Keep a Changelog](https://keepachangelog.com/es/1.0.0/)
y se adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### Tipos de cambios
- `Añadido` para nuevas características
- `Cambiado` para cambios en funcionalidad existente
- `Deprecado` para características que se eliminarán pronto
- `Eliminado` para características eliminadas
- `Corregido` para corrección de bugs
- `Seguridad` para vulnerabilidades
