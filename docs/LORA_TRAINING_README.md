# 🧠 Entrenamiento LoRA Cara Personal - ButterVision

Guía completa para entrenar y usar un LoRA de tu cara en ButterVision.

## 📋 Requisitos

- **Hardware**: GTX 1650 (4GB VRAM) o superior
- **Software**: Python 3.10+, PyTorch con CUDA
- **Dataset**: 15-30 fotos de tu cara (variedad de ángulos, iluminación, expresiones)

## 📸 Preparar Dataset

### 1. Crear directorio
```bash
mkdir -p data/mi_cara
```

### 2. Recopilar fotos
- **Cantidad**: 15-30 fotos mínimas
- **Calidad**: Alta resolución (mínimo 512x512)
- **Variedad**:
  - Ángulos: frontal, 3/4, perfil
  - Expresiones: neutral, sonrisa, serio
  - Iluminación: natural, estudio, variada
  - Fondos: preferiblemente neutros o variados

### 3. Formatos soportados
- `.jpg`, `.jpeg`, `.png`
- RGB (sin transparencias)

## 🚀 Entrenar LoRA

### Instalar dependencias adicionales
```bash
pip install peft accelerate tqdm
```

### Ejecutar entrenamiento
```bash
# Entrenamiento básico (1000 pasos)
python train_lora_mi_cara.py

# Entrenamiento personalizado
python train_lora_mi_cara.py --steps 2000 --lr 5e-5 --batch_size 2
```

### Parámetros importantes
- `--steps`: Pasos de entrenamiento (1000-3000 recomendado)
- `--lr`: Learning rate (1e-4 por defecto)
- `--batch_size`: Tamaño del batch (1 para GTX 1650)

### Tiempo estimado
- **GTX 1650**: ~30-60 minutos para 1000 pasos
- **Monitorizar**: El script muestra progreso con barra

## 📁 Resultados

### Ubicación del LoRA
```
loras/mi_cara.safetensors
```

### Carga automática
El LoRA se carga automáticamente al iniciar ButterVision si existe el archivo.

## 🎨 Usar en la UI

### Prompts recomendados
```
foto de [tu nombre], cara realista, sonrisa, fondo neutro, alta calidad
[tu nombre] en un parque, iluminación natural, expresión feliz
retrato de [tu nombre], estilo profesional, iluminación de estudio
[tu nombre] con gafas, expresión seria, fondo blanco
```

### Negative prompts
```
blur, low quality, distorted face, ugly, deformed, extra limbs
```

### Configuración recomendada
- **Steps**: 20-30
- **CFG Scale**: 7-9
- **Seed**: -1 (random) o fijo para consistencia

## 🔧 Solución de Problemas

### Error: "No se encontraron imágenes"
- Verificar que las fotos estén en `./data/mi_cara/`
- Verificar formatos (.jpg, .png)
- Verificar permisos de lectura

### Error: CUDA out of memory
- Reducir batch_size a 1
- Reducir resolución si es necesario
- Cerrar otras aplicaciones que usen GPU

### LoRA no se carga
- Verificar que el archivo existe: `ls -la loras/mi_cara.safetensors`
- Reiniciar ButterVision
- Revisar logs en terminal

### Calidad baja del LoRA
- Más fotos (mínimo 20)
- Más variedad en dataset
- Más pasos de entrenamiento (2000+)
- Ajustar learning rate

## 📊 Monitoreo del Entrenamiento

El script muestra:
- Pérdida (loss) - debe disminuir
- Progreso de pasos
- Checkpoints guardados cada 500 pasos

## 🔄 Re-entrenamiento

Para mejorar el LoRA:
1. Añadir más fotos al dataset
2. Ajustar parámetros de entrenamiento
3. Re-ejecutar `python train_lora_mi_cara.py`

## 🎯 Consejos para Mejor Resultado

### Dataset
- Fotos de alta calidad
- Variedad de iluminación
- Expresiones naturales
- Ángulos variados

### Entrenamiento
- 1000-2000 pasos mínimo
- Batch size pequeño para estabilidad
- Learning rate 1e-4

### Generación
- CFG 7-8 para realismo
- Negative prompts específicos
- Testear diferentes seeds

## 📚 Recursos Adicionales

- [Diffusers LoRA Training Guide](https://huggingface.co/docs/diffusers/training/lora)
- [Stable Diffusion LoRA Concepts](https://stable-diffusion-art.com/lora/)
- [Dataset Preparation Tips](https://stable-diffusion-art.com/how-to-create-a-lora/)

## ❓ Preguntas Frecuentes

**¿Puedo usar fotos de otras personas?**
Sí, pero el LoRA aprenderá esa cara específica.

**¿Cuánto tarda el entrenamiento?**
30-60 minutos en GTX 1650 para 1000 pasos.

**¿El LoRA funciona con videos?**
Sí, se aplica automáticamente en el modo video.

**¿Puedo tener múltiples LoRAs?**
Sí, colócalos en `./loras/` con nombres diferentes.