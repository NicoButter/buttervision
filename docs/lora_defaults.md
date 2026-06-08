# LoRA Directory - ButterVision

Este directorio es para almacenar tus LoRAs personalizados para entrenamiento.

## Cómo agregar un LoRA:

### Opción 1: Descargar desde la Interfaz Web
1. Ve a la pestaña **"Train LoRA"** en la aplicación
2. En la sección **"📥 Descargar LoRA Existente"**
3. Pega la URL directa del archivo `.safetensors`
4. Haz clic en **"📥 Descargar LoRA"**
5. El LoRA se descargará automáticamente y se habilitará

### Opción 2: Descarga Manual
1. **Descarga un LoRA** desde:
   - HuggingFace (recomendado): Busca repos con "LoRA" o "lora-weights"
   - Civitai: Modelos marcados como "LoRA"

2. **Archivo requerido**:
   - Nombre: `lcm_lora.safetensors` (este nombre específico)
   - Formato: `.safetensors` (obligatorio)
   - Tamaño: Generalmente 10MB - 200MB

3. **Coloca el archivo** aquí: `models/lora/defaults/lcm_lora.safetensors`

4. **Habilita en el código**:
   - Edita `core/pipeline.py`
   - Cambia `self.detail_enhancer_enabled = True`

## Notas importantes:

- Solo se usa para **entrenamiento**, no para generación básica
- El archivo debe ser compatible con Stable Diffusion 1.5
- Si no tienes LoRA, la aplicación funciona normalmente sin él

## Ejemplos de LoRAs:

- LCM LoRA: https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors
- Detail Enhancer: Busca "add_detail" en HuggingFace
- Estilo específico: Busca según tu necesidad

¡El directorio está listo para tu LoRA!