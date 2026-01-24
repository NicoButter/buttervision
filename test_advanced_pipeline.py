#!/usr/bin/env python3
"""
Ejemplo de uso del ButterVision Advanced Pipeline
Prueba rápida del pipeline optimizado para GTX 1650
"""

import sys
from pathlib import Path

# Agregar raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from core import create_buttervision_pipeline
from datetime import datetime
import torch

def main():
    print("🎨 ButterVision Advanced Pipeline - Test")
    print("=" * 50)

    # Verificar GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ No GPU disponible")
        return

    # Crear pipeline
    print("\n📦 Inicializando pipeline...")
    pipeline = create_buttervision_pipeline(enable_lcm=True)

    # Mostrar LoRAs disponibles
    print(f"🎭 LoRAs encontrados: {list(pipeline.available_loras.keys())}")

    # Cargar LoRA si existe
    if "lcm_lora" in pipeline.available_loras:
        pipeline.load_lora("lcm_lora", weight=1.0)
        print("✅ LCM LoRA cargado")
    else:
        print("ℹ️ LCM LoRA no encontrado (opcional)")

    # Prompt de prueba
    prompt = "a professional portrait photo of a person, photorealistic, high detail, 8k"
    negative = "blurry, low quality, deformed, ugly"

    print(f"\n🎨 Generando imagen de prueba...")
    print(f"Prompt: {prompt}")

    try:
        # Generar imagen
        start_time = datetime.now()
        images = pipeline.generate_image(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=6 if pipeline.enable_lcm else 20,  # Menos steps con LCM
            guidance_scale=1.5 if pipeline.enable_lcm else 7.5,    # Menos guidance con LCM
            width=512,
            height=512,
            seed=42,
            num_images=1
        )
        end_time = datetime.now()

        # Calcular tiempo
        duration = (end_time - start_time).total_seconds()
        print(".1f")

        # Verificar VRAM
        vram_info = pipeline.get_vram_usage()
        print(f"💾 VRAM usado: {vram_info}")

        # Guardar imagen
        if images:
            output_dir = Path("outputs/test")
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/test_advanced_{timestamp}.png"
            images[0].save(filename)
            print(f"💾 Imagen guardada: {filename}")

            print("✅ ¡Generación exitosa!")
        else:
            print("❌ No se generaron imágenes")

    except Exception as e:
        print(f"❌ Error en generación: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Limpiar
        print("\n🧹 Limpiando pipeline...")
        pipeline.cleanup()
        print("✅ Pipeline limpiado")

if __name__ == "__main__":
    main()