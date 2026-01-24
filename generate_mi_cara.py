#!/usr/bin/env python3
"""
Script de ejemplo para generar imágenes con LoRA de cara personal
Usa el pipeline avanzado de ButterVision
"""

import torch
from core.advanced_pipeline import ButterVisionPipeline
import config

def main():
    """Generar imágenes de ejemplo con cara personal"""

    print("🎨 Generando imágenes con LoRA de cara personal...")

    # Inicializar pipeline
    pipeline = ButterVisionPipeline(
        model_id="runwayml/stable-diffusion-v1-5",
        enable_optimizations=True,
        enable_lcm=False  # Usar generación normal para mejor calidad
    )

    # Cargar pipeline SD
    pipeline.load_sd_pipeline()

    # Verificar que LoRA esté cargado
    if "mi_cara" in pipeline.loaded_loras:
        print("✅ LoRA 'mi_cara' cargado correctamente")
    else:
        print("⚠️ LoRA 'mi_cara' no encontrado. Se generarán imágenes sin LoRA.")

    # Prompts de ejemplo
    prompts = [
        "foto de una persona, cara realista, sonrisa amable, fondo neutro, iluminación natural, alta calidad",
        "retrato profesional de una persona, expresión seria, fondo blanco, iluminación de estudio",
        "una persona en un parque, sonrisa, luz del sol, fondo natural",
        "foto casual de una persona con gafas, expresión relajada, fondo urbano",
    ]

    negative_prompt = "blur, low quality, distorted face, ugly, deformed, extra limbs, bad anatomy"

    # Generar imágenes
    for i, prompt in enumerate(prompts, 1):
        print(f"\n🖼️ Generando imagen {i}/{len(prompts)}...")

        try:
            # Generar imagen
            images = pipeline.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                width=512,
                height=512,
                num_images_per_prompt=1,
                seed=42 + i  # Seed diferente para cada imagen
            )

            # Guardar imagen
            output_path = f"outputs/mi_cara_ejemplo_{i}.png"
            images[0].save(output_path)
            print(f"✅ Imagen guardada: {output_path}")

        except Exception as e:
            print(f"❌ Error generando imagen {i}: {e}")

    print("\n🎉 ¡Generación completada!")
    print("Revisa las imágenes en outputs/")

if __name__ == "__main__":
    main()