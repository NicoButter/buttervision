#!/usr/bin/env python3
"""
Test simple con la solución aplicada: VAE en CPU
"""

import os
# Desactivar xformers completamente
os.environ["XFORMERS_DISABLED"] = "1"

def test_vae_cpu_solution():
    """Test con VAE en CPU (solución aplicada)"""
    print("🧪 Test con SOLUCIÓN APLICADA: VAE en CPU")
    print("=" * 50)

    try:
        import torch
        from diffusers import StableDiffusionPipeline, DDIMScheduler

        print("📥 Cargando pipeline con solución aplicada...")

        # Pipeline con la solución
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            local_files_only=True
        ).to("cuda")

        # SOLUCIÓN APLICADA: VAE en CPU
        pipe.vae.to("cpu")
        pipe.vae.to(dtype=torch.float32)

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_attention_slicing()

        print("✅ Configuración aplicada:")
        print(f"   UNet: {pipe.unet.dtype} en GPU")
        print(f"   VAE: {pipe.vae.dtype} en CPU")
        print("   Scheduler: DDIM")
        print("   xFormers: DESACTIVADO")

        print("\n🎨 Generando imagen de test...")
        result = pipe(
            "a red apple on a white table, studio lighting, photo",
            num_inference_steps=20,
            guidance_scale=5.0
        )

        image = result.images[0]
        print(f"✅ Imagen generada: {type(image)} - {image.size}")

        # Verificar calidad
        import numpy as np
        img_array = np.array(image)
        mean_val = img_array.mean()
        std_val = img_array.std()

        print(".2f")
        print(".2f")

        if mean_val > 50 and std_val > 20:
            print("🎉 ¡ÉXITO! La solución funciona")
            print("   ✅ Imagen válida generada")
            image.save("test_solution_success.png")
            return True
        else:
            print("❌ Imagen sigue negra/inválida")
            image.save("test_solution_fail.png")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'pipe' in locals():
            del pipe
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("🔧 Test de SOLUCIÓN APLICADA")
    print("VAE en CPU + xFormers desactivado")
    print("=" * 50)

    success = test_vae_cpu_solution()

    if success:
        print("\n🎯 ¡PROBLEMA RESUELTO!")
        print("Ahora ejecuta: python main.py")
        print("Las imágenes deberían aparecer correctamente en la UI")
    else:
        print("\n❌ La solución no funcionó")
        print("Necesario downgrade a PyTorch 2.1.2 + CUDA 11.8")