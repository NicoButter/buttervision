#!/usr/bin/env python3
"""
Pipeline MINIMAL que SÍ funciona en GTX 1650
Test definitivo para descartar problemas de infraestructura
"""

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image

def test_minimal_pipeline():
    """Test con pipeline minimal que debería funcionar"""
    print("🧪 Test MINIMAL de pipeline - GTX 1650")
    print("=" * 50)

    try:
        print("📥 Cargando pipeline minimal...")

        # Pipeline minimal como sugiere el usuario
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            local_files_only=True  # Usar solo archivos locales
        ).to("cuda")

        # VAE en float32
        pipe.vae.to(dtype=torch.float32)
        print("✅ VAE forzado a float32")

        # DDIM scheduler
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        print("✅ DDIM Scheduler")

        # Attention slicing
        pipe.enable_attention_slicing()
        print("✅ Attention slicing")

        # Verificar dtypes
        print(f"   UNet dtype: {pipe.unet.dtype}")
        print(f"   VAE dtype: {pipe.vae.dtype}")
        print(f"   Scheduler: {type(pipe.scheduler).__name__}")

        print("\n🎨 Generando imagen de test...")

        # Prompt absurdo pero estable
        result = pipe(
            "a red apple on a white table, studio lighting, photo",
            num_inference_steps=20,
            guidance_scale=5.0,
            height=512,
            width=512
        )

        image = result.images[0]
        print(f"✅ Imagen generada: {type(image)} - {image.size}")

        # Verificar que no es negra
        # Convertir a numpy para check
        import numpy as np
        img_array = np.array(image)
        mean_value = img_array.mean()
        std_value = img_array.std()

        print(f"   Media de píxeles: {mean_value:.2f}")
        print(f"   Std de píxeles: {std_value:.2f}")

        if mean_value < 10:  # Imagen muy oscura
            print("❌ IMAGEN MUY OSCURA - Posible problema de NaNs")
            return False
        elif std_value < 5:  # Sin variación
            print("❌ IMAGEN SIN VARIACIÓN - Posible uniforme/negra")
            return False
        else:
            print("✅ Imagen parece válida")

        # Guardar para verificación visual
        image.save("test_minimal.png")
        print("💾 Imagen guardada: test_minimal.png")

        # Limpiar
        del pipe
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ Error en pipeline minimal: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_buttervision():
    """Test con el pipeline de ButterVision"""
    print("\n🎨 Test con ButterVision Pipeline...")
    print("=" * 50)

    try:
        from core.advanced_pipeline import ButterVisionPipeline

        # Crear pipeline con configuración minimal
        pipeline = ButterVisionPipeline(
            model_id="runwayml/stable-diffusion-v1-5",
            enable_optimizations=True,
            enable_lcm=False  # Forzar DDIM
        )

        # Forzar local_files_only en el pipeline
        import config
        config.model_config.model_id = "runwayml/stable-diffusion-v1-5"  # Ya debería estar

        print("📥 Cargando pipeline ButterVision...")

        # Cargar pipeline
        pipe = pipeline.load_sd_pipeline()

        print("🎨 Generando con ButterVision...")

        # Usar mismo prompt
        images = pipeline.generate_image(
            prompt="a red apple on a white table, studio lighting, photo",
            num_inference_steps=20,
            guidance_scale=5.0,
            width=512,
            height=512,
            seed=42
        )

        image = images[0]
        print(f"✅ Imagen generada: {type(image)} - {image.size}")

        # Verificar
        import numpy as np
        img_array = np.array(image)
        mean_value = img_array.mean()
        std_value = img_array.std()

        print(f"   Media de píxeles: {mean_value:.2f}")
        print(f"   Std de píxeles: {std_value:.2f}")

        if mean_value < 10 or std_value < 5:
            print("❌ IMAGEN PROBLEMÁTICA")
            return False
        else:
            print("✅ Imagen válida")

        # Guardar
        image.save("test_buttervision.png")
        print("💾 Imagen guardada: test_buttervision.png")

        return True

    except Exception as e:
        print(f"❌ Error en ButterVision: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔬 DIAGNÓSTICO DEFINITIVO - GTX 1650")
    print("Si el pipeline minimal falla → Problema de infraestructura")
    print("Si ButterVision falla pero minimal funciona → Problema en nuestro código")
    print("=" * 60)

    # Test 1: Pipeline minimal
    minimal_ok = test_minimal_pipeline()

    # Test 2: ButterVision
    buttervision_ok = test_with_buttervision()

    print("\n" + "=" * 60)
    print("📊 RESULTADOS:")

    if minimal_ok and buttervision_ok:
        print("🎉 ¡AMBOS TESTS PASARON!")
        print("   ✅ Infraestructura OK")
        print("   ✅ ButterVision OK")
        print("   Las imágenes deberían ser visibles en la UI")
    elif minimal_ok and not buttervision_ok:
        print("⚠️ Pipeline minimal OK, ButterVision falla")
        print("   ✅ Infraestructura OK")
        print("   ❌ Problema en ButterVision")
        print("   Revisar configuración de optimizaciones")
    else:
        print("💥 PIPELINE MINIMAL FALLA")
        print("   ❌ Problema de infraestructura serio")
        print("   Revisar: drivers CUDA, PyTorch, diffusers")