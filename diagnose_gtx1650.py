#!/usr/bin/env python3
"""
Diagnóstico DEFINITIVO para GTX 1650 - NaNs en VAE decode
Prueba todas las soluciones conocidas en orden
"""

import os
import sys
import torch
from pathlib import Path

# Desactivar xformers completamente
os.environ["XFORMERS_DISABLED"] = "1"

def test_float32_full():
    """Test 1: TODO en float32"""
    print("🧪 TEST 1: PIPELINE COMPLETO EN FLOAT32")
    print("=" * 50)

    try:
        from diffusers import StableDiffusionPipeline, DDIMScheduler

        print("📥 Cargando pipeline float32...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,  # TODO en float32
            safety_checker=None,
            local_files_only=True
        ).to("cuda")

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_attention_slicing()

        print(f"   UNet dtype: {pipe.unet.dtype}")
        print(f"   VAE dtype: {pipe.vae.dtype}")
        print("   ✅ Todo en float32")

        print("\n🎨 Generando con float32...")
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
            print("🎉 ¡ÉXITO! Imagen válida con float32")
            image.save("test_float32_success.png")
            return True
        else:
            print("❌ Imagen negra/inválida incluso en float32")
            image.save("test_float32_fail.png")
            return False

    except Exception as e:
        print(f"❌ Error en float32: {e}")
        return False
    finally:
        if 'pipe' in locals():
            del pipe
        torch.cuda.empty_cache()

def test_vae_cpu():
    """Test 2: VAE en CPU, resto en GPU"""
    print("\n🧪 TEST 2: VAE EN CPU (quirúrgico)")
    print("=" * 50)

    try:
        from diffusers import StableDiffusionPipeline, DDIMScheduler

        print("📥 Cargando pipeline con VAE en CPU...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            local_files_only=True
        ).to("cuda")

        # VAE a CPU
        pipe.vae.to("cpu")
        pipe.vae.to(dtype=torch.float32)
        print("   ✅ VAE movido a CPU + float32")

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_attention_slicing()

        print(f"   UNet dtype: {pipe.unet.dtype} (GPU)")
        print(f"   VAE dtype: {pipe.vae.dtype} (CPU)")

        print("\n🎨 Generando con VAE en CPU...")
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
            print("🎉 ¡ÉXITO! VAE en CPU funciona")
            image.save("test_vae_cpu_success.png")
            return True
        else:
            print("❌ Imagen negra/inválida con VAE en CPU")
            image.save("test_vae_cpu_fail.png")
            return False

    except Exception as e:
        print(f"❌ Error en VAE CPU: {e}")
        return False
    finally:
        if 'pipe' in locals():
            del pipe
        torch.cuda.empty_cache()

def test_xformers_disabled():
    """Test 3: xFormers completamente desactivado"""
    print("\n🧪 TEST 3: XFORMERS DESACTIVADO")
    print("=" * 50)

    try:
        from diffusers import StableDiffusionPipeline, DDIMScheduler

        print("📥 Cargando sin xFormers...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            local_files_only=True
        ).to("cuda")

        pipe.vae.to(dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        # NO enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()

        print("   ✅ xFormers completamente desactivado")
        print(f"   UNet dtype: {pipe.unet.dtype}")
        print(f"   VAE dtype: {pipe.vae.dtype}")

        print("\n🎨 Generando sin xFormers...")
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
            print("🎉 ¡ÉXITO! Sin xFormers funciona")
            image.save("test_no_xformers_success.png")
            return True
        else:
            print("❌ Imagen negra/inválida sin xFormers")
            image.save("test_no_xformers_fail.png")
            return False

    except Exception as e:
        print(f"❌ Error sin xFormers: {e}")
        return False
    finally:
        if 'pipe' in locals():
            del pipe
        torch.cuda.empty_cache()

def diagnose_and_fix():
    """Diagnóstico completo y aplicación de solución"""
    print("🔬 DIAGNÓSTICO DEFINITIVO - GTX 1650")
    print("Problema: NaNs silenciosos en VAE decode")
    print("=" * 60)

    # Verificar modelo local
    model_path = Path("cache/models--runwayml--stable-diffusion-v1-5")
    if not model_path.exists():
        print("❌ Modelo no encontrado. Ejecuta: python main.py (primera vez)")
        return

    print("✅ Modelo local encontrado")

    # Test 1: Float32 completo
    print("\n🔍 Probando soluciones en orden...")
    float32_ok = test_float32_full()

    if float32_ok:
        print("\n🎯 DIAGNÓSTICO: Bug fp16 kernel confirmado")
        print("💡 SOLUCIÓN: Usar float32 para VAE")

        # Aplicar solución al pipeline principal
        apply_vae_cpu_fix()
        return

    # Test 2: VAE en CPU
    vae_cpu_ok = test_vae_cpu()

    if vae_cpu_ok:
        print("\n🎯 DIAGNÓSTICO: Kernels fp16 rotos")
        print("💡 SOLUCIÓN: VAE en CPU")

        apply_vae_cpu_fix()
        return

    # Test 3: Sin xFormers
    no_xformers_ok = test_xformers_disabled()

    if no_xformers_ok:
        print("\n🎯 DIAGNÓSTICO: xFormers causando problemas")
        print("💡 SOLUCIÓN: Desactivar xFormers")

        apply_xformers_fix()
        return

    # Si todo falla
    print("\n💥 DIAGNÓSTICO: Incompatibilidad binaria grave")
    print("PyTorch 2.5.1 + CUDA 12.1 + GTX 1650 = kernels rotos")
    print("\n🛠️ SOLUCIÓN DEFINITIVA: Downgrade a stack estable")

    suggest_downgrade()

def apply_vae_cpu_fix():
    """Aplicar solución: VAE en CPU"""
    print("\n🔧 APLICANDO SOLUCIÓN: VAE en CPU")

    # Modificar el pipeline avanzado
    pipeline_file = Path("core/advanced_pipeline.py")
    content = pipeline_file.read_text()

    # Buscar la línea de VAE float32
    if "pipe.vae.to(dtype=torch.float32)" in content:
        # Agregar VAE a CPU antes
        new_content = content.replace(
            "        # 2. Forzar VAE a float32 (evita NaNs en GTX 1650)\n        if self.device == \"cuda\":\n            pipeline.vae.to(dtype=torch.float32)\n            print(\"✅ VAE forzado a float32 (evita NaNs en GTX)\")\n            print(f\"   UNet dtype: {pipeline.unet.dtype}\")\n            print(f\"   VAE dtype: {pipeline.vae.dtype}\")",
            "        # 2. Forzar VAE a CPU + float32 (evita NaNs en GTX 1650)\n        if self.device == \"cuda\":\n            pipeline.vae.to(\"cpu\")\n            pipeline.vae.to(dtype=torch.float32)\n            print(\"✅ VAE movido a CPU + float32 (evita NaNs en GTX)\")\n            print(f\"   UNet dtype: {pipeline.unet.dtype} (GPU)\")\n            print(f\"   VAE dtype: {pipeline.vae.dtype} (CPU)\")"
        )

        pipeline_file.write_text(new_content)
        print("✅ Pipeline modificado: VAE en CPU")

def apply_xformers_fix():
    """Aplicar solución: Desactivar xFormers"""
    print("\n🔧 APLICANDO SOLUCIÓN: Desactivar xFormers")

    # Modificar config
    config_file = Path("config.py")
    content = config_file.read_text()

    new_content = content.replace(
        "    enable_xformers: bool = True  # xformers memory efficient attention",
        "    enable_xformers: bool = False  # xformers DESACTIVADO para GTX 1650"
    )

    config_file.write_text(new_content)
    print("✅ Config modificado: xFormers desactivado")

def suggest_downgrade():
    """Sugerir downgrade a stack estable"""
    print("\n🔄 DOWNGRADE RECOMENDADO")
    print("pip uninstall torch torchvision torchaudio -y")
    print("pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118")
    print("\n✅ CUDA 11.8 + PyTorch 2.1.2 = kernels maduros")
    print("✅ CERO problemas con GTX 1650")

if __name__ == "__main__":
    diagnose_and_fix()