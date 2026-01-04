"""
Ejemplo de script de prueba para ButterVision
Genera una imagen de prueba sin la UI
"""
import sys
from pathlib import Path

# Añadir raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from core import StableDiffusionManager
import config

def main():
    print("🎨 ButterVision - Test Script")
    print("="*50)
    
    # Configurar para test (modelo pequeño)
    config.model_config.model_id = "runwayml/stable-diffusion-v1-5"
    config.model_config.use_fp16 = True
    config.model_config.enable_xformers = True
    
    # Crear manager
    print("\n📦 Cargando modelo...")
    sd_manager = StableDiffusionManager()
    
    # Verificar CUDA
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️  CPU mode (muy lento)")
    
    # Prompt de prueba
    prompt = "a beautiful landscape with mountains and a lake, sunset, highly detailed, 8k"
    negative = "blurry, low quality, bad anatomy"
    
    print(f"\n🎨 Generando imagen de prueba...")
    print(f"Prompt: {prompt}")
    
    try:
        # Generar
        images = sd_manager.generate_txt2img(
            prompt=prompt,
            negative_prompt=negative,
            steps=20,
            cfg_scale=7.5,
            width=512,
            height=512,
            seed=42,
            num_images=1,
        )
        
        # Guardar
        output_path = config.OUTPUTS_DIR / "test_image.png"
        images[0].save(output_path)
        
        print(f"\n✅ ¡Imagen generada exitosamente!")
        print(f"📁 Guardada en: {output_path}")
        
        # Limpiar
        sd_manager.unload_pipeline("all")
        print("\n♻️  Pipeline descargado")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
