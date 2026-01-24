"""
ButterVision Advanced Pipeline
Pipeline completo optimizado para GTX 1650 (4GB VRAM)
Soporta: Imágenes + Videos + LoRA personalizado + AnimateDiff

Diagrama lógico del pipeline:

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Modelo Base   │ -> │   Checkpoint     │ -> │   LoRA Fusion   │
│   SD 1.5        │    │   Realistic V5.1 │    │   Cara Personal │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │     Optimizaciones VRAM   │
                    │  - fp16, xformers        │
                    │  - attention slicing     │
                    │  - VAE slicing           │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │   Generación Dual        │
                    │  ┌─────────────┐         │
                    │  │  Imágenes   │         │
                    │  │  SD Pipeline│         │
                    │  └─────────────┘         │
                    │                          │
                    │  ┌─────────────┐         │
                    │  │   Videos    │         │
                    │  │ AnimateDiff │         │
                    │  └─────────────┘         │
                    └──────────────────────────┘

Flujo de datos:
1. Carga modelo base SD 1.5
2. Aplica checkpoint realista (Realistic Vision v5.1)
3. Detecta LoRAs en models/lora/
4. Permite activar LoRA por nombre (ej: "n1c0 person")
5. Genera imágenes con SD Pipeline
6. Genera videos con AnimateDiff (8-16 frames)
7. Mantiene VRAM < 4GB con optimizaciones

Optimizaciones VRAM para GTX 1650:
- torch_dtype=torch.float16 (50% menos VRAM)
- enable_xformers_memory_efficient_attention (si disponible)
- enable_attention_slicing(slice_size="auto")
- enable_vae_slicing() para VAE
- enable_sequential_cpu_offload() como último recurso
"""

import torch
import gc
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    AnimateDiffPipeline,
    DDIMScheduler,
    MotionAdapter,
    LCMScheduler,
)
from diffusers.utils import logging
import config

# Configurar logging
logger = logging.get_logger(__name__)


class ButterVisionPipeline:
    """
    Pipeline avanzado de ButterVision optimizado para GTX 1650
    Soporta imágenes, videos y LoRA personalizado
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        enable_optimizations: bool = True,
        enable_lcm: bool = False
    ):
        """
        Inicializa el pipeline avanzado

        Args:
            model_id: ID del modelo base de HuggingFace
            checkpoint_path: Ruta al checkpoint realista (opcional)
            device: 'cuda' o 'cpu'
            enable_optimizations: Activar optimizaciones de VRAM
            enable_lcm: Usar LCM LoRA para generación más rápida
        """
        self.model_id = model_id
        self.checkpoint_path = checkpoint_path or "SG161222/Realistic_Vision_V5.1_noVAE"
        self.device = device if torch.cuda.is_available() else "cpu"
        self.enable_optimizations = enable_optimizations
        self.enable_lcm = enable_lcm

        # Pipelines (cargados bajo demanda)
        self.sd_pipeline: Optional[StableDiffusionPipeline] = None
        self.animatediff_pipeline: Optional[AnimateDiffPipeline] = None

        # Estado LoRA
        self.loaded_loras: Dict[str, float] = {}  # {nombre: peso}
        self.available_loras = self._scan_available_loras()

        # Configuración AnimateDiff
        self.motion_adapter_id = "guoyww/animatediff-motion-adapter-v1-5-2"

        print("🎨 ButterVision Advanced Pipeline inicializado")
        print(f"📦 Modelo base: {self.model_id}")
        print(f"🎯 Checkpoint: {self.checkpoint_path}")
        print(f"🔧 Dispositivo: {self.device}")
        print(f"⚡ Optimizaciones: {'Activadas' if enable_optimizations else 'Desactivadas'}")
        print(f"🚀 LCM: {'Activado' if enable_lcm else 'Desactivado'}")
        print(f"🎭 LoRAs encontrados: {len(self.available_loras)}")

    def _scan_available_loras(self) -> Dict[str, str]:
        """Escanea LoRAs disponibles en models/lora/ y ./loras/"""
        available_loras = {}

        # Escanear models/lora/
        lora_dir = Path("models/lora")
        if lora_dir.exists():
            for lora_file in lora_dir.glob("**/*.safetensors"):
                name = lora_file.stem
                available_loras[name] = str(lora_file)

        # Escanear ./loras/
        custom_lora_dir = Path("loras")
        if custom_lora_dir.exists():
            for lora_file in custom_lora_dir.glob("**/*.safetensors"):
                name = lora_file.stem
                available_loras[name] = str(lora_file)

        return available_loras

    def _apply_vram_optimizations(self, pipeline):
        """Aplica todas las optimizaciones de VRAM para GTX 1650"""
        if not self.enable_optimizations:
            return pipeline

        print("🔧 Aplicando optimizaciones VRAM...")

        # 1. Float16 para todo el pipeline (reduce ~50% VRAM) - SOLO si está activado en config
        if self.device == "cuda" and config.model_config.use_fp16:
            pipeline = pipeline.to(torch_dtype=torch.float16)
            print("✅ Float16 activado para pipeline completo")
        else:
            pipeline = pipeline.to(torch_dtype=torch.float32)
            print("✅ Float32 activado para pipeline completo (GTX 1650 compatible)")

        # 2. Forzar VAE a CPU + float32 (evita NaNs en GTX 1650)
        if self.device == "cuda":
            pipeline.vae.to("cpu")
            pipeline.vae.to(dtype=torch.float32)
            print("✅ VAE movido a CPU + float32 (evita NaNs en GTX)")
            print(f"   UNet dtype: {pipeline.unet.dtype} (GPU)")
            print(f"   VAE dtype: {pipeline.vae.dtype} (CPU)")

        # 3. xFormers (reduce VRAM significativamente) - SOLO si está activado en config
        if config.model_config.enable_xformers:
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                print("✅ xFormers activado")
            except Exception as e:
                print(f"⚠️ xFormers no disponible: {e}")
        else:
            print("ℹ️ xFormers desactivado (configurado para GTX 1650)")

        # 4. Attention slicing (divide atención en chunks)
        pipeline.enable_attention_slicing(slice_size="auto")
        print("✅ Attention slicing activado")

        # 5. VAE slicing (procesa VAE en batches pequeños)
        pipeline.enable_vae_slicing()
        print("✅ VAE slicing activado")

        # 6. CPU offload como último recurso (si VRAM muy baja)
        # Solo activar si es necesario
        # pipeline.enable_sequential_cpu_offload()

        return pipeline

    def load_sd_pipeline(self) -> StableDiffusionPipeline:
        """Carga el pipeline de Stable Diffusion con optimizaciones"""
        if self.sd_pipeline is not None:
            return self.sd_pipeline

        print("📥 Cargando pipeline SD...")

        # Cargar pipeline base
        torch_dtype = torch.float16 if (self.device == "cuda" and config.model_config.use_fp16) else torch.float32
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,  # Desactivar para más velocidad
            cache_dir=str(config.model_config.cache_dir),
        )

        # Aplicar checkpoint realista si es diferente del base
        if self.checkpoint_path != self.model_id:
            print(f"🎨 Aplicando checkpoint realista: {self.checkpoint_path}")
            # Para checkpoints de HuggingFace, usar load_lora o from_pretrained
            try:
                # Si es un modelo de HF, intentar cargarlo
                if "/" in self.checkpoint_path:
                    # Cargar como LoRA o modelo adicional
                    pipeline.load_lora_weights(self.checkpoint_path)
                    pipeline.fuse_lora()
                    print("✅ Checkpoint aplicado como LoRA")
                else:
                    print("ℹ️ Usando modelo base (sin checkpoint adicional)")
            except Exception as e:
                print(f"⚠️ No se pudo aplicar checkpoint: {e}")

        # Configurar scheduler - SIEMPRE DDIM para estabilidad en GTX 1650
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        print("✅ DDIM Scheduler forzado (estable para GTX 1650)")

        # Aplicar optimizaciones VRAM
        pipeline = self._apply_vram_optimizations(pipeline)
        pipeline = pipeline.to(self.device)

        # Cargar LCM LoRA si está habilitado
        if self.enable_lcm:
            try:
                lcm_path = Path("models/lora/defaults/lcm_lora.safetensors")
                if lcm_path.exists():
                    pipeline.load_lora_weights(str(lcm_path.parent), weight_name=lcm_path.name)
                    print("✅ LCM LoRA cargado")
                else:
                    print("⚠️ LCM LoRA no encontrado")
            except Exception as e:
                print(f"⚠️ Error cargando LCM LoRA: {e}")

        self.sd_pipeline = pipeline
        print("✅ Pipeline SD listo")

        # Cargar automáticamente LoRA de mi_cara si existe
        if "mi_cara" in self.available_loras:
            print("🎭 Cargando LoRA automático: mi_cara")
            self.load_lora("mi_cara", weight=0.8)

        return pipeline

    def load_animatediff_pipeline(self) -> AnimateDiffPipeline:
        """Carga el pipeline de AnimateDiff para videos"""
        if self.animatediff_pipeline is not None:
            return self.animatediff_pipeline

        print("🎬 Cargando pipeline AnimateDiff...")

        # Cargar motion adapter
        motion_adapter = MotionAdapter.from_pretrained(
            self.motion_adapter_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        # Crear pipeline con SD base + motion adapter
        pipeline = AnimateDiffPipeline.from_pretrained(
            self.model_id,
            motion_adapter=motion_adapter,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            cache_dir=str(config.model_config.cache_dir),
        )

        # Configurar scheduler para videos
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

        # Aplicar optimizaciones VRAM (más agresivas para videos)
        pipeline = self._apply_vram_optimizations(pipeline)
        pipeline = pipeline.to(self.device)

        self.animatediff_pipeline = pipeline
        print("✅ Pipeline AnimateDiff listo")
        return pipeline

    def load_lora(self, lora_name: str, weight: float = 1.0):
        """
        Carga un LoRA por nombre

        Args:
            lora_name: Nombre del LoRA (sin extensión)
            weight: Peso del LoRA (0.0 - 2.0)
        """
        if lora_name not in self.available_loras:
            print(f"❌ LoRA '{lora_name}' no encontrado")
            return False

        lora_path = self.available_loras[lora_name]

        try:
            # Recargar pipeline si es necesario
            if self.sd_pipeline is not None:
                self.sd_pipeline.load_lora_weights(
                    str(Path(lora_path).parent),
                    weight_name=Path(lora_path).name,
                    adapter_name=lora_name
                )
                self.sd_pipeline.set_adapters([lora_name], adapter_weights=[weight])
                print(f"✅ LoRA '{lora_name}' cargado con peso {weight}")
            else:
                print("⚠️ Pipeline SD no cargado, LoRA se aplicará en próxima carga")

            self.loaded_loras[lora_name] = weight
            return True

        except Exception as e:
            print(f"❌ Error cargando LoRA '{lora_name}': {e}")
            return False

    def unload_lora(self, lora_name: str):
        """Descarga un LoRA"""
        if lora_name in self.loaded_loras:
            try:
                if self.sd_pipeline is not None:
                    # Remover adapter
                    current_adapters = self.sd_pipeline.get_active_adapters()
                    if lora_name in current_adapters:
                        new_adapters = [a for a in current_adapters if a != lora_name]
                        self.sd_pipeline.set_adapters(new_adapters)

                del self.loaded_loras[lora_name]
                print(f"✅ LoRA '{lora_name}' descargado")
                return True
            except Exception as e:
                print(f"❌ Error descargando LoRA '{lora_name}': {e}")
                return False
        return False

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 5.0,  # REDUCIDO de 7.5 a 5.0 para estabilidad
        seed: int = -1,
        num_images: int = 1,
        lora_trigger: Optional[str] = None
    ) -> List[Image.Image]:
        """
        Genera imágenes con el pipeline optimizado

        Args:
            prompt: Descripción de la imagen
            negative_prompt: Lo que evitar
            width/height: Dimensiones (múltiplos de 8)
            num_inference_steps: Pasos de denoising
            guidance_scale: Fuerza del prompt
            seed: Semilla (-1 = random)
            num_images: Número de imágenes
            lora_trigger: Trigger word para activar LoRA (ej: "n1c0 person")
        """
        pipeline = self.load_sd_pipeline()

        # Configurar seed
        generator = None
        if seed != -1:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Modificar prompt si hay LoRA activado
        final_prompt = prompt
        if lora_trigger and self.loaded_loras:
            final_prompt = f"{lora_trigger}, {prompt}"
            print(f"🎭 Prompt con LoRA: {final_prompt}")

        print(f"🎨 Generando {num_images} imagen(es)...")

        # Generar
        result = pipeline(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        )

        # Limpiar VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"✅ Generadas {len(result.images)} imagen(es) PIL")
        return result.images

    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 16,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = -1,
        lora_trigger: Optional[str] = None
    ) -> List[Image.Image]:
        """
        Genera video corto con AnimateDiff

        Args:
            prompt: Descripción del video
            negative_prompt: Lo que evitar
            num_frames: Frames del video (8-16 recomendado)
            num_inference_steps: Pasos de denoising
            guidance_scale: Fuerza del prompt
            seed: Semilla (-1 = random)
            lora_trigger: Trigger word para LoRA
        """
        pipeline = self.load_animatediff_pipeline()

        # Configurar seed
        generator = None
        if seed != -1:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Modificar prompt si hay LoRA
        final_prompt = prompt
        if lora_trigger and self.loaded_loras:
            final_prompt = f"{lora_trigger}, {prompt}"
            print(f"🎭 Prompt con LoRA: {final_prompt}")

        print(f"🎬 Generando video de {num_frames} frames...")

        # Generar video
        result = pipeline(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        # Limpiar VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result.frames

    def get_vram_usage(self) -> str:
        """Retorna información de uso de VRAM"""
        if not torch.cuda.is_available():
            return "CPU mode - No VRAM info"

        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return ".2f"
        except:
            return "Unable to read VRAM"

    def cleanup(self):
        """Limpia pipelines y libera memoria"""
        if self.sd_pipeline:
            del self.sd_pipeline
            self.sd_pipeline = None

        if self.animatediff_pipeline:
            del self.animatediff_pipeline
            self.animatediff_pipeline = None

        self.loaded_loras.clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("🧹 Pipelines limpiados")


# Función de conveniencia para crear pipeline optimizado
def create_buttervision_pipeline(
    enable_lcm: bool = False,
    checkpoint: Optional[str] = None
) -> ButterVisionPipeline:
    """
    Crea un pipeline optimizado para GTX 1650

    Args:
        enable_lcm: Activar LCM para generación más rápida
        checkpoint: Checkpoint personalizado (opcional)

    Returns:
        ButterVisionPipeline configurado
    """
    return ButterVisionPipeline(
        model_id="runwayml/stable-diffusion-v1-5",
        checkpoint_path=checkpoint,
        device="cuda",
        enable_optimizations=True,
        enable_lcm=enable_lcm
    )


# Ejemplo de uso
if __name__ == "__main__":
    # Crear pipeline
    pipeline = create_buttervision_pipeline(enable_lcm=True)

    # Cargar LoRA de cara personal
    pipeline.load_lora("n1c0_person", weight=0.8)

    # Generar imagen
    images = pipeline.generate_image(
        prompt="foto de persona sonriendo",
        lora_trigger="n1c0 person",
        num_inference_steps=20
    )

    # Generar video
    video_frames = pipeline.generate_video(
        prompt="persona caminando en parque",
        lora_trigger="n1c0 person",
        num_frames=16
    )

    # Limpiar
    pipeline.cleanup()