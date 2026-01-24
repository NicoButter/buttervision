"""
Core Pipeline - StableDiffusionManager
Gestiona la carga y ejecución del pipeline de Stable Diffusion con optimizaciones low-VRAM
"""
import gc
import torch
from pathlib import Path
from typing import Optional, List, Union
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
)
from diffusers.utils import logging
import config


# Configurar logging
logger = logging.get_logger(__name__)


class StableDiffusionManager:
    """
    Administrador principal del pipeline de Stable Diffusion
    Optimizado para GPUs con baja VRAM (4GB+)
    """
    
    def __init__(
        self,
        model_id: str = None,
        device: str = "cuda",
        enable_optimizations: bool = True
    ):
        """
        Inicializa el manager
        
        Args:
            model_id: ID del modelo de HuggingFace (default: desde config)
            device: 'cuda' o 'cpu'
            enable_optimizations: Activar optimizaciones de memoria
        """
        self.model_id = model_id or config.model_config.model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.enable_optimizations = enable_optimizations
        
        # Pipelines (se cargan bajo demanda)
        self.txt2img_pipe: Optional[StableDiffusionPipeline] = None
        self.img2img_pipe: Optional[StableDiffusionImg2ImgPipeline] = None
        self.inpaint_pipe: Optional[StableDiffusionInpaintPipeline] = None
        
        # Estado
        self.current_scheduler = config.model_config.default_scheduler
        self.loaded_loras = {}  # {nombre: peso}
        self.detail_enhancer_enabled = False  # LoRA deshabilitado - el usuario debe descargarlo manualmente para entrenamiento
        self.detail_enhancer_weight = 0.6   # Peso por defecto
        
        print(f"🎨 ButterVision iniciando...")
        print(f"📦 Modelo: {self.model_id}")
        print(f"🔧 Dispositivo: {self.device}")
        print(f"⚡ Optimizaciones: {'Activadas' if enable_optimizations else 'Desactivadas'}")
        
        # Precargar modelo y LoRA
        self.preload()
    
    def preload(self):
        """
        Precarga el modelo base y el LoRA de mejora de detalles
        para que estén disponibles inmediatamente
        """
        print("📥 Precargando recursos...")
        
        try:
            # 1. Verificar/descargar modelo base
            print("   Verificando modelo base...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                self.model_id,
                cache_dir=str(config.model_config.cache_dir),
                local_files_only=False,
                resume_download=True
            )
            print("   ✅ Modelo base disponible")
            
        except Exception as e:
            print(f"   ⚠️  Error al verificar modelo: {e}")
            print("   Continuando (se descargará cuando sea necesario)")
        
        # 2. Verificar LoRA de detalles (opcional para entrenamiento)
        if self.detail_enhancer_enabled:
            print("   Verificando LoRA de mejora de detalles...")
            detail_lora_path = config.LORA_DIR / "defaults" / "lcm_lora.safetensors"
            if not detail_lora_path.exists():
                print("   Descargando LoRA...")
                if self._download_detail_enhancer_lora():
                    print("   ✅ LoRA disponible")
                else:
                    print("   ⚠️  No se pudo descargar LoRA, deshabilitando...")
                    self.detail_enhancer_enabled = False
            else:
                print("   ✅ LoRA ya disponible")
        else:
            print("   ℹ️  LoRA deshabilitado (consulta models/lora/defaults/README.md para uso opcional)")
        
        print("✅ Precarga completada")
    
    def _apply_optimizations(self, pipe):
        """Aplica optimizaciones de memoria al pipeline"""
        if not self.enable_optimizations:
            return pipe
        
        # 1. Usar float16 (ahorra ~50% VRAM)
        if config.model_config.use_fp16 and self.device == "cuda":
            pipe = pipe.to(torch_dtype=torch.float16)
        
        # 2. xformers memory efficient attention (requiere xformers instalado)
        if config.model_config.enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("✅ xformers activado")
            except Exception as e:
                print(f"⚠️  xformers no disponible: {e}")
        
        # 3. Attention slicing (divide attention en chunks)
        if config.model_config.enable_attention_slicing:
            pipe.enable_attention_slicing(slice_size="auto")
            print("✅ Attention slicing activado")
        
        # 4. VAE slicing (procesa VAE en batches pequeños)
        if config.model_config.enable_vae_slicing:
            pipe.enable_vae_slicing()
            print("✅ VAE slicing activado")
        
        # 5. CPU offload (para VRAM extremadamente baja < 4GB)
        if config.model_config.enable_cpu_offload and self.device == "cuda":
            pipe.enable_sequential_cpu_offload()
            print("✅ CPU offload activado (secuencial)")
        
        return pipe
    
    def load_txt2img_pipeline(self):
        """Carga el pipeline de Text-to-Image"""
        if self.txt2img_pipe is not None:
            print("♻️  Pipeline txt2img ya cargado")
            return self.txt2img_pipe
        
        print("📥 Cargando pipeline txt2img...")
        
        # Cargar pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if config.model_config.use_fp16 else torch.float32,
            safety_checker=None if not config.model_config.safety_checker else "default",
            cache_dir=str(config.model_config.cache_dir),
        )
        
        # Aplicar optimizaciones
        pipe = self._apply_optimizations(pipe)
        pipe = pipe.to(self.device)
        
        # Configurar scheduler por defecto
        pipe.scheduler = self._get_scheduler(self.current_scheduler, pipe.scheduler.config)
        
        self.txt2img_pipe = pipe
        print("✅ Pipeline txt2img listo")
        
        # Aplicar LoRA de mejora de detalles si está habilitado
        if self.detail_enhancer_enabled:
            self._apply_detail_enhancer_lora(pipe)
        
        return pipe
    
    def _apply_detail_enhancer_lora(self, pipe):
        """
        Aplica el LoRA de mejora de detalles al pipeline
        
        Args:
            pipe: Pipeline de Stable Diffusion
        """
        try:
            detail_lora_path = config.LORA_DIR / "defaults" / "lcm_lora.safetensors"
            if not detail_lora_path.exists():
                print("📥 Descargando LoRA de mejora de detalles...")
                if not self._download_detail_enhancer_lora():
                    print("⚠️  No se pudo descargar el LoRA de detalles, omitiendo...")
                    return
            
            print(f"🎨 Aplicando LoRA de mejora de detalles (peso: {self.detail_enhancer_weight})")
            
            # Cargar LoRA usando el método de diffusers
            pipe.load_lora_weights(
                str(detail_lora_path.parent),
                weight_name=detail_lora_path.name,
                adapter_name="detail_enhancer",
            )
            
            # Configurar peso del adaptador
            pipe.set_adapters(["detail_enhancer"], adapter_weights=[self.detail_enhancer_weight])
            
            # Fusionar LoRA para mejor rendimiento (reduce VRAM)
            pipe.fuse_lora()
            
            print("✅ LoRA de mejora de detalles aplicado")
            
        except Exception as e:
            print(f"⚠️  Error al aplicar LoRA de detalles: {e}")
            print("   Continuando sin LoRA de mejora de detalles...")
    
    def _download_detail_enhancer_lora(self):
        """
        Descarga del LoRA deshabilitada - el usuario debe hacerlo manualmente
        """
        print("   ℹ️  Descarga automática deshabilitada. Descarga manualmente si necesitas LoRA para entrenamiento.")
        return False
    
    def set_detail_enhancer(self, enabled: bool, weight: float = 0.6):
        """
        Configura el LoRA de mejora de detalles
        
        Args:
            enabled: Si activar el LoRA
            weight: Peso del LoRA (0.0 a 1.0)
        """
        self.detail_enhancer_enabled = enabled
        self.detail_enhancer_weight = weight
        
        # Si hay pipelines cargados, recargarlos con la nueva configuración
        if self.txt2img_pipe is not None:
            print("🔄 Recargando pipeline txt2img con nueva configuración de LoRA...")
            self.unload_pipeline("txt2img")
            self.load_txt2img_pipeline()
        
        if self.img2img_pipe is not None:
            print("🔄 Recargando pipeline img2img con nueva configuración de LoRA...")
            self.unload_pipeline("img2img")
            self.load_img2img_pipeline()
    
    def load_img2img_pipeline(self):
        """Carga el pipeline de Image-to-Image"""
        if self.img2img_pipe is not None:
            return self.img2img_pipe
        
        print("📥 Cargando pipeline img2img...")
        
        # Reutilizar componentes del txt2img si existe
        if self.txt2img_pipe is not None:
            pipe = StableDiffusionImg2ImgPipeline(
                vae=self.txt2img_pipe.vae,
                text_encoder=self.txt2img_pipe.text_encoder,
                tokenizer=self.txt2img_pipe.tokenizer,
                unet=self.txt2img_pipe.unet,
                scheduler=self.txt2img_pipe.scheduler,
                safety_checker=None,
                feature_extractor=None,
            )
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if config.model_config.use_fp16 else torch.float32,
                safety_checker=None,
                cache_dir=str(config.model_config.cache_dir),
            )
            pipe = self._apply_optimizations(pipe)
        
        pipe = pipe.to(self.device)
        self.img2img_pipe = pipe
        print("✅ Pipeline img2img listo")
        
        # Aplicar LoRA de mejora de detalles si está habilitado
        if self.detail_enhancer_enabled:
            self._apply_detail_enhancer_lora(pipe)
        
        return pipe
    
    def load_inpaint_pipeline(self):
        """Carga el pipeline de Inpainting"""
        if self.inpaint_pipe is not None:
            return self.inpaint_pipe
        
        print("📥 Cargando pipeline inpaint...")
        
        # Para inpainting, necesitas un modelo específico entrenado para ello
        # o usar un modelo base con adaptaciones
        inpaint_model = self.model_id.replace("v1-5", "inpainting")
        
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                inpaint_model,
                torch_dtype=torch.float16 if config.model_config.use_fp16 else torch.float32,
                safety_checker=None,
                cache_dir=str(config.model_config.cache_dir),
            )
        except:
            # Fallback: crear desde componentes del txt2img
            if self.txt2img_pipe is None:
                self.load_txt2img_pipeline()
            
            pipe = StableDiffusionInpaintPipeline(
                vae=self.txt2img_pipe.vae,
                text_encoder=self.txt2img_pipe.text_encoder,
                tokenizer=self.txt2img_pipe.tokenizer,
                unet=self.txt2img_pipe.unet,
                scheduler=self.txt2img_pipe.scheduler,
                safety_checker=None,
                feature_extractor=None,
            )
        
        pipe = self._apply_optimizations(pipe)
        pipe = pipe.to(self.device)
        self.inpaint_pipe = pipe
        print("✅ Pipeline inpaint listo")
        return pipe
    
    def _get_scheduler(self, scheduler_name: str, config_base):
        """Obtiene una instancia del scheduler especificado"""
        schedulers = {
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
            "EulerDiscreteScheduler": EulerDiscreteScheduler,
            "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
            "DDIMScheduler": DDIMScheduler,
            "PNDMScheduler": PNDMScheduler,
            "LMSDiscreteScheduler": LMSDiscreteScheduler,
            "KDPM2DiscreteScheduler": KDPM2DiscreteScheduler,
            "KDPM2AncestralDiscreteScheduler": KDPM2AncestralDiscreteScheduler,
        }
        
        scheduler_class = schedulers.get(scheduler_name, DPMSolverMultistepScheduler)
        return scheduler_class.from_config(config_base)
    
    def change_scheduler(self, scheduler_name: str):
        """Cambia el scheduler del pipeline activo"""
        self.current_scheduler = scheduler_name
        
        if self.txt2img_pipe:
            self.txt2img_pipe.scheduler = self._get_scheduler(
                scheduler_name, 
                self.txt2img_pipe.scheduler.config
            )
        
        if self.img2img_pipe:
            self.img2img_pipe.scheduler = self._get_scheduler(
                scheduler_name,
                self.img2img_pipe.scheduler.config
            )
        
        print(f"🔄 Scheduler cambiado a: {scheduler_name}")
    
    def generate_txt2img(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 30,
        cfg_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int = -1,
        num_images: int = 1,
    ) -> List[Image.Image]:
        """
        Genera imágenes desde texto
        
        Args:
            prompt: Descripción de la imagen
            negative_prompt: Lo que NO quieres en la imagen
            steps: Número de pasos de denoising (20-100)
            cfg_scale: Classifier Free Guidance (1-20, típico 7-9)
            width/height: Dimensiones (múltiplos de 8)
            seed: Semilla aleatoria (-1 = random)
            num_images: Cantidad de imágenes a generar
        
        Returns:
            Lista de imágenes PIL
        """
        pipe = self.load_txt2img_pipeline()
        
        # Configurar semilla
        generator = None
        if seed != -1:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generar
        print(f"🎨 Generando {num_images} imagen(es)...")
        
        def progress_callback(step, timestep, latents):
            print(f"   Paso {step+1}/{steps}")
        
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            width=width,
            height=height,
            num_images_per_prompt=num_images,
            generator=generator,
            callback=progress_callback,
            callback_steps=1,
        )
        
        return result.images
    
    def generate_img2img(
        self,
        init_image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 30,
        cfg_scale: float = 7.5,
        strength: float = 0.75,
        seed: int = -1,
    ) -> List[Image.Image]:
        """
        Genera imágenes desde una imagen inicial
        
        Args:
            init_image: Imagen de entrada
            strength: Cuánto transformar la imagen (0.0=sin cambio, 1.0=cambio total)
            ... (resto similar a txt2img)
        """
        pipe = self.load_img2img_pipeline()
        
        generator = None
        if seed != -1:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        print(f"🖼️  Transformando imagen...")
        
        def progress_callback(step, timestep, latents):
            print(f"   Paso {step+1}/{steps}")
        
        result = pipe(
            prompt=prompt,
            image=init_image,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            strength=strength,
            generator=generator,
            callback=progress_callback,
            callback_steps=1,
        )
        
        return result.images
    
    def change_model(self, new_model_id: str):
        """
        Cambia el modelo base y recarga los pipelines
        
        Args:
            new_model_id: ID del modelo de HF o ruta local
        """
        if new_model_id == self.model_id:
            print("ℹ️  El modelo ya está cargado")
            return
        
        print(f"🔄 Cambiando modelo de '{self.model_id}' a '{new_model_id}'...")
        
        # Descargar pipelines actuales
        self.unload_pipeline("all")
        
        # Actualizar modelo
        self.model_id = new_model_id
        
        print("✅ Modelo cambiado. Los pipelines se recargarán automáticamente en la próxima generación.")
    
    def unload_pipeline(self, pipeline_type: str = "all"):
        """Libera memoria descargando pipelines"""
        if pipeline_type in ["txt2img", "all"] and self.txt2img_pipe:
            del self.txt2img_pipe
            self.txt2img_pipe = None
            print("♻️  Pipeline txt2img descargado")
        
        if pipeline_type in ["img2img", "all"] and self.img2img_pipe:
            del self.img2img_pipe
            self.img2img_pipe = None
            print("♻️  Pipeline img2img descargado")
        
        if pipeline_type in ["inpaint", "all"] and self.inpaint_pipe:
            del self.inpaint_pipe
            self.inpaint_pipe = None
            print("♻️  Pipeline inpaint descargado")
        
        # Forzar limpieza de VRAM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 VRAM limpiada")
