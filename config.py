"""
ButterVision - Configuración centralizada del proyecto
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Rutas base del proyecto
PROJECT_ROOT = Path(__file__).parent.absolute()
MODELS_DIR = PROJECT_ROOT / "models"
SD_MODELS_DIR = MODELS_DIR / "stable-diffusion"
LORA_DIR = MODELS_DIR / "lora"
CONTROLNET_DIR = MODELS_DIR / "controlnet"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / "cache"
EXTENSIONS_DIR = PROJECT_ROOT / "extensions"

# Crear directorios si no existen
for directory in [MODELS_DIR, SD_MODELS_DIR, LORA_DIR, CONTROLNET_DIR, EMBEDDINGS_DIR, 
                  OUTPUTS_DIR, CACHE_DIR, EXTENSIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Configuración del modelo Stable Diffusion"""
    # Modelo base por defecto. ButterVision lo descarga en el primer arranque si falta.
    model_id: str = "cyberrealistic_final"
    default_model_filename: str = "cyberrealistic_final.safetensors"
    default_model_url: str = "https://civitai.com/api/download/models/2681234?fileId=2567874"
    # También puedes usar un repo Hugging Face compatible, por ejemplo:
    # "runwayml/stable-diffusion-v1-5"
    
    # Optimizaciones de memoria
    use_fp16: bool = False  # DESACTIVADO para GTX 1650 (evita type mismatches)
    enable_xformers: bool = False  # xformers DESACTIVADO para GTX 1650 (causa NaNs)
    enable_attention_slicing: bool = True  # Divide attention en chunks
    enable_vae_slicing: bool = True  # Procesa VAE en batches
    enable_cpu_offload: bool = False  # Offload a CPU (más lento pero ahorra VRAM)
    
    # Parámetros por defecto de generación
    default_steps: int = 30
    default_cfg_scale: float = 5.0  # Reducido para estabilidad en GTX 1650
    default_width: int = 512
    default_height: int = 512
    default_scheduler: str = "DDIMScheduler"  # Más estable en GTX 1650
    
    # Seguridad
    safety_checker: bool = False  # Desactivado por defecto (ahorra VRAM)
    
    # Cache
    cache_dir: Path = CACHE_DIR


@dataclass
class ServerConfig:
    """Configuración del servidor web"""
    host: str = "0.0.0.0"
    port: int = 7861
    share: bool = True  # Gradio share link
    auth: Optional[tuple] = None  # ("usuario", "contraseña") para autenticación
    queue_concurrency: int = 1  # Número de generaciones simultáneas


@dataclass
class UIConfig:
    """Configuración de la interfaz"""
    theme: str = "default"  # "default", "soft", "monochrome"
    allow_flagging: bool = False
    show_error: bool = True
    analytics_enabled: bool = False


# Instancias globales de configuración
model_config = ModelConfig()
server_config = ServerConfig()
ui_config = UIConfig()


def get_available_schedulers():
    """Lista de schedulers disponibles"""
    return [
        "DPMSolverMultistepScheduler",  # Recomendado: rápido y calidad
        "EulerDiscreteScheduler",
        "EulerAncestralDiscreteScheduler",
        "KDPM2DiscreteScheduler",
        "KDPM2AncestralDiscreteScheduler",
        "DDIMScheduler",
        "PNDMScheduler",
        "LMSDiscreteScheduler",
    ]


def update_model_config(**kwargs):
    """Actualiza la configuración del modelo dinámicamente"""
    global model_config
    for key, value in kwargs.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
