"""Independent local SDXL backend; designed for offload, not validated on 4 GB."""
import os
from pathlib import Path

import psutil
import torch
from huggingface_hub import try_to_load_from_cache
from huggingface_hub.constants import HF_HUB_CACHE
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import is_accelerate_available

import config
from core.engine_memory import lifecycle, measured_phase, release_owned
from core.prompt_limits import limit_prompts

_GIB = 1024**3
_LOW_CPU_MEM_SAFETY_BYTES = 2 * _GIB
_SEQUENTIAL_OFFLOAD_MIN_VRAM_BYTES = 1536 * 1024**2


def local_sdxl_config(checkpoint):
    path = Path(checkpoint)
    candidates = [Path(os.environ["BUTTERVISION_SDXL_CONFIG"]).expanduser()] if os.environ.get("BUTTERVISION_SDXL_CONFIG") else []
    candidates += [path.with_suffix(""), path.parent]
    for cache in dict.fromkeys((str(config.CACHE_DIR), HF_HUB_CACHE)):
        index = try_to_load_from_cache("stabilityai/stable-diffusion-xl-base-1.0", "model_index.json", cache_dir=cache)
        if isinstance(index, str):
            candidates.append(Path(index).parent)
    from core.model_catalog import checkpoint_family, SDXL
    for candidate in candidates:
        if checkpoint_family(candidate) == SDXL and all((candidate / name).is_file() for name in (
            "scheduler/scheduler_config.json", "vae/config.json", "tokenizer/tokenizer_config.json", "text_encoder/config.json",
            "tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer_2/vocab.json", "tokenizer_2/merges.txt",
        )):
            return str(candidate.resolve())
    raise ValueError("Falta una configuración SDXL completa y local; configura BUTTERVISION_SDXL_CONFIG. No se descargará automáticamente.")


def available_ram():
    available = psutil.virtual_memory().available
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        used = int(Path("/sys/fs/cgroup/memory.current").read_text())
        if limit != "max":
            available = min(available, max(0, int(limit) - used))
    except (OSError, ValueError):
        pass
    return available


def _weight_bytes(model):
    path = Path(model)
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(item.stat().st_size for item in path.rglob("*.safetensors") if item.is_file())
    except OSError as error:
        raise ValueError("No se pueden leer los pesos SDXL locales.") from error


def inspect_load_budget(model):
    size = _weight_bytes(model)
    if not size:
        raise ValueError("No se encontraron pesos safetensors locales para SDXL.")
    accelerate = bool(is_accelerate_available())
    ram = available_ram()
    cuda = torch.cuda.is_available()
    # The current CPU fallback uses FP32, which can roughly double an FP16
    # checkpoint's in-memory parameter size even with low-memory initialization.
    ram_weight_multiplier = 1 if cuda else 2
    required_ram = size * ram_weight_multiplier + _LOW_CPU_MEM_SAFETY_BYTES
    free_vram = None
    required_vram = None
    if cuda:
        try:
            free_vram = torch.cuda.mem_get_info()[0]
        except Exception:
            free_vram = None
        required_vram = (_SEQUENTIAL_OFFLOAD_MIN_VRAM_BYTES if config.model_config.enable_cpu_offload
                         else size + _LOW_CPU_MEM_SAFETY_BYTES)
    reasons = []
    if not accelerate:
        reasons.append("Accelerate no está instalado; la carga de bajo uso de RAM no está disponible")
    if ram < required_ram:
        reasons.append("RAM libre insuficiente para carga progresiva de pesos con margen")
    if cuda and free_vram is None:
        reasons.append("No se pudo medir la VRAM disponible para SDXL")
    elif cuda and free_vram < required_vram:
        reasons.append("VRAM libre insuficiente para la configuración de offload actual")
    return {
        "strategy": "diffusers_low_cpu_mem_usage_sequential_cpu_offload" if cuda and config.model_config.enable_cpu_offload
                   else "diffusers_low_cpu_mem_usage",
        "checkpoint_bytes": size,
        "available_ram_bytes": ram,
        "estimated_required_ram_bytes": required_ram,
        "ram_weight_multiplier": ram_weight_multiplier,
        "ram_safety_margin_bytes": _LOW_CPU_MEM_SAFETY_BYTES,
        "low_cpu_mem_usage_available": accelerate,
        "cuda_available": cuda,
        "cpu_offload_enabled": bool(cuda and config.model_config.enable_cpu_offload),
        "available_vram_bytes": free_vram,
        "required_vram_bytes": required_vram,
        "admitted": not reasons,
        "reasons": reasons,
    }


def check_budget(model):
    assessment = inspect_load_budget(model)
    if not assessment["admitted"]:
        raise ValueError("; ".join(assessment["reasons"]))
    return assessment


class SDXLPipeline:
    def __init__(self, model_id):
        self.model_id = model_id
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_state = "selected"
        self.load_error = ""

    def load_pipeline(self):
        if self.pipe is not None:
            self.load_state = "loaded"
            return self.pipe
        self.load_state = "loading"
        self.load_error = ""
        try:
            return self._build_pipeline()
        except Exception as error:
            self.load_state = "error"
            self.load_error = type(error).__name__
            if self.pipe is not None:
                try:
                    self.cleanup()
                except Exception:
                    # Keep the load failure as the primary recoverable error.
                    pass
            raise

    @lifecycle("sdxl", "load")
    def _build_pipeline(self):
        check_budget(self.model_id)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        source = Path(self.model_id)
        if source.is_file():
            pipe = StableDiffusionXLPipeline.from_single_file(
                str(source), config=local_sdxl_config(source), torch_dtype=dtype,
                local_files_only=True, low_cpu_mem_usage=True,
            )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                str(source), torch_dtype=dtype, local_files_only=True,
                use_safetensors=True, low_cpu_mem_usage=True,
            )
        self.pipe = pipe
        pipe.enable_attention_slicing("max")
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        if self.device == "cuda" and config.model_config.enable_cpu_offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(self.device)
        self.load_state = "loaded"
        self.load_error = ""
        return pipe

    def generate_image(self, prompt, negative_prompt="", seed=-1, num_images=1, prompt_warning_callback=None, **kwargs):
        pipe = self.load_pipeline()
        # Each SDXL text encoder tokenizes the same prompt independently.
        first_prompt, first_negative = limit_prompts(pipe, prompt, negative_prompt, prompt_warning_callback)
        second = type("EncoderView", (), {"tokenizer": pipe.tokenizer_2, "text_encoder": pipe.text_encoder_2})()
        second_prompt, second_negative = limit_prompts(second, prompt, negative_prompt, prompt_warning_callback)
        generator = torch.Generator(device="cpu").manual_seed(seed) if seed != -1 else None
        with measured_phase("inference_and_decoding"):
            result = pipe(prompt=first_prompt, prompt_2=second_prompt, negative_prompt=first_negative,
                          negative_prompt_2=second_negative, generator=generator, num_images_per_prompt=num_images, **kwargs)
        return result.images

    @lifecycle("sdxl", "release")
    def cleanup(self):
        release_owned(self, ("pipe",))
        if self.load_state != "error":
            self.load_state = "selected"
