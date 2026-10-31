"""Independent local SDXL backend; designed for offload, not validated on 4 GB."""
import os
from pathlib import Path

import psutil
import torch
from huggingface_hub import try_to_load_from_cache
from huggingface_hub.constants import HF_HUB_CACHE
from diffusers import StableDiffusionXLPipeline

import config
from core.engine_memory import lifecycle, measured_phase, release_owned
from core.prompt_limits import limit_prompts


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


def check_budget(model):
    path = Path(model)
    try:
        size = path.stat().st_size if path.is_file() else sum(p.stat().st_size for p in path.rglob("*.safetensors"))
    except OSError as error:
        raise ValueError("No se pueden leer los pesos SDXL locales.") from error
    if not size:
        raise ValueError("No se encontraron pesos safetensors locales para SDXL.")
    cuda = torch.cuda.is_available()
    minimum = size * (2 if cuda else 4) + 2 * 1024**3
    if available_ram() < minimum:
        raise ValueError("RAM disponible insuficiente para cargar SDXL con margen. Cierra otras aplicaciones o usa un modelo más pequeño.")
    if cuda:
        try:
            free = torch.cuda.mem_get_info()[0]
        except RuntimeError as error:
            raise ValueError("No se pudo medir la VRAM disponible para SDXL.") from error
        required = 1536 * 1024**2 if config.model_config.enable_cpu_offload else size + 2 * 1024**3
        if free < required:
            raise ValueError("VRAM libre insuficiente para iniciar SDXL con la configuración de offload actual.")


class SDXLPipeline:
    def __init__(self, model_id):
        self.model_id = model_id
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_pipeline(self):
        if self.pipe is not None:
            return self.pipe
        return self._build_pipeline()

    @lifecycle("sdxl", "load")
    def _build_pipeline(self):
        check_budget(self.model_id)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        source = Path(self.model_id)
        if source.is_file():
            pipe = StableDiffusionXLPipeline.from_single_file(str(source), config=local_sdxl_config(source), torch_dtype=dtype, local_files_only=True)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(str(source), torch_dtype=dtype, local_files_only=True, use_safetensors=True)
        self.pipe = pipe
        pipe.enable_attention_slicing("max")
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        if self.device == "cuda" and config.model_config.enable_cpu_offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(self.device)
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
