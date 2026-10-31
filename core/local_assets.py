"""Resolve existing SD1.5 configuration without Diffusers' network fallback."""
from pathlib import Path

from huggingface_hub import try_to_load_from_cache
from huggingface_hub.constants import HF_HUB_CACHE

import config


def local_sd15_config(checkpoint):
    required = ("model_index.json", "scheduler/scheduler_config.json",
                "text_encoder/config.json", "tokenizer/tokenizer_config.json",
                "unet/config.json", "vae/config.json")
    candidates = [Path(checkpoint).with_suffix(""), Path(checkpoint).parent]
    for cache in dict.fromkeys((str(config.model_config.cache_dir), HF_HUB_CACHE)):
        for repo in ("stable-diffusion-v1-5/stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"):
            cached_index = try_to_load_from_cache(repo, "model_index.json", cache_dir=cache)
            if isinstance(cached_index, str):
                candidates.append(Path(cached_index).parent)
    for candidate in candidates:
        if all((candidate / name).is_file() for name in required):
            return str(candidate)
    raise RuntimeError("Falta la configuración local de SD1.5 en disco o en caché. "
                       "No se descargarán componentes automáticamente.")
