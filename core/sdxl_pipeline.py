"""Independent local SDXL backend; designed for offload, not validated on 4 GB."""
import os
import tempfile
from pathlib import Path

import psutil
import torch
from huggingface_hub import try_to_load_from_cache
from huggingface_hub.constants import HF_HUB_CACHE
from diffusers import StableDiffusionXLPipeline

import config
from core.engine_memory import lifecycle, measured_phase, release_owned
from core.prompt_limits import limit_prompts
from core.sdxl_memory import plan_strategy, collect_resources, _cgroup_memory

_GIB = 1024**3
_LOW_CPU_MEM_SAFETY_BYTES = 2 * _GIB


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
    _, _, cgroup_available = _cgroup_memory()
    if cgroup_available is not None:
        available = min(available, cgroup_available)
    return available


def _weight_bytes(model):
    path = Path(model)
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(item.stat().st_size for item in path.rglob("*.safetensors") if item.is_file())
    except OSError as error:
        raise ValueError("No se pueden leer los pesos SDXL locales.") from error


def inspect_load_budget(model, requested="AUTO", *, allow_experimental_disk=False):
    size = _weight_bytes(model)
    if not size:
        raise ValueError("No se encontraron pesos safetensors locales para SDXL.")
    resources = collect_resources()
    resources["ram_available_bytes"] = available_ram()
    assessment = plan_strategy(size, resources, requested, allow_experimental_disk=allow_experimental_disk)
    assessment.update({
        "strategy": assessment["selected_profile"],
        "estimated_required_ram_bytes": assessment["ram_low_required_bytes"],
        "ram_safety_margin_bytes": _LOW_CPU_MEM_SAFETY_BYTES,
        "low_cpu_mem_usage_available": assessment["capabilities"]["low_cpu_mem_usage"],
        "cuda_available": resources["cuda_available"],
        "available_vram_bytes": resources["vram_available_bytes"],
        "required_vram_bytes": (assessment["vram_low_required_bytes"] if assessment["cpu_offload_enabled"]
                                 else assessment["vram_standard_required_bytes"]),
    })
    return assessment


def check_budget(model, requested="AUTO", *, allow_experimental_disk=False):
    assessment = inspect_load_budget(model, requested, allow_experimental_disk=allow_experimental_disk)
    if not assessment["admitted"]:
        raise ValueError("; ".join(assessment["reasons"]) or "Ningún perfil de memoria SDXL es viable con los recursos actuales.")
    return assessment


class SDXLPipeline:
    def __init__(self, model_id, requested_strategy="AUTO", allow_experimental_disk=False):
        self.model_id = model_id
        self.requested_strategy = str(requested_strategy or "AUTO").upper()
        self.allow_experimental_disk = bool(allow_experimental_disk)
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_state = "selected"
        self.load_error = ""
        self.load_plan = None
        self._disk_offload_temp = None

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
            if self.load_plan is not None:
                self.load_plan["load_phase"] = "error"
                self.load_plan["reasons"] = [type(error).__name__]
            if self.pipe is not None:
                try:
                    self.cleanup()
                except Exception:
                    # Keep the load failure as the primary recoverable error.
                    pass
            raise

    @lifecycle("sdxl", "load")
    def _build_pipeline(self):
        plan = check_budget(self.model_id, requested=self.requested_strategy,
                            allow_experimental_disk=self.allow_experimental_disk)
        self.load_plan = plan
        self.load_plan["load_phase"] = "loading"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        low_cpu = plan["selected_profile"] != "STANDARD"
        source = Path(self.model_id)
        if source.is_file():
            pipe = StableDiffusionXLPipeline.from_single_file(
                str(source), config=local_sdxl_config(source), torch_dtype=dtype,
                local_files_only=True, low_cpu_mem_usage=low_cpu, disable_mmap=False,
            )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                str(source), torch_dtype=dtype, local_files_only=True,
                use_safetensors=True, low_cpu_mem_usage=low_cpu,
            )
        self.pipe = pipe
        pipe.enable_attention_slicing("max")
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        if plan["selected_profile"] == "DISK_OFFLOAD":
            target = Path(os.environ["BUTTERVISION_SDXL_DISK_OFFLOAD_DIR"]).expanduser().resolve()
            target.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._disk_offload_temp = tempfile.TemporaryDirectory(prefix="buttervision-sdxl-", dir=target)
            pipe.enable_group_offload(
                onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                offload_type="block_level", num_blocks_per_group=1,
                offload_to_disk_path=self._disk_offload_temp.name,
            )
        elif plan["cpu_offload_enabled"]:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(self.device)
        self.load_state = "loaded"
        self.load_plan["load_phase"] = "loaded"
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
        try:
            release_owned(self, ("pipe",))
        finally:
            disk_temp, self._disk_offload_temp = self._disk_offload_temp, None
            if disk_temp is not None:
                disk_temp.cleanup()
            if self.load_state != "error":
                self.load_state = "selected"
