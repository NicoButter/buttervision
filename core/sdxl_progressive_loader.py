"""Bounded SDXL base loading. Conversion dictionaries contain recipes, never weights.

Diffusers 0.36 converters are used only with symbolic references. Unsupported
converter operations fail during planning, before any checkpoint tensor is read.
"""
import gc
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import psutil
import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)

DEFAULT_BUFFER_BYTES = 256 * 1024**2


class ProgressiveLoadError(ValueError):
    pass


@dataclass(frozen=True)
class TensorRef:
    key: str
    shape: tuple
    dtype: str
    operations: tuple = ()

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, index):
        shape = tuple(torch.empty(self.shape, device="meta")[index].shape)
        return TensorRef(self.key, shape, self.dtype, self.operations + (("slice", index),))

    @property
    def T(self):
        return TensorRef(self.key, self.shape[::-1], self.dtype, self.operations + (("transpose", None),))

    def contiguous(self):
        return self

    def clone(self):
        return self

    def detach(self):
        return self

    def apply(self, tensor):
        for operation, argument in self.operations:
            tensor = tensor[argument] if operation == "slice" else tensor.T
        return tensor


def read_index(path):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {key: TensorRef(key, tuple(handle.get_slice(key).get_shape()),
                               handle.get_slice(key).get_dtype()) for key in handle.keys()}


def validate_source(path):
    from core.model_catalog import approved_path, checkpoint_family, SDXL
    path = approved_path(path, "checkpoint")
    if path is None or path.suffix.lower() != ".safetensors" or checkpoint_family(path) != SDXL:
        raise ProgressiveLoadError("La carga progresiva requiere un checkpoint SDXL safetensors registrado.")
    return path


def conversion_plan(index, models):
    from diffusers.loaders.single_file_utils import (
        convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint,
        convert_ldm_clip_checkpoint, convert_open_clip_checkpoint,
    )
    try:
        plans = {
            "unet": convert_ldm_unet_checkpoint(index, models["unet"].config),
            "vae": convert_ldm_vae_checkpoint(index, models["vae"].config),
            "text_encoder": convert_ldm_clip_checkpoint(index, "conditioner.embedders.0.transformer."),
            "text_encoder_2": convert_open_clip_checkpoint(models["text_encoder_2"], index,
                                                         "conditioner.embedders.1.model."),
        }
        for name, model in models.items():
            expected = model.state_dict()
            # Old CLIP checkpoints may serialize this deterministic nonpersistent buffer.
            plans[name].pop("text_model.embeddings.position_ids", None)
            missing = set(expected) - set(plans[name])
            extra = set(plans[name]) - set(expected)
            if missing or extra:
                raise ProgressiveLoadError(f"{name}: faltantes={sorted(missing)}, inesperados={sorted(extra)}")
            for key, target in expected.items():
                ref = plans[name][key]
                if not isinstance(ref, TensorRef) or ref.shape != tuple(target.shape):
                    raise ProgressiveLoadError(f"{name}.{key}: forma incompatible")
                if ref.dtype not in ("F16", "BF16", "F32"):
                    raise ProgressiveLoadError(f"{name}.{key}: dtype no soportado {ref.dtype}")
        return plans
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ProgressiveLoadError(f"Variante SDXL o conversión incompatible: {error}") from error


def empty_models(config_dir):
    from accelerate import init_empty_weights
    from diffusers import UNet2DConditionModel, AutoencoderKL
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection
    root = Path(config_dir)
    with init_empty_weights():
        return {
            "unet": UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(root / "unet")),
            "vae": AutoencoderKL.from_config(AutoencoderKL.load_config(root / "vae")),
            "text_encoder": CLIPTextModel(CLIPTextConfig.from_pretrained(root / "text_encoder", local_files_only=True)),
            "text_encoder_2": CLIPTextModelWithProjection(CLIPTextConfig.from_pretrained(root / "text_encoder_2", local_files_only=True)),
        }


def assert_initialized(models):
    for name, model in models.items():
        for key, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            if tensor.is_meta:
                raise ProgressiveLoadError(f"Parámetro/buffer meta pendiente: {name}.{key}")


def load_progressive(path, config_dir, *, dtype=torch.float16,
                     max_buffer_bytes=DEFAULT_BUFFER_BYTES, cancelled=None, report=None):
    """Load on CPU, owning only final weights plus one bounded source/output pair.

    Each safetensors mapping is closed after copying into owned storage, avoiding
    accumulated file mappings/page residency. VAE uses the requested pipeline
    dtype; its local force_upcast configuration preserves Diffusers decoding
    precision handling (an initially FP32 VAE breaks FP16 latent decoding).
    """
    from accelerate.utils import set_module_tensor_to_device
    from diffusers import StableDiffusionXLPipeline
    from diffusers import schedulers
    from transformers import CLIPTokenizer
    from core.sdxl_memory import _cgroup_memory
    path = validate_source(path)
    started = time.monotonic()
    models = {}
    metrics = {"rss_peak_bytes": 0, "cgroup_peak_bytes": None, "vram_peak_bytes": None,
               "parameters_loaded": 0, "parameter_tensors_loaded": 0,
               "uninitialized": 0, "mapping_errors": 0, "read_seconds": 0.0,
               "conversion_seconds": 0.0, "components_seconds": {}}

    def sample(phase, emit=True):
        metrics["rss_peak_bytes"] = max(metrics["rss_peak_bytes"], psutil.Process().memory_info().rss)
        current = _cgroup_memory()[1]
        if current is not None:
            metrics["cgroup_peak_bytes"] = max(metrics["cgroup_peak_bytes"] or 0, current)
        if torch.cuda.is_available():
            metrics["vram_peak_bytes"] = max(metrics["vram_peak_bytes"] or 0, torch.cuda.max_memory_allocated())
        if emit:
            logger.info("SDXL progressive phase=%s metrics=%s", phase, metrics)
        if report and emit:
            report({"event": "sdxl_progressive_load", "phase": phase, **metrics})

    try:
        index = read_index(path)
        models = empty_models(config_dir)
        try:
            plans = conversion_plan(index, models)
        except ProgressiveLoadError:
            metrics["mapping_errors"] += 1
            raise
        sizes = {"F16": 2, "BF16": 2, "F32": 4}
        # Bound the full source plus its destination, including split QKV sources.
        for name, plan in plans.items():
            output_size = torch.empty((), dtype=dtype).element_size()
            for ref in plan.values():
                temporary = math.prod(index[ref.key].shape) * sizes[ref.dtype] + math.prod(ref.shape) * output_size
                if temporary > max_buffer_bytes:
                    raise ProgressiveLoadError(f"Buffer insuficiente para {ref.key}: {temporary} > {max_buffer_bytes}")
        root = Path(config_dir)
        scheduler_config = json.loads((root / "scheduler/scheduler_config.json").read_text())
        scheduler_class = getattr(schedulers, scheduler_config["_class_name"], None)
        if scheduler_class is None:
            raise ProgressiveLoadError("Scheduler local no soportado")
        scheduler = scheduler_class.from_pretrained(root / "scheduler", local_files_only=True)
        tokenizers = {name: CLIPTokenizer.from_pretrained(root / name, local_files_only=True)
                      for name in ("tokenizer", "tokenizer_2")}
        sample("validated")
        for name, model in models.items():
            component_start = time.monotonic()
            target_dtype = dtype
            for key, ref in plans[name].items():
                if cancelled and cancelled():
                    raise InterruptedError("Carga SDXL cancelada")
                tick = time.monotonic()
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    source = handle.get_tensor(ref.key)
                    metrics["read_seconds"] += time.monotonic() - tick
                    tick = time.monotonic()
                    value = torch.empty(ref.shape, dtype=target_dtype, device="cpu")
                    value.copy_(ref.apply(source))
                    set_module_tensor_to_device(model, key, "cpu", value=value, dtype=target_dtype)
                    metrics["parameters_loaded"] += value.numel()
                    metrics["parameter_tensors_loaded"] += 1
                    metrics["conversion_seconds"] += time.monotonic() - tick
                    # Observe while source and destination coexist.
                    sample(name, emit=False)
                    del value, source
            model.eval()
            metrics["components_seconds"][name] = time.monotonic() - component_start
            sample(name)
        metrics["uninitialized"] = sum(t.is_meta for model in models.values()
                                       for t in list(model.parameters()) + list(model.buffers()))
        assert_initialized(models)
        pipeline = StableDiffusionXLPipeline(**models, **tokenizers, scheduler=scheduler,
                                            force_zeros_for_empty_prompt=True, add_watermarker=False)
        metrics["total_seconds"] = time.monotonic() - started
        pipeline.progressive_load_metrics = metrics
        sample("complete")
        return pipeline
    except BaseException:
        model = source = value = pipeline = None
        models.clear()
        gc.collect()
        try:
            sample("failed_released")
        except Exception:
            logger.exception("No se pudo registrar la liberación tras el fallo SDXL")
        raise


def progressive_estimate(path, dtype):
    """Header-only conservative residency; excludes EMA/optimizer weights."""
    index = read_index(path)
    # Only original SDXL base layout is supported by this adapter.
    if not all(any(k.startswith(prefix) for k in index) for prefix in (
        "model.diffusion_model.", "first_stage_model.",
        "conditioner.embedders.0.transformer.", "conditioner.embedders.1.model.",
    )):
        return None
    prefixes = ("model.diffusion_model.", "first_stage_model.",
                "conditioner.embedders.0.transformer.", "conditioner.embedders.1.model.")
    final = sum(math.prod(ref.shape) * (4 if dtype == torch.float32 else 2)
                for key, ref in index.items() if key.startswith(prefixes))
    return {"resident_bytes": final, "temporary_bytes": DEFAULT_BUFFER_BYTES}
