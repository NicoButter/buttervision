"""Local-only architecture inspection. Safetensors headers are read without tensors."""
import json
import os
from pathlib import Path

from safetensors import SafetensorError, safe_open

import config

SD15 = "sd15"
SDXL = "sdxl"
UNKNOWN = "unknown"


def _header(path):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = handle.keys()
        metadata = handle.metadata() or {}
        shapes = {key: handle.get_slice(key).get_shape() for key in keys
                  if "attn2" in key and ("to_k" in key or "to_v" in key)}
        return keys, metadata, shapes


def checkpoint_family(path):
    path = Path(path)
    try:
        if path.is_dir():
            index = json.loads((path / "model_index.json").read_text())
            unet = json.loads((path / "unet/config.json").read_text())
            dim = unet.get("cross_attention_dim")
            cls = index.get("_class_name")
            if cls == "StableDiffusionXLPipeline" and dim == 2048 and all(
                (path / name).is_file() for name in ("text_encoder_2/config.json", "tokenizer_2/tokenizer_config.json")
            ):
                return SDXL
            if cls == "StableDiffusionPipeline" and dim == 768 and not index.get("text_encoder_2"):
                return SD15
            return UNKNOWN
        if path.suffix.lower() != ".safetensors":
            return UNKNOWN
        keys, _, shapes = _header(path)
        dims = {shape[1] for key, shape in shapes.items() if len(shape) == 2 and
                (key.endswith(".weight")) and not any(x in key for x in ("lora", "adapter"))}
        is_xl = any(k.startswith("conditioner.embedders.1.") or k.startswith("text_encoder_2.") for k in keys)
        is_model = any(k.startswith(("model.diffusion_model.", "unet.")) for k in keys)
        has_vae = any(k.startswith(("first_stage_model.", "vae.")) for k in keys)
        has_text = any(k.startswith(("cond_stage_model.", "conditioner.embedders.0.", "text_encoder.")) for k in keys)
        if is_model and has_vae and has_text and is_xl and 2048 in dims and 768 not in dims:
            return SDXL
        if is_model and has_vae and has_text and not is_xl and 768 in dims and 2048 not in dims:
            return SD15
    except (OSError, ValueError, KeyError, SafetensorError, RuntimeError):
        pass
    return UNKNOWN


def lora_family(path):
    try:
        keys, metadata, shapes = _header(path)
        if not any("lora" in key.lower() for key in keys):
            return UNKNOWN
        evidence = set()
        version = str(metadata.get("ss_base_model_version", "")).lower()
        if version in ("sdxl_base_v1-0", "sdxl", "sd_xl_base_1.0"):
            evidence.add(SDXL)
        elif version in ("sd_v1", "sd1", "sd1.5", "sd15"):
            evidence.add(SD15)
        if any(k.startswith(("lora_te2_", "text_encoder_2.")) for k in keys):
            evidence.add(SDXL)
        for key, shape in shapes.items():
            if ("lora_down" in key or "lora_A" in key) and len(shape) == 2:
                if shape[1] == 768:
                    evidence.add(SD15)
                elif shape[1] == 2048:
                    evidence.add(SDXL)
        return evidence.pop() if len(evidence) == 1 else UNKNOWN
    except (OSError, ValueError, KeyError, SafetensorError, RuntimeError):
        return UNKNOWN


def roots(kind):
    defaults = [config.SD_MODELS_DIR] if kind == "checkpoint" else [config.LORA_DIR, config.PROJECT_ROOT / "loras"]
    variable = "BUTTERVISION_CHECKPOINT_DIRS" if kind == "checkpoint" else "BUTTERVISION_LORA_DIRS"
    values = defaults + [Path(p).expanduser() for p in os.environ.get(variable, "").split(os.pathsep) if p]
    result = []
    for root in values:
        try:
            root = root.resolve(strict=True)
            if root.is_dir() and root != Path(root.anchor) and root not in result:
                result.append(root)
        except (OSError, RuntimeError):
            continue
    return result


def entries(kind):
    found = set()
    for root in roots(kind):
        try:
            for item in (root.iterdir() if kind == "checkpoint" else root.rglob("*.safetensors")):
                try:
                    path = item.resolve(strict=True)
                    if not path.is_relative_to(root):
                        continue
                    if kind == "checkpoint" and ((path.is_file() and path.suffix.lower() in (".safetensors", ".ckpt")) or
                                                  (path.is_dir() and (path / "model_index.json").is_file())):
                        found.add(path)
                    elif kind == "lora" and path.is_file() and path.suffix.lower() == ".safetensors":
                        found.add(path)
                except (OSError, RuntimeError):
                    continue
        except OSError:
            continue
    return sorted(found, key=lambda p: str(p).lower())


def approved_path(value, kind):
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError):
        return None
    return path if path in entries(kind) else None
