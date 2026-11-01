"""Resource inspection and conservative automatic SDXL strategy planning."""
import inspect
import os
import shutil
from pathlib import Path

import psutil
import torch
import diffusers
from diffusers.utils import is_torch_version
import config

GIB = 1024**3
STANDARD_RAM_HEADROOM = 2 * GIB
LOW_RAM_HEADROOM = 2 * GIB
SEQUENTIAL_OFFLOAD_MIN_VRAM = 1536 * 1024**2

PROFILES = ("AUTO", "STANDARD", "LOW_VRAM", "LOW_RAM", "DISK_OFFLOAD")


def _cgroup_memory():
    mount = Path("/sys/fs/cgroup")
    roots = [mount]
    try:
        for row in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, controllers, relative = row.split(":", 2)
            if hierarchy == "0" and not controllers:
                current_dir = mount / relative.lstrip("/")
                roots = [current_dir, *current_dir.parents]
                break
    except (OSError, ValueError):
        pass
    candidates = []
    seen = set()
    for directory in roots:
        if directory in seen or not directory.is_relative_to(mount):
            continue
        seen.add(directory)
        try:
            limit_text = (directory / "memory.max").read_text().strip()
            current = int((directory / "memory.current").read_text())
            if limit_text == "max":
                continue
            limit = int(limit_text)
            candidates.append((max(0, limit - current), limit, current))
        except (OSError, ValueError):
            continue
    if candidates:
        available, limit, current = min(candidates)
        return limit, current, available
    return None, None, None


def _disk_target():
    raw = os.environ.get("BUTTERVISION_SDXL_DISK_OFFLOAD_DIR", "").strip()
    if not raw:
        return {"configured": False, "path": None, "free_bytes": None, "valid": False,
                "reason": "No hay un directorio de offload a disco configurado."}
    target = Path(raw).expanduser().resolve()
    normalized_parts = [part.lower() for part in target.parts]
    if any(normalized_parts[index:index + 2] == ["models", "stable-diffusion"]
           for index in range(max(0, len(normalized_parts) - 1))):
        return {"configured": True, "path": str(target), "free_bytes": None, "valid": False,
                "reason": "El directorio de offload no puede estar dentro de modelos/Stable-diffusion (Forge)."}
    from core.model_catalog import roots
    for root in roots("checkpoint"):
        if target == root or target.is_relative_to(root):
            return {"configured": True, "path": str(target), "free_bytes": None, "valid": False,
                    "reason": "El directorio de offload no puede estar dentro de una carpeta de checkpoints."}
    ancestor = target
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK | os.X_OK):
        return {"configured": True, "path": str(target), "free_bytes": None, "valid": False,
                "reason": "El directorio de offload no tiene permisos de escritura."}
    try:
        free = shutil.disk_usage(ancestor).free
    except OSError:
        return {"configured": True, "path": str(target), "free_bytes": None, "valid": False,
                "reason": "No se pudo medir el espacio libre del directorio de offload."}
    return {"configured": True, "path": str(target), "free_bytes": free, "valid": True, "reason": ""}


def collect_resources():
    """Collect process, host, cgroup, accelerator, and optional local disk data."""
    vm = psutil.virtual_memory()
    cgroup_limit, cgroup_current, cgroup_available = _cgroup_memory()
    ram_available = vm.available
    if cgroup_available is not None:
        ram_available = min(ram_available, cgroup_available)
    process_rss = psutil.Process().memory_info().rss
    accelerate_available = False
    try:
        from diffusers.utils import is_accelerate_available
        accelerate_available = bool(is_accelerate_available())
    except (ImportError, RuntimeError):
        pass
    try:
        from safetensors.torch import load_file
        mmap_available = callable(load_file)
    except ImportError:
        mmap_available = False
    sequential_available = False
    group_disk_available = False
    try:
        from diffusers import StableDiffusionXLPipeline
        sequential_available = callable(getattr(StableDiffusionXLPipeline, "enable_sequential_cpu_offload", None))
        group_offload = getattr(StableDiffusionXLPipeline, "enable_group_offload", None)
        group_parameters = inspect.signature(group_offload).parameters if callable(group_offload) else {}
        group_disk_available = "offload_to_disk_path" in group_parameters
    except (AttributeError, ImportError, TypeError, ValueError):
        group_disk_available = False
    cuda = torch.cuda.is_available()
    vram_free = vram_total = None
    if cuda:
        try:
            vram_free, vram_total = torch.cuda.mem_get_info()
        except Exception:
            pass
    return {
        "ram_total_bytes": vm.total,
        "ram_available_bytes": ram_available,
        "process_rss_bytes": process_rss,
        "cgroup_limit_bytes": cgroup_limit,
        "cgroup_current_bytes": cgroup_current,
        "cgroup_available_bytes": cgroup_available,
        "cuda_available": bool(cuda),
        "vram_available_bytes": vram_free,
        "vram_total_bytes": vram_total,
        "accelerate_available": accelerate_available,
        "safetensors_mmap_available": mmap_available,
        "low_cpu_mem_usage_available": accelerate_available and is_torch_version(">=", "1.9.0"),
        "sequential_cpu_offload_available": accelerate_available and sequential_available,
        "cpu_offload_configured": bool(config.model_config.enable_cpu_offload),
        "group_disk_offload_api_available": group_disk_available,
        "diffusers_version": getattr(diffusers, "__version__", "unknown"),
        "disk": _disk_target(),
    }


def plan_strategy(checkpoint_bytes, resources, requested="AUTO", *, allow_experimental_disk=False, progressive=None):
    """Return a strategy plan. No profile is admitted by changing thresholds alone."""
    requested = str(requested or "AUTO").upper()
    if requested not in PROFILES:
        raise ValueError(f"Estrategia SDXL desconocida: {requested}")
    size = int(checkpoint_bytes)
    ram = int(resources.get("ram_available_bytes") or 0)
    vram = resources.get("vram_available_bytes")
    cuda = bool(resources.get("cuda_available"))
    accelerate = bool(resources.get("accelerate_available"))
    mmap = bool(resources.get("safetensors_mmap_available"))
    low_cpu = bool(resources.get("low_cpu_mem_usage_available")) and accelerate
    seq = (bool(resources.get("sequential_cpu_offload_available")) and accelerate
           and bool(resources.get("cpu_offload_configured", True)))
    standard_ram = 2 * size + STANDARD_RAM_HEADROOM
    low_ram = size + LOW_RAM_HEADROOM
    if progressive is not None:
        # Final CPU weights + bounded conversion workspace + unchanged headroom.
        # Inference VRAM requirements below remain independent of loading.
        standard_ram = low_ram = (progressive["resident_bytes"] +
                                  progressive["temporary_bytes"] + LOW_RAM_HEADROOM)
    standard_vram = size + STANDARD_RAM_HEADROOM
    disk = resources.get("disk") or {}
    disk_required = int(size * 1.25 + GIB)
    disk_viable = bool(
        allow_experimental_disk and cuda and low_cpu and mmap
        and resources.get("group_disk_offload_api_available")
        and disk.get("valid") and (disk.get("free_bytes") or 0) >= disk_required
        and ram >= low_ram and vram is not None and vram >= SEQUENTIAL_OFFLOAD_MIN_VRAM
    )
    standard_viable = bool(
        (cuda and ram >= standard_ram and vram is not None and vram >= standard_vram)
        or (not cuda and ram >= standard_ram)
    )
    standard_viable = standard_viable and (progressive is None or accelerate)
    low_ram_viable = bool(cuda and low_cpu and mmap and ram >= low_ram and
                          (vram is not None and vram >= standard_vram or seq and vram is not None and vram >= SEQUENTIAL_OFFLOAD_MIN_VRAM))
    low_vram_viable = bool(cuda and seq and vram is not None and vram >= SEQUENTIAL_OFFLOAD_MIN_VRAM and ram >= low_ram)

    selected = None
    if requested == "AUTO":
        if standard_viable:
            selected = "STANDARD"
        elif low_ram_viable and ram < standard_ram:
            selected = "LOW_RAM"
        elif low_vram_viable:
            selected = "LOW_VRAM"
    elif requested == "STANDARD" and standard_viable:
        selected = requested
    elif requested == "LOW_RAM" and low_ram_viable:
        selected = requested
    elif requested == "LOW_VRAM" and low_vram_viable:
        selected = requested
    elif requested == "DISK_OFFLOAD" and disk_viable:
        selected = requested

    cpu_offload = bool(selected == "LOW_VRAM" or (selected == "LOW_RAM" and cuda and vram is not None and vram < standard_vram))
    reasons = []
    if selected is None:
        if not cuda and ram < standard_ram:
            reasons.append("CUDA no está disponible para un perfil SDXL de baja memoria validable.")
        if not accelerate:
            reasons.append("Accelerate no está disponible para inicialización de bajo uso de RAM y offload.")
        if ram < low_ram:
            reasons.append("La RAM disponible no alcanza pesos residentes + buffer + margen de carga progresiva."
                           if progressive is not None else
                           "La RAM disponible no alcanza el mínimo estimado para cargar los pesos en mmap con margen.")
        if cuda and (vram is None or vram < SEQUENTIAL_OFFLOAD_MIN_VRAM):
            reasons.append("La VRAM es menor que el mínimo medido para offload secuencial.")
        if requested == "DISK_OFFLOAD" and not disk_viable:
            reasons.append(disk.get("reason") or "DISK_OFFLOAD requiere API disponible, 1.25×checkpoint + 1 GiB libres y validación experimental.")
        elif requested == "STANDARD" and not standard_viable:
            if ram < standard_ram:
                reasons.append("STANDARD no tiene margen de RAM para la carga convencional.")
            if cuda and vram is not None and vram < standard_vram:
                reasons.append("STANDARD no tiene margen de VRAM para mantener el pipeline en GPU.")
        elif requested in ("LOW_RAM", "LOW_VRAM") and ram >= low_ram and not (low_ram_viable if requested == "LOW_RAM" else low_vram_viable):
            reasons.append(f"La combinación de recursos no admite el perfil {requested}.")
    return {
        "progressive_load": progressive is not None,
        "progressive_resident_bytes": progressive["resident_bytes"] if progressive else None,
        "progressive_temporary_bytes": progressive["temporary_bytes"] if progressive else None,
        "requested_profile": requested,
        "selected_profile": selected,
        "admitted": selected is not None,
        "load_phase": "planned" if selected else "rejected",
        "checkpoint_bytes": size,
        "ram_available_bytes": ram,
        "ram_process_rss_bytes": resources.get("process_rss_bytes"),
        "ram_cgroup_limit_bytes": resources.get("cgroup_limit_bytes"),
        "ram_cgroup_current_bytes": resources.get("cgroup_current_bytes"),
        "ram_standard_required_bytes": standard_ram,
        "ram_low_required_bytes": low_ram,
        "vram_available_bytes": vram,
        "vram_standard_required_bytes": standard_vram,
        "vram_low_required_bytes": SEQUENTIAL_OFFLOAD_MIN_VRAM,
        "cpu_offload_enabled": cpu_offload,
        "disk_offload_enabled": selected == "DISK_OFFLOAD",
        "disk_directory_configured": bool(disk.get("configured")),
        "disk_available_bytes": disk.get("free_bytes"),
        "disk_required_bytes": disk_required,
        "disk_experimental": selected == "DISK_OFFLOAD",
        "diffusers_version": resources.get("diffusers_version"),
        "capabilities": {
            "accelerate": accelerate,
            "low_cpu_mem_usage": low_cpu,
            "safetensors_mmap": mmap,
            "sequential_cpu_offload": seq,
            "group_disk_offload_api": bool(resources.get("group_disk_offload_api_available")),
        },
        "reasons": reasons,
    }
