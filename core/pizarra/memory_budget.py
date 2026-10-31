"""Observational memory planning; never modifies admission or inference settings.

The system-wide used values are not attributed to individual processes.  In
particular, CUDA's device-used minus this process's PyTorch reservation includes
other CUDA contexts *and* non-PyTorch allocations in this process.
"""
import math
import os
from pathlib import Path

import psutil
import torch


def _fraction(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} debe ser una fracción entre 0 y 1") from error
    if not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError(f"{name} debe ser una fracción entre 0 y 1")
    return value


def _cgroup_available():
    """Best-effort cgroup v2 headroom. Absence is represented by None."""
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        used = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
        if limit != "max":
            return max(0, int(limit) - used)
    except (OSError, ValueError):
        pass
    return None


def inspect_budget(operation, *, storage=None, gpu_margin_fraction=None, ram_margin_fraction=None):
    """Return measured resource headroom and estimates as JSON-compatible data.

    ``storage`` can be any object exposing ``stats()``; failures and unavailable
    resources become null, not fabricated zeroes. The returned estimates are
    informational and cannot change memory settings or approve an operation.
    Margins are fractions of currently available memory, not fixed capacities.
    """
    if not isinstance(operation, str) or not operation.strip():
        raise ValueError("operation requerido")
    gpu_ratio = _fraction(gpu_margin_fraction if gpu_margin_fraction is not None else
                          os.environ.get("BUTTERVISION_PIZARRA_GPU_MARGIN_FRACTION", "0.1"), "margen GPU")
    ram_ratio = _fraction(ram_margin_fraction if ram_margin_fraction is not None else
                          os.environ.get("BUTTERVISION_PIZARRA_RAM_MARGIN_FRACTION", "0.1"), "margen RAM")
    ram = {"system_total_bytes": None, "system_used_bytes": None,
           "system_available_bytes": None, "process_rss_bytes": None,
           "visible_other_processes_rss_sum_bytes": None,
           "visible_other_processes_count": None,
           "other_processes_measurement_partial": None,
           "cgroup_available_bytes": None, "available_bytes": None,
           "safety_margin_bytes": None, "target_budget_bytes": None}
    try:
        vm = psutil.virtual_memory()
        ram.update(system_total_bytes=int(vm.total), system_used_bytes=int(vm.used),
                   system_available_bytes=int(vm.available))
        cgroup = _cgroup_available()
        ram["cgroup_available_bytes"] = cgroup
        ram["available_bytes"] = min(vm.available, cgroup) if cgroup is not None else vm.available
        ram["safety_margin_bytes"] = int(ram["available_bytes"] * ram_ratio)
        ram["target_budget_bytes"] = max(0, ram["available_bytes"] - ram["safety_margin_bytes"])
    except (OSError, psutil.Error):
        pass
    # Summed RSS counts shared pages more than once and excludes processes the
    # caller cannot inspect. Do not call this an exact "other RAM" value.
    try:
        rss_sum, count, partial = 0, 0, False
        for process in psutil.process_iter(["pid", "memory_info"]):
            try:
                if process.info["pid"] == os.getpid():
                    continue
                memory = process.info.get("memory_info")
                if memory is None:
                    partial = True
                    continue
                rss_sum += int(memory.rss)
                count += 1
            except (psutil.Error, OSError, KeyError, TypeError):
                partial = True
        ram.update(visible_other_processes_rss_sum_bytes=rss_sum,
                   visible_other_processes_count=count,
                   other_processes_measurement_partial=partial)
    except (psutil.Error, OSError):
        pass
    try:
        ram["process_rss_bytes"] = int(psutil.Process().memory_info().rss)
    except (OSError, psutil.Error):
        pass

    gpu = {"device_total_bytes": None, "device_free_bytes": None, "device_used_bytes": None,
           "torch_allocated_bytes": None, "torch_reserved_bytes": None,
           "device_used_not_torch_reserved_bytes": None, "safety_margin_bytes": None,
           "target_budget_bytes": None}
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
            used = total - free
            margin = int(free * gpu_ratio)
            gpu.update(device_total_bytes=int(total), device_free_bytes=int(free),
                       device_used_bytes=int(used), torch_allocated_bytes=int(allocated),
                       torch_reserved_bytes=int(reserved),
                       device_used_not_torch_reserved_bytes=int(used - reserved) if used >= reserved else None,
                       safety_margin_bytes=margin, target_budget_bytes=max(0, int(free - margin)))
    except (RuntimeError, OSError):
        pass

    disk = {"total_bytes": None, "free_bytes": None, "managed_bytes": None,
            "quota_remaining_bytes": None, "available_bytes": None}
    if storage is not None:
        try:
            value = storage.stats()
            for name in disk:
                if value.get(name) is not None:
                    disk[name] = int(value[name])
        except (OSError, RuntimeError, ValueError):
            pass
    return {"operation": operation, "observational": True,
            "gpu": gpu, "ram": ram, "pizarra": disk}
