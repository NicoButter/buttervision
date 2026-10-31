"""Small, privacy-safe lifecycle diagnostics and destructive model disposal."""
import gc
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import psutil
import torch

logger = logging.getLogger("uvicorn.error.engine")
_pizarra_metrics = ContextVar("pizarra_metrics", default=None)


@contextmanager
def bind_pizarra_metrics(collector):
    """Bind metrics to this worker only; inactive outside a PIZARRA job."""
    token = _pizarra_metrics.set(collector)
    try:
        yield
    finally:
        _pizarra_metrics.reset(token)


@contextmanager
def measured_phase(name):
    collector = _pizarra_metrics.get()
    if collector is None:
        yield
    else:
        with collector.phase(name):
            yield


def diagnose(event, pipeline="none", job_state="idle", **details):
    values = dict(event=event, pipeline=pipeline, job_state=job_state, **details)
    try:
        ram = psutil.virtual_memory()
        values.update(ram_used=ram.used, ram_available=ram.available,
                      process_rss=psutil.Process().memory_info().rss)
    except (OSError, psutil.Error):
        pass
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            values.update(vram_used=total - free, vram_available=free,
                          vram_allocated=torch.cuda.memory_allocated(),
                          vram_reserved=torch.cuda.memory_reserved())
    except RuntimeError:
        values["vram_unavailable"] = True
    logger.info("engine %s", values)


def collect_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def lifecycle(pipeline, operation):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            diagnose(f"{operation}_start", pipeline, "busy")
            state = "ok"
            try:
                with measured_phase("model_loading" if operation == "load" else "cleanup"):
                    return function(*args, **kwargs)
            except Exception:
                state = "failed"
                raise
            finally:
                diagnose(f"{operation}_end", pipeline, state)
        return wrapped
    return decorate


def _detach(resource):
    from accelerate.hooks import remove_hook_from_module

    if hasattr(resource, "remove_all_hooks"):
        resource.remove_all_hooks()
    # Some Diffusers versions only visit components with root hooks.
    # Sequential offload may install hooks on children alone.
    if isinstance(resource, torch.nn.Module):
        remove_hook_from_module(resource, recurse=True)
    else:
        for component in resource.components.values():
            if isinstance(component, torch.nn.Module):
                remove_hook_from_module(component, recurse=True)


def release_owned(owner, attributes):
    """Detach offload hooks, drop ALL owned references, then clear CUDA cache.

    Never move a whole sequentially offloaded pipeline to CUDA or CPU: doing so
    can materialize all weights at once. Accelerate detaches individual modules.
    Shared components are safe to visit twice. No objects survive in this frame
    when collect_memory is called.
    """
    failed = False
    for attribute in attributes:
        resource = getattr(owner, attribute, None)
        setattr(owner, attribute, None)
        try:
            if resource is not None:
                _detach(resource)
        except Exception as error:
            failed = True
            logger.error("engine release_error type=%s", type(error).__name__)
        finally:
            del resource
    collect_memory()
    if failed:
        raise RuntimeError("No se pudo completar la liberación del pipeline.")


def is_memory_error(error):
    return isinstance(error, (MemoryError, torch.OutOfMemoryError)) or any(
        marker in str(error).lower()
        for marker in ("out of memory", "can't allocate memory", "cannot allocate memory")
    )
