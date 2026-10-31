"""Numerical operators over persistent tensor fragments (CPU only)."""
import math
import sys
import threading
import time

import numpy as np
import psutil

from .tensor_store import BufferLimiter, TensorRef, TensorStore


class ResourceSampler:
    """Sample process RSS and system availability without touching CUDA state."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.stop_event = threading.Event()
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.system_available_before = psutil.virtual_memory().available
        self.system_available_after = self.system_available_before
        self._thread = None
        self.measurements = None

    def _sample(self):
        while not self.stop_event.wait(self.interval):
            self.peak = max(self.peak, self.process.memory_info().rss)
            self.system_available_after = psutil.virtual_memory().available

    def start(self):
        self._thread = threading.Thread(target=self._sample, name="pizarra-tensor-resources", daemon=True)
        self._thread.start()

    def finish(self):
        if self.stop_event.is_set():
            return self.measurements
        self.stop_event.set()
        if self._thread:
            self._thread.join(timeout=max(1.0, self.interval * 4))
        current = self.process.memory_info().rss
        self.peak = max(self.peak, current)
        self.system_available_after = psutil.virtual_memory().available
        cuda = {"device_total_bytes": None, "device_free_bytes": None,
                "process_allocated_bytes": None, "process_reserved_bytes": None}
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    free, total = torch.cuda.mem_get_info()
                    cuda.update(device_total_bytes=total, device_free_bytes=free,
                                process_allocated_bytes=torch.cuda.memory_allocated(),
                                process_reserved_bytes=torch.cuda.memory_reserved())
            except (RuntimeError, OSError):
                pass
        self.measurements = {"rss_process_before_bytes": self.baseline, "rss_process_after_bytes": current,
                             "rss_process_peak_sampled_bytes": self.peak,
                             "system_available_before_bytes": self.system_available_before,
                             "system_available_after_bytes": self.system_available_after, "vram": cuda}
        return self.measurements


def _metrics(limiter):
    result = {"vram": "not used by the CPU executor"}
    result.update(limiter.snapshot())
    return result


def _finish_metrics(metrics, limiter):
    metrics.update(limiter.snapshot())
    return metrics


def transform(store: TensorStore, ref: TensorRef, function, *, block_elements=None):
    """Apply a bounded elementwise function and persist a new tensor."""
    if not callable(function):
        raise TypeError("function debe ser invocable")
    if store.policy.max_simultaneous_buffers < 4:
        raise MemoryError("La transformación requiere al menos cuatro buffers simultáneos")
    limiter = BufferLimiter(store.policy)
    metrics = _metrics(limiter)
    requested = block_elements or ref.block_elements
    max_output = min(store.policy.max_fragment_bytes, store.policy.result_partial_bytes) // np.dtype(ref.dtype).itemsize
    requested = min(requested, max_output)
    sampler = ResourceSampler()
    sampler.start()

    def output_blocks():
        for chunk in store.read_blocks(ref, limiter=limiter):
            with limiter.reserve(chunk.nbytes):
                result = np.asarray(function(chunk))
                if result.shape != chunk.shape or result.dtype != chunk.dtype:
                    raise ValueError("La transformación debe conservar forma y dtype")
                yield result

    try:
        result = store.write_tensor(ref.job_id, ref.shape, ref.dtype, output_blocks(),
                                    block_elements=requested, limiter=limiter)
    finally:
        sampled = sampler.finish()
    metrics.update(sampled)
    result_metrics = _finish_metrics(metrics, limiter)
    return result, result_metrics


def scale(store: TensorStore, ref: TensorRef, factor, *, block_elements=None):
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor):
        raise ValueError("factor debe ser finito")
    return transform(store, ref, lambda values: values * factor, block_elements=block_elements)


def reduce_sum(store: TensorStore, ref: TensorRef):
    """Float64 accumulation over one persisted chunk at a time."""
    limiter = BufferLimiter(store.policy)
    metrics = _metrics(limiter)
    sampler = ResourceSampler()
    sampler.start()
    total = np.float64(0)
    try:
        for chunk in store.read_blocks(ref, limiter=limiter):
            total += np.sum(chunk, dtype=np.float64)
    finally:
        sampled = sampler.finish()
    metrics.update(sampled)
    return float(total), _finish_metrics(metrics, limiter)


def matmul(store: TensorStore, left: TensorRef, right: TensorRef, *, block_size=64):
    """Multiply 2D tensors by row/column/K fragments, never loading full inputs."""
    left = store.open_tensor(left.job_id, left.reference)
    right = store.open_tensor(right.job_id, right.reference)
    if len(left.shape) != 2 or len(right.shape) != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("Dimensiones incompatibles para multiplicación matricial")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size debe ser entero positivo")
    required_buffers = (4 + int(left.dtype != "float32") + int(right.dtype != "float32"))
    if store.policy.max_simultaneous_buffers < required_buffers:
        raise MemoryError("La política no permite los buffers de multiplicación requeridos")
    itemsize = max(np.dtype(left.dtype).itemsize, np.dtype(right.dtype).itemsize, 4)
    partial_budget = store.policy.result_partial_bytes
    max_tile = min(int(math.sqrt(store.policy.max_fragment_bytes // itemsize)),
                   int(math.sqrt(partial_budget // np.dtype("float32").itemsize)))
    if max_tile < 1:
        raise MemoryError("El presupuesto no permite ni una celda de resultado")
    tile = min(block_size, max_tile)
    result_dtype = "float32"
    output_shape = (left.shape[0], right.shape[1])
    limiter = BufferLimiter(store.policy)
    metrics = _metrics(limiter)
    sampler = ResourceSampler()
    sampler.start()
    acc_dtype = np.dtype("float32")

    def products():
        for row in range(output_shape[0]):
            for col_start in range(0, output_shape[1], tile):
                col_stop = min(output_shape[1], col_start + tile)
                width = col_stop - col_start
                acc_bytes = width * acc_dtype.itemsize
                with limiter.reserve(acc_bytes):
                    acc = np.zeros((1, width), dtype=acc_dtype)
                    for k_start in range(0, left.shape[1], tile):
                        k_stop = min(left.shape[1], k_start + tile)
                        left_bytes = (k_stop - k_start) * np.dtype(left.dtype).itemsize
                        right_bytes = (k_stop - k_start) * width * np.dtype(right.dtype).itemsize
                        product_bytes = acc_bytes
                        cast_left = left.dtype != result_dtype
                        cast_right = right.dtype != result_dtype
                        with limiter.reserve(left_bytes), limiter.reserve(right_bytes), limiter.reserve(product_bytes):
                            a = store.read_block(left, row, row + 1, k_start, k_stop,
                                                limiter=limiter, result_reserved=True)
                            b = store.read_block(right, k_start, k_stop, col_start, col_stop,
                                                limiter=limiter, result_reserved=True)
                            extra_a = k_stop - k_start if cast_left else 0
                            extra_b = (k_stop - k_start) * width if cast_right else 0
                            with limiter.reserve(extra_a * acc_dtype.itemsize), limiter.reserve(extra_b * acc_dtype.itemsize):
                                a_work = a.astype(acc_dtype, copy=False)
                                b_work = b.astype(acc_dtype, copy=False)
                                product = np.empty((1, width), dtype=acc_dtype)
                                np.matmul(a_work, b_work, out=product)
                                acc += product
                                del product
                            del a_work, b_work
                        del a, b
                    yield acc.reshape(-1)
                    del acc

    try:
        result, write_metrics = _write_result(store, left.job_id, output_shape, result_dtype,
                                              products(), block_elements=min(store.policy.max_fragment_bytes,
                                                                            store.policy.result_partial_bytes) // 4,
                                              limiter=limiter)
    finally:
        sampled = sampler.finish()
    metrics.update(write_metrics)
    metrics.update(sampled)
    return result, _finish_metrics(metrics, limiter)


def _write_result(store, job_id, shape, dtype, blocks, *, block_elements, limiter):
    # The result iterator yields at most one row fragment. The store applies
    # its independent fragment cap and commits each chunk before requesting more.
    return store.write_tensor(job_id, shape, dtype, blocks, block_elements=block_elements,
                              limiter=limiter), limiter.snapshot()
