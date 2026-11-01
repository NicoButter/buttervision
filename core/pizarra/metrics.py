"""Low-overhead, privacy-safe measurements; no CUDA synchronization."""
from contextlib import contextmanager
import threading
import time

import psutil
import torch


class PizarraMetrics:
    """Sample resource usage during one job.

    RSS and CUDA allocation peaks are sampled, not true instantaneous peaks.
    The CUDA allocator's separate high-water mark is process-lifetime scoped;
    it is deliberately not reset because doing so would alter global state.
    """

    def __init__(self, job_id, engine, enabled=True, interval=0.25):
        self.job_id = job_id
        self.engine = engine
        self.enabled = enabled
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._started = None
        self._process = None
        self._cpu_start = None
        self._phases = {}
        self._measurements = {
            "cpu_system_percent": None,
            "cpu_process_seconds": None,
            "process_threads": None,
            "ram_system_available_bytes": None,
            "rss_process_initial_bytes": None,
            "rss_process_bytes": None,
            "rss_process_final_bytes": None,
            "rss_process_peak_sampled_bytes": None,
            "vram_device_total_bytes": None,
            "vram_device_free_bytes": None,
            "vram_device_free_initial_bytes": None,
            "vram_device_free_final_bytes": None,
            "vram_process_allocated_bytes": None,
            "vram_process_reserved_bytes": None,
            "vram_process_peak_sampled_bytes": None,
            "vram_process_peak_lifetime_bytes": None,
            "gpu_execution_seconds": None,
        }

    def start(self):
        if not self.enabled:
            return
        self._started = time.monotonic()
        try:
            self._process = psutil.Process()
            times = self._process.cpu_times()
            self._cpu_start = times.user + times.system
            psutil.cpu_percent(interval=None)  # prime the counter; no blocking
        except (OSError, psutil.Error):
            self._process = None
        self._sample()
        self._thread = threading.Thread(target=self._poll, name="pizarra-metrics", daemon=True)
        self._thread.start()

    def _poll(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self):
        values = {}
        try:
            values["cpu_system_percent"] = psutil.cpu_percent(interval=None)
            values["ram_system_available_bytes"] = psutil.virtual_memory().available
            if self._process:
                values["rss_process_bytes"] = self._process.memory_info().rss
                values["process_threads"] = self._process.num_threads()
                times = self._process.cpu_times()
                if self._cpu_start is not None:
                    values["cpu_process_seconds"] = max(0.0, times.user + times.system - self._cpu_start)
        except (OSError, psutil.Error):
            pass
        try:
            # Avoid initializing a CUDA context merely for telemetry.
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                free, total = torch.cuda.mem_get_info()
                values.update(vram_device_total_bytes=total, vram_device_free_bytes=free,
                              vram_process_allocated_bytes=torch.cuda.memory_allocated(),
                              vram_process_reserved_bytes=torch.cuda.memory_reserved(),
                              vram_process_peak_lifetime_bytes=torch.cuda.max_memory_allocated())
        except (RuntimeError, OSError):
            pass
        with self._lock:
            self._measurements.update(values)
            for source, initial in (("rss_process_bytes", "rss_process_initial_bytes"),
                                    ("vram_device_free_bytes", "vram_device_free_initial_bytes")):
                if values.get(source) is not None and self._measurements[initial] is None:
                    self._measurements[initial] = values[source]
            for source, peak in (("rss_process_bytes", "rss_process_peak_sampled_bytes"),
                                 ("vram_process_allocated_bytes", "vram_process_peak_sampled_bytes")):
                value = values.get(source)
                if value is not None:
                    previous = self._measurements[peak]
                    self._measurements[peak] = value if previous is None else max(previous, value)

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        start = time.monotonic()
        try:
            yield
        finally:
            with self._lock:
                self._phases[name] = self._phases.get(name, 0.0) + time.monotonic() - start

    def finish(self, status, storage_stats=None):
        if self.enabled:
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=max(1.0, self.interval * 2))
            self._sample()
        with self._lock:
            metrics = dict(self._measurements)
            phases = dict(self._phases)
            metrics["rss_process_final_bytes"] = metrics.get("rss_process_bytes")
            metrics["vram_device_free_final_bytes"] = metrics.get("vram_device_free_bytes")
        metrics["duration_seconds"] = time.monotonic() - self._started if self._started is not None else None
        metrics["phases_seconds"] = phases
        metrics["pizarra_storage"] = dict(storage_stats) if storage_stats is not None else None
        return {"format_version": 1, "job_id": self.job_id, "engine": self.engine,
                "status": status, "metrics": metrics}
