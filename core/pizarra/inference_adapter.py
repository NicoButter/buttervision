"""Experimental PIZARRA orchestration for one isolated, local SDXL inference.

This adapter plans and observes Diffusers execution. It does not implement tensor
math or allow Diffusers to write directly into the protected PIZARRA volume.
"""
from datetime import datetime, timezone
import uuid
from pathlib import Path

from core.model_catalog import SDXL, approved_path, checkpoint_family
from core.sdxl_memory import collect_resources, plan_strategy
from core.sdxl_pipeline import SDXLPipeline, _weight_bytes
from core.engine_memory import is_memory_error

from .metrics import PizarraMetrics
from .memory_budget import inspect_budget

GIB = 1024**3


class InferencePlanError(RuntimeError):
    def __init__(self, message, plan=None):
        super().__init__(message)
        self.plan = plan or {}


class InferenceExecutionError(RuntimeError):
    def __init__(self, error_type, category, record):
        super().__init__(f"PIZARRA inference failed ({category})")
        self.error_type = error_type
        self.category = category
        self.record = record


class PizarraInferenceAdapter:
    """Select, load, measure and release a single SDXL execution strategy.

    ``storage`` must be an enabled PizarraStorage instance on its authorized
    mount. Only its validation and accounting API are used. Diffusers receives
    no path into the PIZARRA bank.
    """

    def __init__(self, model, storage, *, requested_strategy="AUTO", metrics_enabled=True,
                 sampling_interval=0.25):
        self.model = str(model)
        self.storage = storage
        self.requested_strategy = str(requested_strategy or "AUTO").upper()
        self.metrics_enabled = bool(metrics_enabled)
        self.sampling_interval = float(sampling_interval)

    def plan(self):
        approved = approved_path(self.model, "checkpoint")
        if approved is None or checkpoint_family(approved) != SDXL:
            raise InferencePlanError("El adaptador experimental solo acepta checkpoints SDXL registrados localmente.")
        if self.storage is None or not self.storage.config.enabled:
            raise InferencePlanError("PIZARRA debe estar habilitada para ejecutar el adaptador experimental.")
        storage_stats = self.storage.validate()
        if not storage_stats.get("available"):
            raise InferencePlanError("El volumen PIZARRA autorizado no está disponible.")

        if self.requested_strategy == "DISK_OFFLOAD_EXPERIMENTAL":
            raise InferencePlanError(
                "Offload a disco bloqueado: Diffusers requiere escritura mutable directa y PIZARRA "
                "no ofrece una concesión con cuota impuesta por escritura."
            )
        profile_request = {
            "AUTO": "AUTO",
            "STANDARD": "STANDARD",
            "CPU_OFFLOAD": "LOW_VRAM",
        }.get(self.requested_strategy)
        if profile_request is None:
            raise InferencePlanError("Estrategia PIZARRA desconocida.")

        resources = collect_resources()
        observed_budget = inspect_budget("sdxl_inference", storage=self.storage)
        storage_stats = self.storage.validate()
        if not storage_stats.get("available"):
            raise InferencePlanError("El volumen PIZARRA dejó de estar disponible durante el preflight.")
        ram_target = observed_budget.get("ram", {}).get("target_budget_bytes")
        if ram_target is not None:
            resources["ram_available_bytes"] = min(resources["ram_available_bytes"], ram_target)
        vram_target = observed_budget.get("gpu", {}).get("target_budget_bytes")
        if vram_target is not None and resources.get("vram_available_bytes") is not None:
            resources["vram_available_bytes"] = min(resources["vram_available_bytes"], vram_target)
        checkpoint_bytes = _weight_bytes(approved)
        plan = plan_strategy(checkpoint_bytes, resources, profile_request,
                             allow_experimental_disk=False)
        plan.update({
            "model_identifier": approved.name,
            "family": SDXL,
            "requested_strategy": self.requested_strategy.lower(),
            "storage_volume_available": True,
            "storage_volume_total_bytes": storage_stats.get("total_bytes"),
            "storage_volume_free_bytes": storage_stats.get("free_bytes"),
            "storage_quota_remaining_bytes": storage_stats.get("quota_remaining_bytes"),
            "storage_direct_offload_allowed": False,
            "pizarra_memory_budget": {
                "ram": observed_budget.get("ram"),
                "gpu": observed_budget.get("gpu"),
                "pizarra": observed_budget.get("pizarra"),
            },
        })
        if not plan["admitted"]:
            raise InferencePlanError("No existe una estrategia PIZARRA viable: " + "; ".join(plan["reasons"]), plan)
        plan["strategy"] = "cpu_offload" if plan["cpu_offload_enabled"] else "standard"
        return approved, plan

    def execute(self, prompt, *, seed=230231, width=512, height=512, num_inference_steps=10,
                guidance_scale=5.0, num_images=1, run_id=None, planned=None, on_phase=None):
        approved, plan = planned if planned is not None else self.plan()
        run_id = run_id or str(uuid.uuid4())
        backend_profile = plan["selected_profile"]
        backend = SDXLPipeline(str(approved), requested_strategy=backend_profile)
        metrics = PizarraMetrics(run_id, "sdxl", enabled=self.metrics_enabled,
                                 interval=self.sampling_interval)
        storage_before = self.storage.stats()
        metrics.start()
        started = datetime.now(timezone.utc).isoformat()
        images = None
        failure = None
        failure_category = None
        cleanup_failure = None
        status = "failed"
        try:
            if on_phase:
                on_phase("weight_loading", approved, plan, run_id)
            with metrics.phase("weight_loading"):
                backend.load_pipeline()
            if on_phase:
                on_phase("inference_and_decoding", approved, plan, run_id)
            with metrics.phase("inference_and_decoding"):
                images = backend.generate_image(
                    prompt, seed=seed, num_images=num_images,
                    num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                    width=width, height=height,
                )
            status = "completed"
        except Exception as error:
            failure = error
            failure_category = "memory_allocation" if is_memory_error(error) else "inference_error"
        finally:
            try:
                with metrics.phase("cleanup"):
                    backend.cleanup()
            except Exception as error:
                cleanup_failure = error
                if failure is None:
                    failure = error
                    failure_category = "cleanup_error"

        try:
            storage_after = self.storage.stats()
        except Exception as error:
            storage_after = {"free_bytes": None, "managed_bytes": None,
                             "quota_remaining_bytes": None}
            if failure is None:
                failure = error
                failure_category = "storage_error"
        if failure is not None:
            status = "failed"
        storage_delta = _storage_delta(storage_before, storage_after)
        sampled = metrics.finish(status, storage_stats=storage_delta)
        finished = datetime.now(timezone.utc).isoformat()
        measurements = sampled["metrics"]
        record = {
            "format_version": 1,
            "run_id": run_id,
            "started_at": started,
            "finished_at": finished,
            "model": approved.name,
            "family": SDXL,
            "checkpoint_bytes": plan["checkpoint_bytes"],
            "strategy": plan["strategy"],
            "profile": plan["selected_profile"],
            "status": status,
            "initial_resources": {
                "ram_available_bytes": plan["ram_available_bytes"],
                "rss_process_bytes": measurements.get("rss_process_initial_bytes"),
                "vram_device_free_bytes": measurements.get("vram_device_free_initial_bytes"),
                "vram_device_total_bytes": measurements.get("vram_device_total_bytes"),
            },
            "final_resources": {
                "rss_process_bytes": measurements.get("rss_process_final_bytes"),
                "rss_process_peak_sampled_bytes": measurements.get("rss_process_peak_sampled_bytes"),
                "vram_device_free_bytes": measurements.get("vram_device_free_final_bytes"),
                "vram_process_allocated_bytes": measurements.get("vram_process_allocated_bytes"),
                "vram_process_reserved_bytes": measurements.get("vram_process_reserved_bytes"),
                "vram_process_peak_sampled_bytes": measurements.get("vram_process_peak_sampled_bytes"),
                "vram_process_peak_lifetime_bytes": measurements.get("vram_process_peak_lifetime_bytes"),
            },
            "timings_seconds": {
                "weight_loading": measurements.get("phases_seconds", {}).get("weight_loading"),
                "inference_and_decoding": measurements.get("phases_seconds", {}).get("inference_and_decoding"),
                "decoding_separate": None,
                "cleanup": measurements.get("phases_seconds", {}).get("cleanup"),
                "total": measurements.get("duration_seconds"),
            },
            "pizarra_storage_api": {
                "read_bytes": storage_delta.get("read_bytes"),
                "written_bytes": storage_delta.get("written_bytes"),
                "read_seconds": storage_delta.get("read_seconds"),
                "write_seconds": storage_delta.get("write_seconds"),
                "temporary_offload_bytes": 0,
                "direct_diffusers_writes": False,
            },
            "storage_volume": {
                "available": True,
                "free_bytes": storage_after.get("free_bytes"),
                "managed_bytes": storage_after.get("managed_bytes"),
                "quota_remaining_bytes": storage_after.get("quota_remaining_bytes"),
            },
            "measurements": measurements,
            "measurement_notes": [
                "RSS y VRAM peak son muestras periódicas, no picos instantáneos.",
                "PizarraStorage contabiliza bytes de su API; no representa E/S física total del HDD.",
                "Diffusers devuelve imágenes decodificadas en una llamada; la decodificación separada no se midió.",
                "Este adaptador no copia pesos ni usa Tensor Store para ejecutar tensores.",
            ],
            "failure": ({"category": failure_category or "inference_error",
                         "error_type": type(failure).__name__,
                         "cleanup_error_type": type(cleanup_failure).__name__ if cleanup_failure else None}
                        if failure else None),
        }
        if failure:
            raise InferenceExecutionError(type(failure).__name__, record["failure"]["category"], record) from None
        return images, record


def _storage_delta(before, after):
    result = {}
    for key in ("read_bytes", "written_bytes", "read_seconds", "write_seconds"):
        left, right = before.get(key), after.get(key)
        result[key] = max(0, right - left) if left is not None and right is not None else None
    return result
