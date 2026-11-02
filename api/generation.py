import json
import random
import threading
import traceback
import uuid
import hashlib
import math
import sqlite3
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PIL.PngImagePlugin import PngInfo

import config
from api.output_files import list_public_output_images
from api.uploads import decode_reference_image
from core.advanced_pipeline import ButterVisionPipeline
from core.face_reference_pipeline import SD15FaceReferencePipeline
from core.model_manager import ModelManager
from core.engine_memory import bind_pizarra_metrics, collect_memory, diagnose, is_memory_error, measured_phase
from core.model_catalog import SD15, SDXL, UNKNOWN, approved_path, checkpoint_family, entries, lora_family
from core.sdxl_pipeline import SDXLPipeline, check_budget, inspect_load_budget, local_sdxl_config
from core.inference_lock import acquire_inference_lock, release_inference_lock


class EngineBusyError(RuntimeError):
    def __init__(self):
        super().__init__("El motor está ocupado generando o cambiando de modelo. Intenta nuevamente cuando termine.")


class PizarraUnavailableError(RuntimeError):
    """PIZARRA could not safely accept the job; no private details exposed."""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _public_output_path(path: Path) -> str:
    return f"/outputs/{path.relative_to(config.OUTPUTS_DIR).as_posix()}"


@dataclass
class GenerationJob:
    id: str
    mode: str
    payload: dict
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    result_images: List[str] = field(default_factory=list)
    info: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    warnings: List[str] = field(default_factory=list)
    _lock: object = field(default_factory=threading.RLock, repr=False)
    _defer_completion: bool = field(default=False, repr=False)
    _pending_completion: Optional[dict] = field(default=None, repr=False)

    def update(self, **kwargs):
        with self._lock:
            if self._defer_completion and kwargs.get("status") == "completed":
                self._pending_completion = dict(kwargs)
                self.result_images = list(kwargs.get("result_images", self.result_images))
                self.info = kwargs.get("info", self.info)
                self.progress = 0.95
                self.message = "Finalizing"
                self.updated_at = _now()
                return
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.updated_at = _now()

    def terminal_snapshot(self):
        with self._lock:
            snapshot = self._snapshot()
            if self._pending_completion is not None and self.status != "failed":
                snapshot.update(self._pending_completion)
                snapshot["updated_at"] = _now()
            return snapshot

    def publish_completion(self):
        with self._lock:
            if self._pending_completion is not None:
                self._defer_completion = False
                self.update(**self._pending_completion)
                self._pending_completion = None

    def warn(self, message):
        with self._lock:
            if message not in self.warnings:
                self.warnings.append(message)

    def as_dict(self):
        with self._lock:
            return self._snapshot()

    def _snapshot(self):
        return {
            "id": self.id,
            "mode": self.mode,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "result_images": list(self.result_images),
            "warnings": list(self.warnings),
            "info": self.info,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class GenerationService:
    def __init__(self):
        self.model_manager = ModelManager()
        active_model = self.model_manager.resolve_model_path(config.model_config.model_id) or config.model_config.model_id
        config.model_config.model_id = active_model
        self.family = checkpoint_family(active_model)
        self.sd_manager = (SDXLPipeline(active_model) if self.family == SDXL else ButterVisionPipeline(
            model_id=active_model, enable_optimizations=True, enable_lcm=False,
        ))
        self.lora_registry = {}
        self.face_manager: Optional[SD15FaceReferencePipeline] = None
        self.jobs: Dict[str, GenerationJob] = {}
        self.lock = threading.Lock()
        self.worker_lock = threading.Lock()
        self.active_pipeline = "none"
        self.model_state = "selected" if self.family in (SD15, SDXL) else "incompatible"
        self.model_error = ""
        self.load_budget = None
        from core.pizarra.config import PizarraConfig

        self.pizarra_config = PizarraConfig.from_env()
        self.pizarra_storage = None
        self.pizarra_store = None
        if self.pizarra_config.enabled:
            from core.pizarra.storage import PizarraStorage
            from core.pizarra.job_store import JobStore

            # The ledger stays on application-local storage, not on the bank.
            db_dir = config.PROJECT_ROOT / "data" / "pizarra"
            if not db_dir.resolve().is_relative_to(config.PROJECT_ROOT.resolve()):
                raise PizarraUnavailableError("El directorio local de metadatos no es válido.")
            try:
                self.pizarra_store = JobStore(db_dir / "jobs.sqlite3")
                self.pizarra_storage = PizarraStorage(self.pizarra_config)
            except (OSError, ValueError, sqlite3.Error) as error:
                raise PizarraUnavailableError("No se pudo abrir el registro local PIZARRA.") from error

    def list_models(self):
        model_infos = self.model_manager.list_local_model_infos()
        active = self.sd_manager.model_id
        models = []
        counts = {}
        for info in model_infos:
            counts[info["name"]] = counts.get(info["name"], 0) + 1
        for info in model_infos:
            value = info["path"]
            family = info.get("family", UNKNOWN)
            state = "incompatible" if family == UNKNOWN else ("selected" if value == active else "detected")
            if value == active:
                state = self.model_state
            label = Path(value).name
            if counts.get(info["name"], 0) > 1:
                path_tag = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
                label = f"{label} · {info.get('origin', 'Otra carpeta')} · {path_tag}"
            models.append({"label": label, "value": value, "active": value == active,
                           "family": family, "state": state, "origin": info.get("origin", "Otra carpeta")})
        if active and all(model["value"] != active for model in models):
            models.insert(0, {"label": Path(active).name, "value": active, "active": True,
                              "family": self.family, "state": self.model_state})
        return models

    def resolutions(self):
        if self.family != SDXL:
            return [256, 320, 384, 448, 512]
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 8 * 1024**3:
            return [512, 640, 768, 896, 1024]
        return [512, 640, 768]

    def engine_state(self):
        loras = self.list_loras()
        pipe = self._pipeline()
        scheduler = getattr(pipe, "scheduler", None) if pipe is not None else None
        load_budget = getattr(self.sd_manager, "load_plan", None) or self.load_budget
        if load_budget is None and self.family == SDXL:
            try:
                self.load_budget = inspect_load_budget(self.sd_manager.model_id)
            except Exception as error:
                self.load_budget = {"requested_profile": "AUTO", "selected_profile": None,
                                    "admitted": False, "load_phase": "rejected",
                                    "reasons": [type(error).__name__]}
            load_budget = self.load_budget
        return {"status": "busy" if self.worker_lock.locked() else "idle",
                "pipeline": self.active_pipeline, "model": Path(self.sd_manager.model_id).name,
                "model_state": self.model_state, "model_error": self.model_error,
                "memory_strategy": load_budget.get("selected_profile") if load_budget else None,
                "load_budget": load_budget,
                "family": self.family, "resolutions": self.resolutions(),
                "active_loras": [item for item in loras if item["active"]],
                "face_supported": self.family == SD15,
                "scheduler": type(scheduler).__name__ if scheduler is not None else None}

    def _pipeline(self):
        return getattr(self.sd_manager, "pipe" if self.family == SDXL else "sd_pipeline", None)

    def _sync_legacy_loras(self):
        if self.family == SD15:
            for name, weight in self.sd_manager.loaded_loras.items():
                path = self.sd_manager.available_loras.get(name)
                if path and approved_path(path, "lora"):
                    self.lora_registry.setdefault(str(Path(path).resolve()), {"adapter": name, "weight": weight})

    def list_loras(self):
        pipeline = self._pipeline()
        result = []
        seen = set()
        for path in entries("lora"):
            seen.add(str(path))
            family = lora_family(path)
            registered = self.lora_registry.get(str(path))
            compatible = family == self.family and family != UNKNOWN
            result.append({"value": str(path), "label": path.stem, "family": family,
                           "compatible": compatible, "active": bool(pipeline is not None and registered),
                           "weight": registered["weight"] if registered else 1.0,
                           "reason": "" if compatible else "Familia desconocida o incompatible con el checkpoint activo."})
        for path, registered in tuple(self.lora_registry.items()):
            if path not in seen and pipeline is not None:
                result.append({"value": path, "label": Path(path).stem, "family": self.family,
                               "compatible": False, "active": True, "weight": registered["weight"],
                               "reason": "El archivo ya no está disponible; puedes desactivar el adaptador cargado."})
        return result

    def set_lora(self, path, enabled, weight=1.0):
        inference_fd = self._acquire_mutation_locks()
        try:
            item = approved_path(path, "lora")
            key = str(item) if item is not None else str(Path(path).expanduser().absolute())
            if item is None and (enabled or key not in self.lora_registry):
                raise ValueError("LoRA fuera de los directorios registrados o inexistente.")
            if enabled and (lora_family(item) != self.family or self.family not in (SD15, SDXL)):
                raise ValueError("LoRA incompatible o de arquitectura desconocida.")
            weight = float(weight)
            if not math.isfinite(weight) or not 0 <= weight <= 2:
                raise ValueError("El peso LoRA debe estar entre 0 y 2.")
            if not enabled and key not in self.lora_registry:
                return self.list_loras()
            if self.family == SDXL and self._pipeline() is None:
                self.load_budget = check_budget(self.sd_manager.model_id)
            self._release_face()
            pipe = None
            try:
                pipe = (self.sd_manager.load_pipeline() if self.family == SDXL else self.sd_manager.load_sd_pipeline())
                self.active_pipeline = "txt2img"
                self._sync_legacy_loras()
                updated = dict(self.lora_registry)
                existing = updated.get(key)
                if enabled:
                    adapter = existing["adapter"] if existing else "lora_" + hashlib.sha256(key.encode()).hexdigest()[:16]
                    if not existing:
                        pipe.load_lora_weights(str(item.parent), weight_name=item.name, adapter_name=adapter, local_files_only=True)
                    updated[key] = {"adapter": adapter, "weight": weight}
                elif existing:
                    pipe.delete_adapters(existing["adapter"])
                    updated.pop(key)
                    if self.family == SD15:
                        self.sd_manager.loaded_loras.pop(existing["adapter"], None)
                if updated:
                    pipe.set_adapters([v["adapter"] for v in updated.values()], adapter_weights=[v["weight"] for v in updated.values()])
                    actual = set(pipe.get_active_adapters())
                    if not {v["adapter"] for v in updated.values()}.issubset(actual):
                        raise RuntimeError("Los adaptadores solicitados no quedaron activos.")
                self.lora_registry = updated
            except Exception as error:
                self.lora_registry.clear()
                traceback.clear_frames(error.__traceback__)
                error.__traceback__ = None
                del pipe
                self.sd_manager.cleanup()
                self.active_pipeline = "none"
                raise
            return self.list_loras()
        finally:
            self._release_mutation_locks(inference_fd)

    def set_model(self, model_id: str):
        inference_fd = self._acquire_mutation_locks()
        try:
            if model_id != self.sd_manager.model_id:
                resolved, family = self.model_manager.catalog_checkpoint(model_id)
                if not resolved or family not in (SD15, SDXL):
                    raise ValueError("Checkpoint no registrado, incompatible o de arquitectura desconocida.")
                if resolved != self.sd_manager.model_id:
                    if family == SDXL and Path(resolved).is_file():
                        local_sdxl_config(resolved)
                    diagnose("checkpoint_change_start", self.active_pipeline, "busy")
                    self._release_face()
                    if family == self.family == SD15:
                        self.sd_manager.change_model(resolved)
                    else:
                        next_manager = (SDXLPipeline(resolved) if family == SDXL else
                                        ButterVisionPipeline(model_id=resolved, enable_optimizations=True, enable_lcm=False))
                        self.sd_manager.cleanup()
                        self.sd_manager = next_manager
                    self.family = family
                    self.model_state = "selected"
                    self.model_error = ""
                    self.load_budget = None
                    self.lora_registry.clear()
                    self.active_pipeline = "none"
                    config.model_config.model_id = resolved
                    diagnose("checkpoint_change_end", self.active_pipeline, "idle")
            return self.list_models()
        except Exception as error:
            diagnose("memory_allocation_error" if is_memory_error(error) else "checkpoint_change_error",
                     self.active_pipeline, "failed", error_type=type(error).__name__)
            raise
        finally:
            self._release_mutation_locks(inference_fd)

    def create_job(self, mode: str, payload: dict) -> GenerationJob:
        inference_fd = self._acquire_mutation_locks()
        job = None
        persisted = False
        try:
            if self.family not in (SD15, SDXL):
                raise ValueError("Selecciona un checkpoint SD 1.5 o SDXL compatible.")
            experimental = payload.get("engine_mode", "standard") == "pizarra_experimental"
            if experimental:
                allowed = {"engine_mode", "prompt", "negative_prompt", "steps", "cfg_scale", "width", "height", "seed", "batch_size"}
                if set(payload) - allowed:
                    raise ValueError("PIZARRA experimental no admite parámetros adicionales, referencias, img2img ni video.")
                if mode != "txt2img":
                    raise ValueError("PIZARRA experimental solo admite txt2img.")
                if self.family != SDXL:
                    raise ValueError("PIZARRA experimental solo admite modelos SDXL.")
                if self.lora_registry or payload.get("loras") or payload.get("lora"):
                    raise ValueError("PIZARRA experimental no admite LoRA activos.")
                if any(payload.get(key) for key in ("reference_image", "image", "init_image", "video")):
                    raise ValueError("PIZARRA experimental no admite referencias, img2img ni video.")
                if (payload.get("width"), payload.get("height")) not in ((64, 64), (128, 128)):
                    raise ValueError("PIZARRA experimental solo admite 64×64 o 128×128 píxeles.")
                if (payload.get("steps") != 1 or payload.get("cfg_scale") != 1.0 or
                    payload.get("batch_size") != 1 or payload.get("negative_prompt")):
                    raise ValueError("PIZARRA experimental requiere 1 paso, CFG 1, una imagen y ningún prompt negativo.")
            if not experimental and mode == "txt2img" and (payload.get("width", 512) < 256 or payload.get("height", 512) < 256):
                raise ValueError("La resolución mínima es 256 píxeles.")
            if mode == "face_reference" and self.family != SD15:
                raise ValueError("Face Reference solo es compatible con SD 1.5; SDXL aún no tiene backend facial.")
            if mode == "txt2img" and self.family == SDXL and not experimental:
                for dimension in (int(payload.get("width", 768)), int(payload.get("height", 768))):
                    if dimension not in self.resolutions():
                        raise ValueError("Resolución SDXL no disponible para este perfil de hardware.")
                if self._pipeline() is None:
                    try:
                        self.load_budget = check_budget(self.sd_manager.model_id)
                    except Exception as error:
                        try:
                            self.load_budget = inspect_load_budget(self.sd_manager.model_id)
                        except Exception:
                            self.load_budget = None
                        self.model_state = "error"
                        self.model_error = type(error).__name__
                        raise
            job = GenerationJob(id=str(uuid.uuid4()), mode=mode, payload=dict(payload))
            job._defer_completion = self.pizarra_store is not None
            if self.pizarra_store is not None:
                try:
                    self.pizarra_store.create(job.as_dict(), engine=self.family,
                                              model_identifier=Path(self.sd_manager.model_id).name)
                    persisted = True
                    self.pizarra_storage.validate()
                    job._pizarra_start_stats = self.pizarra_storage.stats()
                    self.pizarra_storage.create_workspace(job.id)
                except Exception as error:
                    raise PizarraUnavailableError("PIZARRA no está disponible; verifica el almacenamiento y sus permisos.") from error
            with self.lock:
                self.jobs[job.id] = job
            thread = threading.Thread(target=self._run_job, args=(job, inference_fd), daemon=True)
            thread.start()
            return job
        except Exception as error:
            if job is not None:
                if persisted:
                    job.update(status="failed", progress=1.0, message="Failed",
                               error="No se pudo preparar el trabajo PIZARRA.")
                    try:
                        self.pizarra_store.update(job.as_dict(), failure_category=(
                            "storage_unavailable" if isinstance(error, PizarraUnavailableError) else "worker_start"))
                    except Exception:
                        diagnose("pizarra_metadata_error", self.active_pipeline, "failed")
                with self.lock:
                    self.jobs.pop(job.id, None)
                job.payload.clear()
            diagnose("memory_allocation_error" if is_memory_error(error) else "job_start_error",
                     self.active_pipeline, "failed", error_type=type(error).__name__)
            self._release_mutation_locks(inference_fd)
            raise

    def _acquire_mutation_locks(self):
        if not self.worker_lock.acquire(blocking=False):
            raise EngineBusyError()
        try:
            inference_fd = acquire_inference_lock()
        except Exception:
            self.worker_lock.release()
            raise
        if inference_fd is None:
            self.worker_lock.release()
            raise EngineBusyError()
        return inference_fd

    def _release_mutation_locks(self, inference_fd):
        try:
            release_inference_lock(inference_fd)
        finally:
            self.worker_lock.release()

    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is not None or self.pizarra_store is None:
            return job
        row = self.pizarra_store.get(job_id)
        if row is None:
            return None
        # The payload is deliberately absent from the durable record.
        return GenerationJob(id=row["id"], mode=row["mode"], payload={}, status=row["status"],
                             progress=row["progress"], message=row["message"],
                             result_images=row["result_images"], info=row["info"], error=row["error"],
                             warnings=row["warnings"], created_at=row["created_at"], updated_at=row["updated_at"])

    def get_pizarra_metrics(self, job_id):
        return self.pizarra_store.get_metrics(job_id) if self.pizarra_store is not None else None

    def history(self, limit: int = 24):
        image_paths = list_public_output_images(config.OUTPUTS_DIR, limit)
        return [_public_output_path(path) for path in image_paths[:limit]]

    def _run_job(self, job: GenerationJob, inference_fd=None):
        # create_job reserves worker_lock before starting this worker. Retain it
        # through terminal-state updates and all cleanup, including failures.
        failed = False
        failure_category = None
        metrics = None
        try:
            if self.pizarra_store is not None:
                from core.pizarra.metrics import PizarraMetrics

                metrics = PizarraMetrics(job.id, self.family, enabled=self.pizarra_config.metrics_enabled)
                metrics.start()
            with metrics.phase("preparation") if metrics else nullcontext():
                job.update(status="running", progress=0.05, message="Preparing")
                if self.pizarra_store is not None:
                    self.pizarra_storage.validate()
                    self.pizarra_store.update(job.as_dict())
                diagnose("job_start", self.active_pipeline, "running", job_id=job.id)
            with bind_pizarra_metrics(metrics) if metrics else nullcontext():
                with metrics.phase("pipeline_execution") if metrics else nullcontext():
                    if job.mode == "txt2img":
                        if job.payload.get("engine_mode") == "pizarra_experimental":
                            self._run_pizarra_image(job)
                        else:
                            self._run_txt2img(job)
                    elif job.mode == "face_reference":
                        self._run_face_reference(job)
                    else:
                        raise ValueError(f"Modo no soportado: {job.mode}")
            if self.pizarra_store is not None:
                # A completed image is not a successfully instrumented PIZARRA
                # job if the authorized volume disappeared during inference.
                self.pizarra_storage.validate()
        except Exception as error:
            failed = True
            memory_error = is_memory_error(error)
            if self.pizarra_store is not None:
                from core.pizarra.storage import PizarraStorageError

                if isinstance(error, PizarraStorageError):
                    failure_category = "storage_unavailable"
                elif isinstance(error, sqlite3.Error):
                    failure_category = "metadata_error"
                elif memory_error:
                    failure_category = "memory_allocation"
                else:
                    failure_category = "inference_error"
            diagnose("memory_allocation_error" if memory_error else "inference_error",
                     self.active_pipeline, "failed", error_type=type(error).__name__)
            if failure_category == "storage_unavailable":
                message = "El almacenamiento PIZARRA dejó de estar disponible; el trabajo se detuvo."
            elif failure_category == "metadata_error":
                message = "No se pudo actualizar el registro local del trabajo."
            else:
                message = ("Memoria insuficiente para completar la generación. Libera recursos e intenta nuevamente."
                           if memory_error else "No se pudo completar la generación. Revisa los modelos locales y el registro del motor.")
            if job.payload.get("engine_mode") == "pizarra_experimental" and not memory_error:
                message = str(error) or "Falló la generación PIZARRA experimental."
            job.update(status="failed", progress=1.0, error=message, message="Failed")
            # Inference tracebacks otherwise retain pipelines and device tensors
            # while cleanup runs inside the exception handler.
            traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
        finally:
            try:
                with metrics.phase("cleanup") if metrics else nullcontext():
                    if failed:
                        try:
                            self._release_face()
                        finally:
                            self.sd_manager.cleanup()
                            self.lora_registry.clear()
                            self.active_pipeline = "none"
                    collect_memory()
            except Exception:
                diagnose("cleanup_error", self.active_pipeline, job.status)
                if self.pizarra_store is not None:
                    failure_category = failure_category or "cleanup_error"
                    job.update(status="failed", progress=1.0, message="Failed",
                               error="No se pudo finalizar la limpieza del trabajo PIZARRA.")
            finally:
                # Keep public terminal states, not photos or prompts, in memory.
                job.payload.clear()
                try:
                    if self.pizarra_store is not None:
                        self._finish_pizarra_job(job, metrics, failure_category)
                finally:
                    try:
                        diagnose("job_end", self.active_pipeline, job.status, job_id=job.id)
                    finally:
                        try:
                            release_inference_lock(inference_fd)
                        finally:
                            self.worker_lock.release()

    def _finish_pizarra_job(self, job, metrics, failure_category):
        """Persist a terminal state locally, even when the bank disconnects."""
        storage_stats = None
        try:
            self.pizarra_storage.validate()
            storage_stats = self.pizarra_storage.stats()
            initial = getattr(job, "_pizarra_start_stats", {}) or {}
            # I/O counters belong to this storage instance, over this job's
            # interval; never confuse them with whole-device disk activity.
            for key in ("read_bytes", "written_bytes", "read_seconds", "write_seconds"):
                current, previous = storage_stats.get(key), initial.get(key)
                storage_stats[key] = max(0, current - previous) if current is not None and previous is not None else None
            for amount, elapsed, rate in (("read_bytes", "read_seconds", "read_bytes_per_second"),
                                          ("written_bytes", "write_seconds", "write_bytes_per_second")):
                count, seconds = storage_stats.get(amount), storage_stats.get(elapsed)
                storage_stats[rate] = count / seconds if count is not None and seconds else None
        except Exception:
            failure_category = "storage_unavailable"
            job.update(status="failed", progress=1.0, message="Failed",
                       error="El almacenamiento PIZARRA dejó de estar disponible; el trabajo se detuvo.")
        if job.status != "failed" and job._pending_completion is None:
            failure_category = failure_category or "inference_error"
            job.update(status="failed", progress=1.0, message="Failed",
                       error="El trabajo terminó sin confirmar un resultado.")
        snapshot = job.terminal_snapshot()
        report = None
        try:
            if metrics is not None:
                report = metrics.finish(snapshot["status"], storage_stats=storage_stats)
        except Exception:
            failure_category = failure_category or "metrics_error"
            job.update(status="failed", progress=1.0, message="Failed",
                       error="No se pudieron finalizar las métricas PIZARRA.")
            snapshot = job.as_dict()
        try:
            self.pizarra_store.finalize(snapshot, report, failure_category=failure_category)
        except Exception:
            # SQLite is local to the application. If it fails, do not report a
            # false successful job, and make a best effort to retain the error.
            job.update(status="failed", progress=1.0, message="Failed",
                       error="No se pudo guardar el estado local del trabajo PIZARRA.")
            diagnose("pizarra_metadata_error", self.active_pipeline, "failed")
            try:
                self.pizarra_store.update(job.as_dict(), failure_category="metadata_error")
            except Exception:
                pass
        else:
            if snapshot["status"] == "completed":
                job.publish_completion()

    def _release_face(self):
        if self.face_manager is not None:
            manager, self.face_manager = self.face_manager, None
            try:
                manager.cleanup()
            finally:
                del manager
                self.active_pipeline = "none"
                collect_memory()

    def _normalized_common(self, payload: dict):
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Prompt requerido.")
        negative_prompt = (payload.get("negative_prompt") or "").strip()
        seed = int(payload.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        max_width = max(self.resolutions()) if self.family == SDXL else config.model_config.max_width
        max_height = max(self.resolutions()) if self.family == SDXL else config.model_config.max_height
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": int(payload.get("steps", config.model_config.default_steps)),
            "cfg_scale": float(payload.get("cfg_scale", config.model_config.default_cfg_scale)),
            "width": max(256, min(int(payload.get("width", config.model_config.default_width)), max_width)),
            "height": max(256, min(int(payload.get("height", config.model_config.default_height)), max_height)),
            "batch_size": max(1, min(int(payload.get("batch_size", 1)), config.model_config.max_batch_size)),
            "seed": seed,
        }

    def _run_pizarra_image(self, job: GenerationJob):
        from core.pizarra.sdxl_image_service import run_image
        import secrets

        data = dict(job.payload)
        data["seed"] = data.get("seed", -1)
        if data["seed"] == -1:
            data["seed"] = secrets.randbelow(2**32)
        self._release_face()
        self.sd_manager.cleanup()
        self.active_pipeline = "none"
        job.update(progress=0.15, message="Generando con PIZARRA experimental en CPU")
        image = run_image(self.sd_manager.model_id, data["prompt"], data["width"], data["seed"])
        paths, generation_dir = self._save_generation([image], {**data, "module": "txt2img", "model": self.sd_manager.model_id})
        self.model_state = "selected"
        job.update(status="completed", progress=1.0, message="Completed",
                   result_images=[_public_output_path(path) for path in paths],
                   info=self._format_info(data, generation_dir))

    def _run_txt2img(self, job: GenerationJob):
        data = self._normalized_common(job.payload)
        self._release_face()
        self.active_pipeline = "txt2img"
        job.update(progress=0.15, message="Loading model")
        self.model_state = "loading"
        try:
            if self.family == SD15:
                # The legacy automatic LoRA may be installed during pipeline load.
                # Publish it before inference so polling reflects the active state.
                self.sd_manager.load_sd_pipeline()
                self._sync_legacy_loras()
            elif self.family == SDXL:
                self.sd_manager.load_pipeline()
            self.model_state = "loaded"
            self.model_error = ""
            job.update(progress=0.2, message="Generating image")
            self.model_state = "generating"
            images = self.sd_manager.generate_image(
                prompt=data["prompt"],
                negative_prompt=data["negative_prompt"],
                num_inference_steps=data["steps"],
                guidance_scale=data["cfg_scale"],
                width=data["width"],
                height=data["height"],
                seed=data["seed"],
                num_images=data["batch_size"],
                prompt_warning_callback=job.warn,
            )
            self.model_state = "loaded"
        except Exception as error:
            self.model_state = "error"
            self.model_error = type(error).__name__
            raise
        self._sync_legacy_loras()
        self.model_state = "loaded"
        paths, generation_dir = self._save_generation(images, {**data, "module": "txt2img", "model": self.sd_manager.model_id})
        info = self._format_info(data, generation_dir)
        job.update(
            status="completed",
            progress=1.0,
            message="Completed",
            result_images=[_public_output_path(path) for path in paths],
            info=info,
        )

    def _run_face_reference(self, job: GenerationJob):
        data = self._normalized_common(job.payload)
        data["width"] = max(512, min(data["width"], config.model_config.face_max_width))
        data["height"] = max(512, min(data["height"], config.model_config.face_max_height))
        data["batch_size"] = 1
        reference_image_data = job.payload.pop("reference_image")
        reference_image = decode_reference_image(reference_image_data)
        del reference_image_data
        identity_strength = float(job.payload.get("identity_strength", 0.8))
        structure_strength = float(job.payload.get("structure_strength", 0.8))

        job.update(progress=0.15, message="Loading Face Reference")
        if self.active_pipeline != "face_reference":
            self.sd_manager.cleanup()
            self.lora_registry.clear()
        if self.face_manager is None:
            self.face_manager = SD15FaceReferencePipeline(base_model=self.sd_manager.model_id)
        self.active_pipeline = "face_reference"

        job.update(progress=0.25, message="Generating face reference")
        images = self.face_manager.generate(
            face_image=reference_image,
            prompt=data["prompt"],
            negative_prompt=data["negative_prompt"],
            width=data["width"],
            height=data["height"],
            steps=data["steps"],
            guidance_scale=data["cfg_scale"],
            seed=data["seed"],
            num_images=1,
            identity_strength=identity_strength,
            structure_strength=structure_strength,
            prompt_warning_callback=job.warn,
        )
        paths, generation_dir = self._save_generation(
            images,
            {
                **data,
                "module": "face_reference",
                "backend": "SD1.5 IP-Adapter Face",
                "identity_strength": identity_strength,
                "structure_strength": structure_strength,
                "model": self.face_manager.base_model,
            },
        )
        info = self._format_info(data, generation_dir, mode="Face Reference / SD1.5 IP-Adapter Face")
        job.update(
            status="completed",
            progress=1.0,
            message="Completed",
            result_images=[_public_output_path(path) for path in paths],
            info=info,
        )

    def _save_generation(self, images, metadata):
        with measured_phase("saving"):
            return self._write_generation(images, metadata)

    def _write_generation(self, images, metadata):
        created_at = datetime.now()
        base_name = created_at.strftime("%d%m%Y-%H%M%S-generation")
        generation_dir = config.OUTPUTS_DIR / base_name
        suffix = 2
        while generation_dir.exists():
            generation_dir = config.OUTPUTS_DIR / f"{base_name}-{suffix}"
            suffix += 1
        generation_dir.mkdir(parents=True, exist_ok=False)

        metadata = {
            **metadata,
            "created_at": created_at.isoformat(timespec="seconds"),
            "generation_dir": str(generation_dir),
            "images": [],
            "reference_images": [],
        }

        png_info = PngInfo()
        png_info.add_text("ButterVision", metadata["module"])
        png_info.add_text("Generation Metadata", json.dumps(metadata, ensure_ascii=False))

        paths = []
        for index, image in enumerate(images, start=1):
            path = generation_dir / f"image_{index:02d}.png"
            image.save(path, pnginfo=png_info)
            paths.append(path)
            metadata["images"].append(str(path))

        (generation_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return paths, generation_dir

    def _format_info(self, data, generation_dir, mode="Text to Image"):
        return (
            f"Mode: {mode}\n"
            f"Seed: {data['seed']}\n"
            f"Size: {data['width']}x{data['height']}\n"
            f"Steps: {data['steps']} | CFG: {data['cfg_scale']:.1f}\n"
            f"Batch: {data['batch_size']}\n"
            f"Saved: {generation_dir}"
        )
