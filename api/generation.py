import json
import random
import threading
import traceback
import uuid
import hashlib
import math
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
from core.engine_memory import collect_memory, diagnose, is_memory_error
from core.model_catalog import SD15, SDXL, UNKNOWN, approved_path, checkpoint_family, entries, lora_family
from core.sdxl_pipeline import SDXLPipeline, check_budget, local_sdxl_config


class EngineBusyError(RuntimeError):
    def __init__(self):
        super().__init__("El motor está ocupado generando o cambiando de modelo. Intenta nuevamente cuando termine.")


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

    def update(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.updated_at = _now()

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

    def list_models(self):
        model_infos = self.model_manager.list_local_model_infos()
        active = self.sd_manager.model_id
        models = []
        for info in model_infos:
            value = info["path"]
            models.append({"label": Path(value).name, "value": value, "active": value == active,
                           "family": info.get("family", UNKNOWN)})
        if active and all(model["value"] != active for model in models):
            models.insert(0, {"label": Path(active).name, "value": active, "active": True, "family": self.family})
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
        return {"status": "busy" if self.worker_lock.locked() else "idle",
                "pipeline": self.active_pipeline, "model": Path(self.sd_manager.model_id).name,
                "family": self.family, "resolutions": self.resolutions(),
                "active_loras": [item for item in loras if item["active"]],
                "face_supported": self.family == SD15}

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
        if not self.worker_lock.acquire(blocking=False):
            raise EngineBusyError()
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
                check_budget(self.sd_manager.model_id)
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
            self.worker_lock.release()

    def set_model(self, model_id: str):
        if not self.worker_lock.acquire(blocking=False):
            raise EngineBusyError()
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
            self.worker_lock.release()

    def create_job(self, mode: str, payload: dict) -> GenerationJob:
        if not self.worker_lock.acquire(blocking=False):
            raise EngineBusyError()
        job = None
        try:
            if self.family not in (SD15, SDXL):
                raise ValueError("Selecciona un checkpoint SD 1.5 o SDXL compatible.")
            if mode == "face_reference" and self.family != SD15:
                raise ValueError("Face Reference solo es compatible con SD 1.5; SDXL aún no tiene backend facial.")
            if mode == "txt2img" and self.family == SDXL:
                for dimension in (int(payload.get("width", 768)), int(payload.get("height", 768))):
                    if dimension not in self.resolutions():
                        raise ValueError("Resolución SDXL no disponible para este perfil de hardware.")
                if self._pipeline() is None:
                    check_budget(self.sd_manager.model_id)
            job = GenerationJob(id=str(uuid.uuid4()), mode=mode, payload=dict(payload))
            with self.lock:
                self.jobs[job.id] = job
            thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
            thread.start()
            return job
        except Exception as error:
            if job is not None:
                with self.lock:
                    self.jobs.pop(job.id, None)
                job.payload.clear()
            diagnose("memory_allocation_error" if is_memory_error(error) else "job_start_error",
                     self.active_pipeline, "failed", error_type=type(error).__name__)
            self.worker_lock.release()
            raise

    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        with self.lock:
            return self.jobs.get(job_id)

    def history(self, limit: int = 24):
        image_paths = list_public_output_images(config.OUTPUTS_DIR, limit)
        return [_public_output_path(path) for path in image_paths[:limit]]

    def _run_job(self, job: GenerationJob):
        # create_job reserves worker_lock before starting this worker. Retain it
        # through terminal-state updates and all cleanup, including failures.
        failed = False
        try:
            job.update(status="running", progress=0.05, message="Preparing")
            diagnose("job_start", self.active_pipeline, "running", job_id=job.id)
            if job.mode == "txt2img":
                self._run_txt2img(job)
            elif job.mode == "face_reference":
                self._run_face_reference(job)
            else:
                raise ValueError(f"Modo no soportado: {job.mode}")
        except Exception as error:
            failed = True
            memory_error = is_memory_error(error)
            diagnose("memory_allocation_error" if memory_error else "inference_error",
                     self.active_pipeline, "failed", error_type=type(error).__name__)
            message = ("Memoria insuficiente para completar la generación. Libera recursos e intenta nuevamente."
                       if memory_error else "No se pudo completar la generación. Revisa los modelos locales y el registro del motor.")
            job.update(status="failed", progress=1.0, error=message, message="Failed")
            # Inference tracebacks otherwise retain pipelines and device tensors
            # while cleanup runs inside the exception handler.
            traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
        finally:
            try:
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
            finally:
                # Keep public terminal states, not photos or prompts, in memory.
                job.payload.clear()
                diagnose("job_end", self.active_pipeline, job.status, job_id=job.id)
                self.worker_lock.release()

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

    def _run_txt2img(self, job: GenerationJob):
        data = self._normalized_common(job.payload)
        self._release_face()
        self.active_pipeline = "txt2img"
        job.update(progress=0.2, message="Generating image")
        if self.family == SD15:
            # The legacy automatic LoRA may be installed during pipeline load.
            # Publish it before inference so polling reflects the active state.
            self.sd_manager.load_sd_pipeline()
            self._sync_legacy_loras()
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
        self._sync_legacy_loras()
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
