import base64
import gc
import json
import random
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import config
from core.advanced_pipeline import ButterVisionPipeline
from core.face_reference_pipeline import SD15FaceReferencePipeline
from core.model_manager import ModelManager


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _decode_data_image(data_url: str) -> Image.Image:
    if "," in data_url:
        _, data = data_url.split(",", 1)
    else:
        data = data_url
    image_bytes = base64.b64decode(data)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


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

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.updated_at = _now()

    def as_dict(self):
        return {
            "id": self.id,
            "mode": self.mode,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "result_images": self.result_images,
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
        self.sd_manager = ButterVisionPipeline(
            model_id=active_model,
            enable_optimizations=True,
            enable_lcm=False,
        )
        self.face_manager: Optional[SD15FaceReferencePipeline] = None
        self.jobs: Dict[str, GenerationJob] = {}
        self.lock = threading.Lock()
        self.worker_lock = threading.Lock()

    def list_models(self):
        model_infos = self.model_manager.list_local_model_infos()
        active = self.sd_manager.model_id
        models = []
        for info in model_infos:
            value = info["path"]
            models.append({"label": Path(value).name, "value": value, "active": value == active})
        if active and all(model["value"] != active for model in models):
            models.insert(0, {"label": Path(active).name, "value": active, "active": True})
        return models

    def set_model(self, model_id: str):
        with self.lock:
            if model_id != self.sd_manager.model_id:
                self.sd_manager.change_model(model_id)
                config.model_config.model_id = model_id
                if self.face_manager is not None:
                    self.face_manager.cleanup()
                    self.face_manager = None
        return self.list_models()

    def create_job(self, mode: str, payload: dict) -> GenerationJob:
        job = GenerationJob(id=str(uuid.uuid4()), mode=mode, payload=payload)
        self.jobs[job.id] = job
        thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        thread.start()
        return job

    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        return self.jobs.get(job_id)

    def history(self, limit: int = 24):
        if not config.OUTPUTS_DIR.exists():
            return []
        image_paths = []
        for directory in config.OUTPUTS_DIR.iterdir():
            if directory.is_dir():
                image_paths.extend(directory.glob("*.png"))
        image_paths = sorted(image_paths, key=lambda path: path.stat().st_mtime, reverse=True)
        return [_public_output_path(path) for path in image_paths[:limit]]

    def _run_job(self, job: GenerationJob):
        if not self.worker_lock.acquire(blocking=False):
            job.update(status="failed", error="Ya hay una generación en curso.", message="Busy")
            return
        try:
            job.update(status="running", progress=0.05, message="Preparing")
            if job.mode == "txt2img":
                self._run_txt2img(job)
            elif job.mode == "face_reference":
                self._run_face_reference(job)
            else:
                raise ValueError(f"Modo no soportado: {job.mode}")
        except Exception as error:
            job.update(status="failed", progress=1.0, error=str(error), message="Failed")
        finally:
            self.worker_lock.release()
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _normalized_common(self, payload: dict):
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Prompt requerido.")
        negative_prompt = (payload.get("negative_prompt") or "").strip()
        seed = int(payload.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": int(payload.get("steps", config.model_config.default_steps)),
            "cfg_scale": float(payload.get("cfg_scale", config.model_config.default_cfg_scale)),
            "width": max(256, min(int(payload.get("width", config.model_config.default_width)), config.model_config.max_width)),
            "height": max(256, min(int(payload.get("height", config.model_config.default_height)), config.model_config.max_height)),
            "batch_size": max(1, min(int(payload.get("batch_size", 1)), config.model_config.max_batch_size)),
            "seed": seed,
        }

    def _run_txt2img(self, job: GenerationJob):
        data = self._normalized_common(job.payload)
        job.update(progress=0.2, message="Generating image")
        images = self.sd_manager.generate_image(
            prompt=data["prompt"],
            negative_prompt=data["negative_prompt"],
            num_inference_steps=data["steps"],
            guidance_scale=data["cfg_scale"],
            width=data["width"],
            height=data["height"],
            seed=data["seed"],
            num_images=data["batch_size"],
        )
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
        reference_image = _decode_data_image(job.payload["reference_image"])
        identity_strength = float(job.payload.get("identity_strength", 0.8))
        structure_strength = float(job.payload.get("structure_strength", 0.8))

        job.update(progress=0.15, message="Loading Face Reference")
        self.sd_manager.cleanup()
        if self.face_manager is None:
            self.face_manager = SD15FaceReferencePipeline(base_model=self.sd_manager.model_id)

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
            reference_images={"reference_face.png": reference_image},
        )
        info = self._format_info(data, generation_dir, mode="Face Reference / SD1.5 IP-Adapter Face")
        job.update(
            status="completed",
            progress=1.0,
            message="Completed",
            result_images=[_public_output_path(path) for path in paths],
            info=info,
        )

    def _save_generation(self, images, metadata, reference_images=None):
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

        for name, image in (reference_images or {}).items():
            path = generation_dir / name
            image.save(path)
            metadata["reference_images"].append(str(path))

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
