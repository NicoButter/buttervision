from pathlib import Path
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from api.output_files import is_private_output_file
from api.uploads import UploadValidationError, validate_reference_image
from api.schemas import (
    FaceReferenceRequest,
    JobCreateResponse,
    JobStatusResponse,
    RuntimeInfo,
    Txt2ImgRequest,
    LoraUpdate,
)


service = None
service_lock = threading.Lock()


def get_service():
    global service
    with service_lock:
        if service is None:
            from api.generation import GenerationService, PizarraUnavailableError

            try:
                service = GenerationService()
            except PizarraUnavailableError as error:
                raise HTTPException(status_code=503, detail="No se pudo abrir el registro local PIZARRA.") from error
        return service


def engine_call(method, *args):
    from api.generation import EngineBusyError, PizarraUnavailableError

    try:
        return method(*args)
    except EngineBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except PizarraUnavailableError as error:
        raise HTTPException(status_code=503, detail="PIZARRA no está disponible; verifica el volumen y el registro local.") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

app = FastAPI(title="ButterVision API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


class PrivateOutputFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        if is_private_output_file(path):
            raise HTTPException(status_code=404, detail="Archivo no disponible.")
        return await super().get_response(path, scope)


app.mount("/outputs", PrivateOutputFiles(directory=str(config.OUTPUTS_DIR)), name="outputs")


@app.get("/api/runtime", response_model=RuntimeInfo)
def runtime_info():
    return {
        "profile": config.model_config.hardware_profile,
        "cuda_available": None,
        "device_name": "Runtime checked during generation",
        "defaults": {
            "steps": config.model_config.default_steps,
            "cfg_scale": config.model_config.default_cfg_scale,
            "width": config.model_config.default_width,
            "height": config.model_config.default_height,
            "batch_size": config.model_config.default_batch_size,
            "face_steps": config.model_config.face_default_steps,
            "face_width": config.model_config.face_default_width,
            "face_height": config.model_config.face_default_height,
        },
        "upload_limits": {
            "image_bytes": config.MAX_IMAGE_UPLOAD_BYTES,
            "video_bytes": config.MAX_VIDEO_UPLOAD_BYTES,
        },
    }


@app.get("/api/models")
def list_models():
    return get_service().list_models()


@app.post("/api/models/active")
def set_active_model(payload: dict):
    model_id = payload.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id requerido")
    return engine_call(get_service().set_model, model_id)


@app.get("/api/engine")
def engine_status():
    return get_service().engine_state()


@app.get("/api/loras")
def list_loras():
    return get_service().list_loras()


@app.post("/api/loras/active")
def set_lora(payload: LoraUpdate):
    return engine_call(get_service().set_lora, payload.path, payload.enabled, payload.weight)


@app.post("/api/generate/txt2img", response_model=JobCreateResponse)
def generate_txt2img(payload: Txt2ImgRequest):
    job = engine_call(get_service().create_job, "txt2img", payload.model_dump())
    return {"job_id": job.id, "status": job.status}


@app.post("/api/generate/face-reference", response_model=JobCreateResponse)
def generate_face_reference(payload: FaceReferenceRequest):
    try:
        validate_reference_image(payload.reference_image)
    except UploadValidationError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    job = engine_call(get_service().create_job, "face_reference", payload.model_dump())
    return {"job_id": job.id, "status": job.status}


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    job = get_service().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return job.as_dict()


@app.get("/api/history")
def history(limit: int = 24):
    return {"images": get_service().history(limit=limit)}


@app.get("/api/pizarra/status")
def pizarra_status(include_budget: bool = False):
    engine = get_service()
    if not engine.pizarra_config.enabled:
        return {"enabled": False, "available": None, "metrics_enabled": False, "storage": None}
    try:
        stats = engine.pizarra_storage.stats()
        # Public observability is a whitelist: never return root/device paths.
        fields = ("total_bytes", "free_bytes", "managed_bytes", "quota_bytes",
                  "quota_remaining_bytes", "min_free_bytes", "available_bytes")
        result = {"enabled": True, "available": True,
                  "metrics_enabled": engine.pizarra_config.metrics_enabled,
                  "storage": {key: stats.get(key) for key in fields}}
    except Exception:
        result = {"enabled": True, "available": False,
                  "metrics_enabled": engine.pizarra_config.metrics_enabled,
                  "storage": None, "message": "Volumen PIZARRA no disponible o no autorizado."}
    if include_budget:
        from core.pizarra.memory_budget import inspect_budget
        try:
            result["budget"] = inspect_budget("generation", storage=engine.pizarra_storage)
        except ValueError:
            result["budget"] = None
    return result


@app.get("/api/pizarra/jobs/{job_id}/metrics")
def pizarra_job_metrics(job_id: str):
    try:
        result = get_service().get_pizarra_metrics(job_id)
    except Exception as error:
        raise HTTPException(status_code=503, detail="No se pudo consultar el registro local de métricas.") from error
    if result is None:
        raise HTTPException(status_code=404, detail="Métricas no disponibles para este trabajo.")
    return result


WEB_DIST = config.PROJECT_ROOT / "web" / "dist"


@app.get("/{full_path:path}")
def serve_frontend(full_path: str):
    index = WEB_DIST / "index.html"
    requested = WEB_DIST / full_path
    if full_path and requested.exists() and requested.is_file():
        return FileResponse(requested)
    if index.exists():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="Frontend no compilado. Ejecuta `npm run build` en web/.")
