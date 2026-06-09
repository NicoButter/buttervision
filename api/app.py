from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from api.schemas import (
    FaceReferenceRequest,
    JobCreateResponse,
    JobStatusResponse,
    RuntimeInfo,
    Txt2ImgRequest,
)


service = None


def get_service():
    global service
    if service is None:
        from api.generation import GenerationService

        service = GenerationService()
    return service

app = FastAPI(title="ButterVision API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(config.OUTPUTS_DIR)), name="outputs")


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
    }


@app.get("/api/models")
def list_models():
    return get_service().list_models()


@app.post("/api/models/active")
def set_active_model(payload: dict):
    model_id = payload.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id requerido")
    return get_service().set_model(model_id)


@app.post("/api/generate/txt2img", response_model=JobCreateResponse)
def generate_txt2img(payload: Txt2ImgRequest):
    job = get_service().create_job("txt2img", payload.model_dump())
    return {"job_id": job.id, "status": job.status}


@app.post("/api/generate/face-reference", response_model=JobCreateResponse)
def generate_face_reference(payload: FaceReferenceRequest):
    job = get_service().create_job("face_reference", payload.model_dump())
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
