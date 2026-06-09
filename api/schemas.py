from typing import List, Optional

from pydantic import BaseModel, Field


class Txt2ImgRequest(BaseModel):
    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    steps: int = Field(default=20, ge=1, le=60)
    cfg_scale: float = Field(default=5.0, ge=0.0, le=20.0)
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=512, ge=256, le=1024)
    seed: int = -1
    batch_size: int = Field(default=1, ge=1, le=4)


class FaceReferenceRequest(Txt2ImgRequest):
    reference_image: str = Field(min_length=1)
    identity_strength: float = Field(default=0.8, ge=0.0, le=1.5)
    structure_strength: float = Field(default=0.8, ge=0.0, le=1.5)


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    id: str
    mode: str
    status: str
    progress: float
    message: str
    result_images: List[str] = []
    info: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class ModelInfo(BaseModel):
    label: str
    value: str
    active: bool = False


class RuntimeInfo(BaseModel):
    profile: str
    cuda_available: Optional[bool]
    device_name: Optional[str]
    defaults: dict
