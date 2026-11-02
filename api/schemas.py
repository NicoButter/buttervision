from typing import List, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict, model_validator


class Txt2ImgRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    engine_mode: Literal["standard", "pizarra_experimental"] = "standard"
    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    steps: int = Field(default=20, ge=1, le=60)
    cfg_scale: float = Field(default=5.0, ge=0.0, le=20.0)
    width: int = Field(default=512, ge=64, le=1024)
    height: int = Field(default=512, ge=64, le=1024)
    seed: int = -1
    batch_size: int = Field(default=1, ge=1, le=4)

    @model_validator(mode="after")
    def validate_standard_size(self):
        if self.engine_mode == "standard" and (self.width < 256 or self.height < 256):
            raise ValueError("La resolución mínima es 256 píxeles.")
        return self


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
    warnings: List[str] = Field(default_factory=list)
    created_at: str
    updated_at: str


class ModelInfo(BaseModel):
    label: str
    value: str
    active: bool = False


class LoraUpdate(BaseModel):
    path: str = Field(min_length=1)
    enabled: bool
    weight: float = Field(default=1.0, ge=0, le=2, allow_inf_nan=False)


class RuntimeInfo(BaseModel):
    profile: str
    cuda_available: Optional[bool]
    device_name: Optional[str]
    defaults: dict
    upload_limits: dict
