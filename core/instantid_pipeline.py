"""
InstantID backend for identity-preserving face reference generation.

This module is intentionally lazy: heavy dependencies and SDXL models are only
imported when the Face Reference workflow is used.
"""
from pathlib import Path
import gc
import importlib.util
import sys
import zipfile
from urllib.request import urlretrieve
from typing import List, Optional

from PIL import Image
from PIL import ImageDraw

import config


class InstantIDPipeline:
    """Thin wrapper around the InstantID SDXL pipeline."""

    def __init__(
        self,
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        checkpoint_dir: Optional[Path] = None,
        face_model_root: Optional[Path] = None,
        device: str = "cuda",
    ):
        self.base_model = base_model
        self.checkpoint_dir = Path(checkpoint_dir or config.MODELS_DIR / "instantid")
        self.face_model_root = Path(face_model_root or config.PROJECT_ROOT)
        self.face_model_dir = config.MODELS_DIR / "antelopev2"
        self.pipeline_file = self.checkpoint_dir / "pipeline_stable_diffusion_xl_instantid.py"
        self.ip_adapter_dir = self.checkpoint_dir / "ip_adapter"
        self.device = device if self._cuda_is_available() else "cpu"

        self.pipe = None
        self.face_app = None

    def cleanup(self):
        """Libera InstantID de memoria."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None

        gc.collect()
        try:
            torch = self._load_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except RuntimeError:
            pass

    def _load_torch(self):
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "Falta instalar PyTorch. Instala las dependencias base antes de generar con InstantID."
            ) from error

        return torch

    def _cuda_is_available(self):
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def ensure_assets(self, allow_download: bool = True) -> str:
        """Ensures InstantID and InsightFace model files exist locally."""
        controlnet_path, adapter_path = self._checkpoint_paths()
        instantid_ready = controlnet_path.exists() and adapter_path.exists()
        code_ready = self.pipeline_file.exists() and self.ip_adapter_dir.exists()
        face_ready = self.face_model_dir.exists() and any(self.face_model_dir.glob("*.onnx"))

        if instantid_ready and code_ready and face_ready:
            return "InstantID listo: modelos locales encontrados."

        if not allow_download:
            self._validate_assets()
            return "InstantID listo."

        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError(
                "No está instalado `huggingface_hub`; no puedo descargar modelos automáticamente."
            ) from error

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if not instantid_ready:
            print("📥 Descargando modelos InstantID...")
            snapshot_download(
                repo_id="InstantX/InstantID",
                local_dir=str(self.checkpoint_dir),
                local_dir_use_symlinks=False,
                allow_patterns=[
                    "ControlNetModel/*",
                    "ip-adapter.bin",
                ],
            )

        if not code_ready:
            self._download_official_code()

        if not face_ready:
            print("📥 Descargando InsightFace antelopev2...")
            snapshot_download(
                repo_id="Aitrepreneur/insightface",
                local_dir=str(config.PROJECT_ROOT),
                local_dir_use_symlinks=False,
                allow_patterns=[
                    "models/antelopev2/*.onnx",
                ],
            )
            if not self.face_model_dir.exists():
                raise RuntimeError(
                    "La descarga de antelopev2 terminó, pero no encontré los archivos ONNX esperados."
                )

        self._validate_assets()
        return "InstantID listo: modelos descargados correctamente."

    def _download_official_code(self):
        """Descarga el pipeline oficial y su paquete local ip_adapter."""
        print("📥 Descargando código oficial InstantID...")
        code_zip = config.CACHE_DIR / "instantid_official_code.zip"
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        urlretrieve(
            "https://github.com/instantX-research/InstantID/archive/refs/heads/main.zip",
            code_zip,
        )

        with zipfile.ZipFile(code_zip) as archive:
            for member in archive.namelist():
                parts = Path(member).parts
                if len(parts) < 2:
                    continue

                relative = Path(*parts[1:])
                if relative == Path("pipeline_stable_diffusion_xl_instantid.py"):
                    self.pipeline_file.write_bytes(archive.read(member))
                elif relative.parts[:1] == ("ip_adapter",) and not member.endswith("/"):
                    target = self.checkpoint_dir / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(member))

    def _load_dependencies(self):
        try:
            import cv2
            import numpy as np
            from diffusers import ControlNetModel
            from insightface.app import FaceAnalysis
        except ImportError as error:
            raise RuntimeError(
                "InstantID no está instalado todavía. Ejecuta `./install_face_reference.sh` "
                "y reinicia ButterVision. Esta dependencia queda separada del install base "
                "porque insightface/onnxruntime-gpu son paquetes pesados y específicos de "
                "Face Reference."
            ) from error

        return cv2, np, ControlNetModel, FaceAnalysis

    def _load_instantid_pipeline_class(self):
        if not self.pipeline_file.exists() or not self.ip_adapter_dir.exists():
            self.ensure_assets(allow_download=True)

        checkpoint_dir = str(self.checkpoint_dir)
        if checkpoint_dir not in sys.path:
            sys.path.insert(0, checkpoint_dir)

        spec = importlib.util.spec_from_file_location(
            "buttervision_instantid_pipeline",
            self.pipeline_file,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"No pude cargar el pipeline InstantID desde {self.pipeline_file}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.StableDiffusionXLInstantIDPipeline

    def _checkpoint_paths(self):
        controlnet_path = self.checkpoint_dir / "ControlNetModel"
        adapter_path = self.checkpoint_dir / "ip-adapter.bin"
        return controlnet_path, adapter_path

    def _validate_assets(self):
        controlnet_path, adapter_path = self._checkpoint_paths()
        missing = []

        if not controlnet_path.exists():
            missing.append(str(controlnet_path))
        if not adapter_path.exists():
            missing.append(str(adapter_path))
        if not self.pipeline_file.exists():
            missing.append(str(self.pipeline_file))
        if not self.ip_adapter_dir.exists():
            missing.append(str(self.ip_adapter_dir))
        if not self.face_model_dir.exists():
            missing.append(str(self.face_model_dir))

        if missing:
            raise RuntimeError(
                "Faltan modelos de InstantID/InsightFace:\n"
                + "\n".join(f"- {path}" for path in missing)
                + "\n\nUbicación esperada: models/instantid/ y models/antelopev2/."
            )

    def _load_face_app(self, FaceAnalysis):
        if self.face_app is not None:
            return self.face_app

        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            providers.insert(0, "CUDAExecutionProvider")

        self.face_app = FaceAnalysis(
            name="antelopev2",
            root=str(self.face_model_root),
            providers=providers,
        )
        self.face_app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))
        return self.face_app

    def _load_pipeline(self, ControlNetModel):
        if self.pipe is not None:
            return self.pipe

        torch = self._load_torch()
        StableDiffusionXLInstantIDPipeline = self._load_instantid_pipeline_class()
        controlnet_path, adapter_path = self._checkpoint_paths()
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        controlnet = ControlNetModel.from_pretrained(
            str(controlnet_path),
            torch_dtype=torch_dtype,
        )
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            self.base_model,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            cache_dir=str(config.model_config.cache_dir),
        )
        pipe.load_ip_adapter_instantid(str(adapter_path))

        if self.device == "cuda":
            try:
                pipe.enable_attention_slicing(slice_size="max")
            except Exception:
                pass

            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass

            try:
                pipe.vae.enable_tiling()
            except Exception:
                try:
                    pipe.enable_vae_tiling()
                except Exception:
                    pass

            try:
                pipe.enable_sequential_cpu_offload()
            except Exception:
                pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cpu")

        self.pipe = pipe
        return self.pipe

    def _extract_face(self, face_image, cv2, np):
        face_app = self.face_app
        faces = face_app.get(cv2.cvtColor(np.array(face_image.convert("RGB")), cv2.COLOR_RGB2BGR))
        if not faces:
            raise RuntimeError("No se detectó ninguna cara clara en la imagen de referencia.")

        return sorted(
            faces,
            key=lambda face: (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1]),
        )[-1]

    def _draw_kps(self, image, keypoints):
        """Dibuja landmarks faciales para condicionar IdentityNet."""
        output = Image.new("RGB", image.size, "black")
        draw = ImageDraw.Draw(output)
        points = [(int(x), int(y)) for x, y in keypoints]

        for point in points:
            x, y = point
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 255, 255))

        if len(points) >= 5:
            left_eye, right_eye, nose, left_mouth, right_mouth = points[:5]
            draw.line([left_eye, nose, right_eye], fill=(255, 255, 255), width=3)
            draw.line([left_mouth, nose, right_mouth], fill=(255, 255, 255), width=3)

        return output

    def generate(
        self,
        face_image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int = -1,
        num_images: int = 1,
        identity_strength: float = 0.8,
        structure_strength: float = 0.8,
    ) -> List[Image.Image]:
        torch = self._load_torch()
        cv2, np, ControlNetModel, FaceAnalysis = self._load_dependencies()
        self.ensure_assets(allow_download=True)

        self._load_face_app(FaceAnalysis)
        pipe = self._load_pipeline(ControlNetModel)

        face = self._extract_face(face_image, cv2, np)
        face_emb = face["embedding"]
        face_kps = self._draw_kps(face_image.convert("RGB"), face["kps"])

        generator = None
        if seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=structure_strength,
            ip_adapter_scale=identity_strength,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result.images
