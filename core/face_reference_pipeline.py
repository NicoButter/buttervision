"""
Forge-compatible face reference backend.

Uses SD 1.5 + IP-Adapter Plus Face, matching the workflow that works in Forge:
InsightFace/CLIP-H (IPAdapter) + ip-adapter-plus-face_sd15.
"""
import gc
from pathlib import Path
from typing import List, Optional

from PIL import Image

import config


class SD15FaceReferencePipeline:
    """Low-VRAM SD1.5 face reference pipeline using IP-Adapter Plus Face."""

    def __init__(
        self,
        base_model: str,
        adapter_path: Optional[Path] = None,
        clip_vision_path: Optional[Path] = None,
        device: str = "cuda",
    ):
        self.base_model = str(base_model)
        self.adapter_path = Path(
            adapter_path
            or "/home/lordcommander/stable-diffusion-webui-forge/models/ControlNet/ip-adapter-plus-face_sd15.safetensors"
        )
        self.clip_vision_path = Path(
            clip_vision_path
            or "/home/lordcommander/stable-diffusion-webui-forge/models/ControlNetPreprocessor/CLIP-ViT-H-14.safetensors"
        )
        self.device = device if self._cuda_is_available() else "cpu"
        self.pipe = None
        self.image_encoder = None

    def cleanup(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if self.image_encoder is not None:
            del self.image_encoder
            self.image_encoder = None

        gc.collect()
        try:
            torch = self._load_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except RuntimeError:
            pass

    def ensure_assets(self, allow_download: bool = True) -> str:
        missing = [
            str(path)
            for path in [Path(self.base_model), self.adapter_path, self.clip_vision_path]
            if not path.exists()
        ]
        if missing:
            raise RuntimeError(
                "Faltan archivos para Face Reference SD1.5/IP-Adapter:\n"
                + "\n".join(f"- {path}" for path in missing)
                + "\n\nButterVision usa el mismo IP-Adapter Face que tu Forge local."
            )

        return "Face Reference listo: SD1.5 + IP-Adapter Face local."

    def _load_torch(self):
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Falta instalar PyTorch.") from error
        return torch

    def _cuda_is_available(self):
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_clip_vision(self, dtype):
        from safetensors.torch import load_file
        from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

        vision_config = CLIPVisionConfig(
            hidden_size=1280,
            intermediate_size=5120,
            num_hidden_layers=32,
            num_attention_heads=16,
            image_size=224,
            patch_size=14,
            projection_dim=1024,
        )
        image_encoder = CLIPVisionModelWithProjection(vision_config)
        state_dict = load_file(str(self.clip_vision_path), device="cpu")
        state_dict.pop("vision_model.embeddings.position_ids", None)
        image_encoder.load_state_dict(state_dict, strict=True)
        image_encoder.eval()
        return image_encoder.to(dtype=dtype)

    def _load_ip_adapter_state_dict(self):
        from safetensors import safe_open

        state_dict = {"image_proj": {}, "ip_adapter": {}}
        with safe_open(str(self.adapter_path), framework="pt", device="cpu") as file:
            for key in file.keys():
                if key.startswith("image_proj."):
                    state_dict["image_proj"][key.replace("image_proj.", "")] = file.get_tensor(key)
                elif key.startswith("ip_adapter."):
                    state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = file.get_tensor(key)
        return state_dict

    def _load_pipeline(self):
        if self.pipe is not None:
            return self.pipe

        torch = self._load_torch()
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"🧠 Cargando Face Reference SD1.5 ({dtype}) desde {self.base_model}", flush=True)

        pipe = StableDiffusionPipeline.from_single_file(
            self.base_model,
            dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        print("🧠 Cargando CLIP-H local para embeddings IP-Adapter Face...", flush=True)
        self.image_encoder = self._load_clip_vision(dtype=dtype)

        print("🧠 Cargando IP-Adapter Plus Face SD1.5...", flush=True)
        pipe.load_ip_adapter(
            self._load_ip_adapter_state_dict(),
            subfolder="",
            weight_name="",
            image_encoder_folder=None,
            low_cpu_mem_usage=True,
        )

        if self.device == "cuda":
            pipe.vae.to(dtype=torch.float32)
            pipe.enable_attention_slicing(slice_size="max")
            try:
                pipe.vae.enable_slicing()
            except Exception:
                pipe.enable_vae_slicing()

            try:
                pipe.enable_sequential_cpu_offload()
                print("✅ Face Reference SD1.5 CPU offload secuencial activado", flush=True)
            except Exception:
                pipe.enable_model_cpu_offload()
                print("✅ Face Reference SD1.5 CPU offload de modelo activado", flush=True)
        else:
            pipe = pipe.to("cpu")

        self.pipe = pipe
        print("✅ Face Reference SD1.5/IP-Adapter listo", flush=True)
        return self.pipe

    def _encode_ip_adapter_image(self, pipe, face_image, num_images_per_prompt, do_classifier_free_guidance):
        torch = self._load_torch()
        from diffusers.models.embeddings import ImageProjection

        if self.image_encoder is None:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.image_encoder = self._load_clip_vision(dtype=dtype)

        image_projection_layer = pipe.unet.encoder_hid_proj.image_projection_layers[0]
        output_hidden_states = not isinstance(image_projection_layer, ImageProjection)

        encode_device = torch.device(self.device)
        dtype = next(self.image_encoder.parameters()).dtype
        print(
            f"🧠 Calculando embeddings IP-Adapter Face en {encode_device} ({dtype})...",
            flush=True,
        )

        self.image_encoder.to(encode_device)
        image = pipe.feature_extractor(face_image.convert("RGB"), return_tensors="pt").pixel_values
        image = image.to(device=encode_device, dtype=dtype)

        with torch.no_grad():
            if output_hidden_states:
                image_embeds = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                negative_image_embeds = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
            else:
                image_embeds = self.image_encoder(image).image_embeds
                negative_image_embeds = torch.zeros_like(image_embeds)

        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        negative_image_embeds = negative_image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)

        image_embeds = image_embeds.detach().to("cpu")
        self.image_encoder.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✅ Embeddings IP-Adapter Face listos", flush=True)
        return [image_embeds]

    def generate(
        self,
        face_image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        guidance_scale: float = 5.0,
        seed: int = -1,
        num_images: int = 1,
        identity_strength: float = 0.35,
        structure_strength: float = 0.8,
    ) -> List[Image.Image]:
        torch = self._load_torch()
        self.ensure_assets(allow_download=False)
        pipe = self._load_pipeline()
        pipe.set_ip_adapter_scale(float(identity_strength))
        ip_adapter_image_embeds = self._encode_ip_adapter_image(
            pipe=pipe,
            face_image=face_image,
            num_images_per_prompt=num_images,
            do_classifier_free_guidance=guidance_scale > 1.0,
        )

        generator = None
        if seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        print(
            f"🎨 Generando Face Reference SD1.5 {width}x{height}, steps={steps}, "
            f"ip_adapter={identity_strength:.2f}",
            flush=True,
        )
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✅ Face Reference SD1.5 generación terminada", flush=True)
        return result.images
