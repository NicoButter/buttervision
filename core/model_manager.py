"""
Model Manager - Gestión de modelos Stable Diffusion
Descarga y gestión de modelos desde Hugging Face y CivitAI
"""
import os
import requests
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from huggingface_hub import HfApi, snapshot_download
import config


class ModelManager:
    """Administrador de modelos Stable Diffusion"""

    def __init__(self):
        self.models_dir = config.SD_MODELS_DIR
        self.hf_api = HfApi()

    def list_local_models(self) -> List[str]:
        """Lista modelos locales disponibles."""
        models = []
        if self.models_dir.exists():
            for item in self.models_dir.iterdir():
                if item.is_dir() and self.is_diffusers_model_dir(item):
                    models.append(item.name)
                elif self.is_single_file_model(item):
                    models.append(item.stem)
        return models

    def _local_name_for_model(self, model_id: str) -> str:
        """Convierte un repo id de Hugging Face en un nombre local estable."""
        return model_id.replace("/", "__")

    def is_diffusers_model_dir(self, path: Path) -> bool:
        """Valida que un directorio tenga la estructura mínima de un modelo Diffusers."""
        required_files = [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "tokenizer/tokenizer_config.json",
            "unet/config.json",
            "vae/config.json",
        ]
        return path.is_dir() and all((path / filename).exists() for filename in required_files)

    def is_single_file_model(self, path: Path) -> bool:
        """Valida checkpoints tipo Forge/Automatic1111."""
        return path.is_file() and path.suffix.lower() in {".safetensors", ".ckpt"}

    def _default_model_path(self) -> Path:
        return self.models_dir / config.model_config.default_model_filename

    def resolve_model_path(self, model_id: str) -> Optional[str]:
        """
        Resuelve un modelo a una ruta local usable si ya está descargado.

        Busca en:
        - rutas locales explícitas
        - models/stable-diffusion/<nombre-local>
        - cache de Hugging Face configurada para ButterVision
        """
        explicit_path = Path(model_id).expanduser()
        if self.is_diffusers_model_dir(explicit_path):
            return str(explicit_path)
        if self.is_single_file_model(explicit_path):
            return str(explicit_path)

        if model_id in {"default", "cyberrealistic_final", config.model_config.default_model_filename}:
            default_path = self._default_model_path()
            if self.is_single_file_model(default_path):
                return str(default_path)

        local_path = self.models_dir / self._local_name_for_model(model_id)
        if self.is_diffusers_model_dir(local_path):
            return str(local_path)

        safetensors_path = self.models_dir / f"{model_id}.safetensors"
        if self.is_single_file_model(safetensors_path):
            return str(safetensors_path)

        legacy_local_path = self.models_dir / model_id.replace("/", "_")
        if self.is_diffusers_model_dir(legacy_local_path):
            return str(legacy_local_path)

        try:
            cached_path = snapshot_download(
                repo_id=model_id,
                cache_dir=str(config.CACHE_DIR),
                local_files_only=True,
            )
            cached_path = Path(cached_path)
            if self.is_diffusers_model_dir(cached_path):
                return str(cached_path)
        except Exception:
            return None

        return None

    def list_hf_models(self, search: str = "stable-diffusion", limit: int = 10) -> List[str]:
        """Busca modelos en Hugging Face"""
        try:
            models = self.hf_api.list_models(
                search=search,
                sort="downloads",
                direction=-1,
                limit=limit
            )
            return [model.id for model in models]
        except Exception as e:
            print(f"Error buscando modelos en HF: {e}")
            return []

    def download_hf_model(self, model_id: str, local_name: Optional[str] = None) -> str:
        """Descarga un modelo desde Hugging Face"""
        try:
            if local_name:
                local_path = self.models_dir / local_name
            else:
                local_path = self.models_dir / self._local_name_for_model(model_id)

            print(f"Descargando {model_id} a {local_path}...")

            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_path),
                cache_dir=str(config.CACHE_DIR),
                local_dir_use_symlinks=False,
                resume_download=True,
            )

            if not self.is_diffusers_model_dir(local_path):
                raise Exception(
                    "La descarga terminó, pero la carpeta no tiene estructura Diffusers válida"
                )

            print(f"✅ Modelo descargado: {local_path}")
            return str(local_path)

        except Exception as e:
            raise Exception(f"Error descargando modelo HF: {e}")

    def ensure_hf_model(self, model_id: str, local_name: Optional[str] = None, allow_download: bool = True) -> str:
        """Garantiza que un modelo base esté disponible localmente."""
        resolved_path = self.resolve_model_path(model_id)
        if resolved_path:
            print(f"✅ Modelo base disponible: {model_id}")
            print(f"   Ruta: {resolved_path}")
            return resolved_path

        if not allow_download:
            raise Exception(f"Modelo base no encontrado localmente: {model_id}")

        print(f"📥 Modelo base no encontrado. Descargando: {model_id}")
        return self.download_hf_model(model_id, local_name)

    def download_model_file(self, url: str, filename: str) -> str:
        """Descarga un checkpoint de un solo archivo desde una URL directa."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        local_path = self.models_dir / filename
        partial_path = local_path.with_suffix(local_path.suffix + ".part")
        token = os.getenv("CIVITAI_API_TOKEN") or os.getenv("CIVITAI_TOKEN")
        request_url = url
        headers = {}

        if token:
            headers["Authorization"] = f"Bearer {token}"
            parts = urlsplit(url)
            query = dict(parse_qsl(parts.query, keep_blank_values=True))
            query.setdefault("token", token)
            request_url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))

        print(f"📥 Descargando checkpoint desde: {url}")
        print(f"   Destino: {local_path}")

        try:
            response = requests.get(
                request_url,
                stream=True,
                timeout=60,
                allow_redirects=True,
                headers=headers,
            )
            if response.status_code == 401:
                raise Exception(
                    "La descarga respondió 401 Unauthorized. Si usas una URL protegida, "
                    "configura el token correspondiente o coloca el modelo manualmente."
                )
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(partial_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = downloaded * 100 / total_size
                        print(f"\r   Progreso: {percent:5.1f}%", end="", flush=True)

            if total_size:
                print()

            if partial_path.stat().st_size < 1024 * 1024:
                raise Exception("El archivo descargado es demasiado pequeño para ser un modelo válido")

            partial_path.replace(local_path)
            print(f"✅ Checkpoint descargado: {local_path}")
            return str(local_path)

        except Exception:
            if partial_path.exists():
                partial_path.unlink()
            raise

    def ensure_default_model(self, allow_download: bool = True) -> str:
        """Garantiza el modelo inicial CyberRealistic."""
        model_path = self._default_model_path()
        if self.is_single_file_model(model_path):
            print(f"✅ Modelo base disponible: {model_path.name}")
            print(f"   Ruta: {model_path}")
            return str(model_path)

        if not allow_download:
            raise Exception(f"Modelo base no encontrado localmente: {model_path}")

        return self.download_model_file(
            url=config.model_config.default_model_url,
            filename=config.model_config.default_model_filename,
        )

    def ensure_model(self, model_id: str, allow_download: bool = True) -> str:
        """Garantiza un modelo local, ya sea checkpoint CivitAI o repo Hugging Face."""
        if model_id in {"default", "cyberrealistic_final", config.model_config.default_model_filename}:
            return self.ensure_default_model(allow_download=allow_download)
        return self.ensure_hf_model(model_id=model_id, allow_download=allow_download)

    def bootstrap_base_models(self, model_ids: List[str], allow_download: bool = True) -> Dict[str, str]:
        """Prepara los modelos base mínimos para que ButterVision pueda arrancar."""
        resolved_models = {}
        for model_id in model_ids:
            resolved_models[model_id] = self.ensure_model(
                model_id=model_id,
                allow_download=allow_download,
            )
        return resolved_models

    def download_civitai_model(self, model_id: str, local_name: str) -> str:
        """Descarga un modelo desde CivitAI"""
        try:
            # URL base de CivitAI
            url = f"https://civitai.com/api/download/models/{model_id}"

            local_path = self.models_dir / f"{local_name}.safetensors"

            print(f"Descargando desde CivitAI: {url}...")

            # Descargar archivo
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"✅ Modelo descargado: {local_path}")
            return str(local_path)

        except Exception as e:
            raise Exception(f"Error descargando modelo CivitAI: {e}")

    def get_model_path(self, model_name: str) -> Optional[str]:
        """Obtiene la ruta completa de un modelo local"""
        model_path = self.models_dir / model_name
        if self.is_diffusers_model_dir(model_path):
            return str(model_path)

        # Si no es directorio, buscar archivo .safetensors
        safetensors_path = self.models_dir / f"{model_name}.safetensors"
        if self.is_single_file_model(safetensors_path):
            return str(safetensors_path)

        return None
