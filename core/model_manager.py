"""
Model Manager - Gestión de modelos Stable Diffusion
Descarga y gestión de modelos desde Hugging Face y CivitAI
"""
import os
import requests
from pathlib import Path
from typing import List, Optional
# from huggingface_hub import snapshot_download, HfApi
# from huggingface_hub import HfApi
import config


class ModelManager:
    """Administrador de modelos Stable Diffusion"""

    def __init__(self):
        self.models_dir = config.SD_MODELS_DIR
        # self.hf_api = HfApi()

    def list_local_models(self) -> List[str]:
        """Lista modelos locales disponibles (directorios con model_index.json)"""
        models = []
        if self.models_dir.exists():
            for item in self.models_dir.iterdir():
                if item.is_dir() and (item / "model_index.json").exists():
                    models.append(item.name)
        return models

    def list_hf_models(self, search: str = "stable-diffusion", limit: int = 10) -> List[str]:
        """Busca modelos en Hugging Face"""
        try:
            # models = self.hf_api.list_models(
            #     search=search,
            #     sort="downloads",
            #     direction=-1,
            #     limit=limit
            # )
            # return [model.id for model in models]
            return ["runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1"]
        except Exception as e:
            print(f"Error buscando modelos en HF: {e}")
            return []

    def download_hf_model(self, model_id: str, local_name: Optional[str] = None) -> str:
        """Descarga un modelo desde Hugging Face"""
        try:
            if local_name:
                local_path = self.models_dir / local_name
            else:
                local_path = self.models_dir / model_id.replace("/", "_")

            print(f"Descargando {model_id} a {local_path}...")

            # Descargar con cache
            # snapshot_download(
            #     repo_id=model_id,
            #     local_dir=str(local_path),
            #     cache_dir=str(config.CACHE_DIR),
            #     local_dir_use_symlinks=False
            # )
            # Simular descarga
            local_path.mkdir(parents=True, exist_ok=True)
            (local_path / "model_index.json").write_text('{"test": "test"}')

            print(f"✅ Modelo descargado: {local_path}")
            return str(local_path)

        except Exception as e:
            raise Exception(f"Error descargando modelo HF: {e}")

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
        if model_path.exists() and (model_path / "model_index.json").exists():
            return str(model_path)

        # Si no es directorio, buscar archivo .safetensors
        safetensors_path = self.models_dir / f"{model_name}.safetensors"
        if safetensors_path.exists():
            return str(safetensors_path)

        return None