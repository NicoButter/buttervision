"""
Core module __init__
"""

__all__ = [
    "StableDiffusionManager",
    "ButterVisionPipeline",
    "create_buttervision_pipeline",
    "LoRAManager",
    "lora_manager",
    "ModelManager",
]


def __getattr__(name):
    if name == "StableDiffusionManager":
        from .pipeline import StableDiffusionManager
        return StableDiffusionManager
    if name in {"ButterVisionPipeline", "create_buttervision_pipeline"}:
        from .advanced_pipeline import ButterVisionPipeline, create_buttervision_pipeline
        return {
            "ButterVisionPipeline": ButterVisionPipeline,
            "create_buttervision_pipeline": create_buttervision_pipeline,
        }[name]
    if name in {"LoRAManager", "lora_manager"}:
        from .lora_manager import LoRAManager, lora_manager
        return {
            "LoRAManager": LoRAManager,
            "lora_manager": lora_manager,
        }[name]
    if name == "ModelManager":
        from .model_manager import ModelManager
        return ModelManager
    raise AttributeError(f"module 'core' has no attribute '{name}'")
