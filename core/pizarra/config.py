"""Configuration for optional PIZARRA infrastructure."""
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re


def _boolean(name, default):
    value = os.environ.get(name, str(default)).strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} debe ser true o false")


def _bytes(name, default, *, positive=False):
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} debe ser un número de GB válido") from error
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} debe ser un número de GB válido")
    result = int(value * 1_000_000_000)
    if positive and result == 0:
        raise ValueError(f"{name} debe ser un número de GB válido")
    return result


@dataclass(frozen=True)
class PizarraConfig:
    enabled: bool = False
    root: Path = Path("/mnt/pizarra")
    quota_bytes: int = 400_000_000_000
    min_free_bytes: int = 20_000_000_000
    metrics_enabled: bool = True
    volume_uuid: str = ""

    def __post_init__(self):
        if self.enabled and (not Path(self.root).is_absolute() or self.quota_bytes <= 0 or self.min_free_bytes < 0
                             or (self.volume_uuid and not re.fullmatch(r"[A-Za-z0-9-]{4,64}", self.volume_uuid))):
            raise ValueError("Configuración PIZARRA inválida")

    @classmethod
    def from_env(cls):
        enabled = _boolean("BUTTERVISION_PIZARRA_ENABLED", False)
        if not enabled:
            # The disabled feature must not interfere with legacy startup even
            # if stale PIZARRA environment values are malformed.
            return cls(enabled=False)
        root = Path(os.environ.get("BUTTERVISION_PIZARRA_ROOT", "/mnt/pizarra")).expanduser()
        if not root.is_absolute():
            raise ValueError("BUTTERVISION_PIZARRA_ROOT debe ser una ruta absoluta")
        return cls(
            enabled=enabled,
            root=root,
            quota_bytes=_bytes("BUTTERVISION_PIZARRA_QUOTA_GB", 400, positive=True),
            min_free_bytes=_bytes("BUTTERVISION_PIZARRA_MIN_FREE_GB", 20),
            metrics_enabled=_boolean("BUTTERVISION_PIZARRA_METRICS_ENABLED", True),
            volume_uuid=os.environ.get("BUTTERVISION_PIZARRA_VOLUME_UUID", "").strip(),
        )
