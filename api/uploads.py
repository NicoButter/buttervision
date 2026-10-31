import base64
import binascii
import re
from io import BytesIO

from PIL import Image, UnidentifiedImageError

import config


ALLOWED_IMAGE_MIME_TYPES = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
DATA_URL_PATTERN = re.compile(r"^data:([^;,]+);base64,(.+)$", re.DOTALL)


class UploadValidationError(ValueError):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def _read_reference_bytes(data_url: str) -> tuple[bytes, str | None]:
    encoded = data_url
    expected_format = None
    if data_url.startswith("data:"):
        match = DATA_URL_PATTERN.fullmatch(data_url)
        if match is None:
            raise UploadValidationError("La foto debe estar codificada como una imagen base64 válida.", 400)
        mime_type, encoded = match.groups()
        expected_format = ALLOWED_IMAGE_MIME_TYPES.get(mime_type.lower())
        if expected_format is None:
            raise UploadValidationError("Formato no admitido. Usa JPEG, PNG o WebP.", 415)

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise UploadValidationError("La foto no contiene datos base64 válidos.", 400) from error

    if not image_bytes:
        raise UploadValidationError("La foto está vacía.", 400)
    if len(image_bytes) > config.MAX_IMAGE_UPLOAD_BYTES:
        raise UploadValidationError(
            f"La foto supera el límite de {config.MAX_IMAGE_UPLOAD_BYTES} bytes.",
            413,
        )
    return image_bytes, expected_format


def _verify_reference_bytes(image_bytes: bytes, expected_format: str | None) -> None:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in ALLOWED_IMAGE_MIME_TYPES.values():
                raise UploadValidationError("Formato no admitido. Usa JPEG, PNG o WebP.", 415)
            if expected_format is not None and image.format != expected_format:
                raise UploadValidationError("El contenido de la foto no coincide con su formato declarado.", 415)
            image.verify()
    except UploadValidationError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as error:
        raise UploadValidationError("El archivo no contiene una imagen válida.", 400) from error


def validate_reference_image(data_url: str) -> None:
    image_bytes, expected_format = _read_reference_bytes(data_url)
    _verify_reference_bytes(image_bytes, expected_format)


def decode_reference_image(data_url: str) -> Image.Image:
    image_bytes, expected_format = _read_reference_bytes(data_url)
    _verify_reference_bytes(image_bytes, expected_format)
    with Image.open(BytesIO(image_bytes)) as image:
        return image.convert("RGB")
