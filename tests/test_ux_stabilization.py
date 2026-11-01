import base64
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

import api.app as api_app
import config
from api.app import PrivateOutputFiles
from api.schemas import FaceReferenceRequest
from api.output_files import list_public_output_images


class FastApiStabilizationTests(unittest.TestCase):
    def test_runtime_endpoint_returns_configured_upload_limits(self):
        response = api_app.runtime_info()
        self.assertEqual(
            response["upload_limits"],
            {
                "image_bytes": config.MAX_IMAGE_UPLOAD_BYTES,
                "video_bytes": config.MAX_VIDEO_UPLOAD_BYTES,
            },
        )

    def test_face_reference_rejects_unsupported_mime_before_queueing(self):
        payload = FaceReferenceRequest(prompt="test", reference_image="data:image/gif;base64,YQ==")
        with patch.object(api_app, "get_service", side_effect=AssertionError("generation service should not load")):
            with self.assertRaises(HTTPException) as response:
                api_app.generate_face_reference(payload)
        self.assertEqual(response.exception.status_code, 415)
        self.assertIn("JPEG, PNG o WebP", response.exception.detail)

    def test_face_reference_rejects_oversized_data_before_queueing(self):
        payload = {
            "prompt": "test",
            "reference_image": f"data:image/png;base64,{base64.b64encode(b'abc').decode()}",
        }
        with patch.object(config, "MAX_IMAGE_UPLOAD_BYTES", 2):
            with patch.object(api_app, "get_service", side_effect=AssertionError("generation service should not load")):
                with self.assertRaises(HTTPException) as response:
                    api_app.generate_face_reference(FaceReferenceRequest(**payload))

        self.assertEqual(response.exception.status_code, 413)

    def test_output_static_files_hide_private_inputs_and_metadata(self):
        with tempfile.TemporaryDirectory() as output_dir:
            folder = Path(output_dir) / "generation"
            folder.mkdir()
            private_files = ("reference_face.png", "metadata.json", "prompt.txt")
            for name in private_files:
                (folder / name).write_bytes(b"private")
            (folder / "image_01.png").write_bytes(b"image")
            private_outputs = PrivateOutputFiles(directory=output_dir)
            scope = {"type": "http", "method": "GET", "path": "/outputs/generation/image_01.png",
                     "headers": [], "query_string": b"", "root_path": ""}

            for name in private_files:
                with self.assertRaises(HTTPException) as response:
                    asyncio.run(private_outputs.get_response(f"generation/{name}", scope))
                self.assertEqual(response.exception.status_code, 404, name)
                self.assertTrue((folder / name).exists(), name)
            with patch("starlette.staticfiles.StaticFiles.get_response", new_callable=AsyncMock) as parent:
                marker = object()
                parent.return_value = marker
                response = asyncio.run(private_outputs.get_response("generation/image_01.png", scope))
            self.assertIs(response, marker)
            parent.assert_awaited_once_with("generation/image_01.png", scope)

    def test_api_and_gradio_history_exclude_private_output_images(self):
        with tempfile.TemporaryDirectory() as output_dir:
            folder = Path(output_dir) / "generation"
            folder.mkdir()
            private_reference = folder / "reference_face.png"
            private_reference.write_bytes(b"private")
            public_result = folder / "image_01.png"
            public_result.write_bytes(b"image")

            self.assertEqual(list_public_output_images(Path(output_dir), 24), [public_result])
            self.assertTrue(private_reference.exists())

    def test_service_defaults_bind_to_loopback_and_keep_gradio_port(self):
        self.assertEqual(config.server_config.host, "127.0.0.1")
        self.assertEqual(config.server_config.port, 7860)
        self.assertFalse(config.server_config.share)


if __name__ == "__main__":
    unittest.main()
