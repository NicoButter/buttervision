import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.app as api_app
import config
from api.app import PrivateOutputFiles, app
from api.output_files import list_public_output_images


class FastApiStabilizationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_runtime_endpoint_returns_configured_upload_limits(self):
        response = self.client.get("/api/runtime")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["upload_limits"],
            {
                "image_bytes": config.MAX_IMAGE_UPLOAD_BYTES,
                "video_bytes": config.MAX_VIDEO_UPLOAD_BYTES,
            },
        )

    def test_face_reference_rejects_unsupported_mime_before_queueing(self):
        payload = {"prompt": "test", "reference_image": "data:image/gif;base64,YQ=="}
        with patch.object(api_app, "get_service", side_effect=AssertionError("generation service should not load")):
            response = self.client.post("/api/generate/face-reference", json=payload)

        self.assertEqual(response.status_code, 415)
        self.assertIn("JPEG, PNG o WebP", response.json()["detail"])

    def test_face_reference_rejects_oversized_data_before_queueing(self):
        payload = {
            "prompt": "test",
            "reference_image": f"data:image/png;base64,{base64.b64encode(b'abc').decode()}",
        }
        with patch.object(config, "MAX_IMAGE_UPLOAD_BYTES", 2):
            with patch.object(api_app, "get_service", side_effect=AssertionError("generation service should not load")):
                response = self.client.post("/api/generate/face-reference", json=payload)

        self.assertEqual(response.status_code, 413)

    def test_output_static_files_hide_private_inputs_and_metadata(self):
        isolated_app = FastAPI()
        with tempfile.TemporaryDirectory() as output_dir:
            folder = Path(output_dir) / "generation"
            folder.mkdir()
            private_files = ("reference_face.png", "metadata.json", "prompt.txt")
            for name in private_files:
                (folder / name).write_bytes(b"private")
            (folder / "image_01.png").write_bytes(b"image")
            isolated_app.mount("/outputs", PrivateOutputFiles(directory=output_dir))
            client = TestClient(isolated_app)

            for name in private_files:
                response = client.get(f"/outputs/generation/{name}")
                self.assertEqual(response.status_code, 404, name)
                self.assertTrue((folder / name).exists(), name)
            self.assertEqual(client.get("/outputs/generation/image_01.png").status_code, 200)

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
