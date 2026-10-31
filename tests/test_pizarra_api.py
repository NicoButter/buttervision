import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import api.app as api_app
from api.generation import PizarraUnavailableError


class PizarraApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api_app.app)

    def test_disabled_status_and_missing_metrics(self):
        engine = Mock()
        engine.pizarra_config = SimpleNamespace(enabled=False)
        engine.get_pizarra_metrics.return_value = None
        with patch.object(api_app, "get_service", return_value=engine):
            status = self.client.get("/api/pizarra/status")
            missing = self.client.get("/api/pizarra/jobs/unknown/metrics")
        self.assertEqual(status.status_code, 200)
        self.assertFalse(status.json()["enabled"])
        self.assertIsNone(status.json()["available"])
        self.assertEqual(missing.status_code, 404)

    def test_status_does_not_disclose_root_or_error_details(self):
        engine = Mock()
        engine.pizarra_config = SimpleNamespace(enabled=True, metrics_enabled=True)
        engine.pizarra_storage.stats.return_value = {"total_bytes": 100, "free_bytes": 50,
                                                      "root": "/private/drive", "managed_bytes": 20}
        with patch.object(api_app, "get_service", return_value=engine):
            result = self.client.get("/api/pizarra/status")
            engine.pizarra_storage.stats.side_effect = OSError("/private/drive")
            unavailable = self.client.get("/api/pizarra/status")
        self.assertEqual(result.json()["storage"]["managed_bytes"], 20)
        self.assertNotIn("/private/drive", result.text)
        self.assertNotIn("/private/drive", unavailable.text)
        self.assertFalse(unavailable.json()["available"])

    def test_metrics_remain_accessible_when_volume_is_absent(self):
        engine = Mock()
        engine.get_pizarra_metrics.return_value = {"job_id": "job", "metrics": {"gpu_execution_seconds": None}}
        engine.pizarra_storage.stats.side_effect = OSError("disconnected")
        with patch.object(api_app, "get_service", return_value=engine):
            response = self.client.get("/api/pizarra/jobs/job/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["metrics"]["gpu_execution_seconds"])

    def test_budget_is_opt_in_and_observational(self):
        engine = Mock()
        engine.pizarra_config = SimpleNamespace(enabled=True, metrics_enabled=True)
        engine.pizarra_storage.stats.return_value = {"total_bytes": 100, "free_bytes": 50}
        budget = {"operation": "generation", "observational": True, "gpu": {"target_budget_bytes": 10}}
        with patch.object(api_app, "get_service", return_value=engine), patch("core.pizarra.memory_budget.inspect_budget", return_value=budget) as inspect:
            ordinary = self.client.get("/api/pizarra/status")
            inspect.assert_not_called()
            result = self.client.get("/api/pizarra/status?include_budget=true")
        self.assertNotIn("budget", ordinary.json())
        self.assertEqual(result.json()["budget"], budget)

    def test_unavailable_generation_is_controlled_503_without_paths(self):
        engine = Mock()
        engine.create_job.side_effect = PizarraUnavailableError("/private/drive is gone")
        with patch.object(api_app, "get_service", return_value=engine):
            result = self.client.post("/api/generate/txt2img", json={"prompt": "test"})
        self.assertEqual(result.status_code, 503)
        self.assertNotIn("/private/drive", result.text)


if __name__ == "__main__":
    unittest.main()
