import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException
import api.app as api_app
from api.generation import PizarraUnavailableError


class PizarraApiTests(unittest.TestCase):
    def test_disabled_status_and_missing_metrics(self):
        engine = Mock()
        engine.pizarra_config = SimpleNamespace(enabled=False)
        engine.get_pizarra_metrics.return_value = None
        with patch.object(api_app, "get_service", return_value=engine):
            status = api_app.pizarra_status()
            with self.assertRaises(HTTPException) as missing:
                api_app.pizarra_job_metrics("unknown")
        self.assertFalse(status["enabled"])
        self.assertIsNone(status["available"])
        self.assertEqual(missing.exception.status_code, 404)

    def test_status_does_not_disclose_root_or_error_details(self):
        engine = Mock()
        engine.pizarra_config = SimpleNamespace(enabled=True, metrics_enabled=True)
        engine.pizarra_storage.stats.return_value = {"total_bytes": 100, "free_bytes": 50,
                                                      "root": "/private/drive", "managed_bytes": 20}
        with patch.object(api_app, "get_service", return_value=engine):
            result = api_app.pizarra_status()
            engine.pizarra_storage.stats.side_effect = OSError("/private/drive")
            unavailable = api_app.pizarra_status()
        self.assertEqual(result["storage"]["managed_bytes"], 20)
        self.assertNotIn("/private/drive", str(result))
        self.assertNotIn("/private/drive", str(unavailable))
        self.assertFalse(unavailable["available"])

    def test_metrics_remain_accessible_when_volume_is_absent(self):
        engine = Mock()
        engine.get_pizarra_metrics.return_value = {"job_id": "job", "metrics": {"gpu_execution_seconds": None}}
        engine.pizarra_storage.stats.side_effect = OSError("disconnected")
        with patch.object(api_app, "get_service", return_value=engine):
            response = api_app.pizarra_job_metrics("job")
        self.assertIsNone(response["metrics"]["gpu_execution_seconds"])

    def test_budget_is_opt_in_and_observational(self):
        engine = Mock()
        engine.pizarra_config = SimpleNamespace(enabled=True, metrics_enabled=True)
        engine.pizarra_storage.stats.return_value = {"total_bytes": 100, "free_bytes": 50}
        budget = {"operation": "generation", "observational": True, "gpu": {"target_budget_bytes": 10}}
        with patch.object(api_app, "get_service", return_value=engine), patch("core.pizarra.memory_budget.inspect_budget", return_value=budget) as inspect:
            ordinary = api_app.pizarra_status()
            inspect.assert_not_called()
            result = api_app.pizarra_status(include_budget=True)
        self.assertNotIn("budget", ordinary)
        self.assertEqual(result["budget"], budget)

    def test_unavailable_generation_is_controlled_503_without_paths(self):
        engine = Mock()
        engine.create_job.side_effect = PizarraUnavailableError("/private/drive is gone")
        with patch.object(api_app, "get_service", return_value=engine):
            with self.assertRaises(HTTPException) as result:
                api_app.engine_call(engine.create_job, "txt2img", {"prompt": "test"})
        self.assertEqual(result.exception.status_code, 503)
        self.assertNotIn("/private/drive", result.exception.detail)


if __name__ == "__main__":
    unittest.main()
