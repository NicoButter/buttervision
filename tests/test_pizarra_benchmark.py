import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "benchmarks/pizarra_benchmark.py"
spec = importlib.util.spec_from_file_location("pizarra_benchmark", path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def test_explicit_execute_is_required_before_any_network_access(self):
        with patch.object(benchmark, "request", side_effect=AssertionError("network should not be used")):
            with self.assertRaisesRegex(ValueError, "--execute"):
                benchmark.run(SimpleNamespace(execute=False))

    def test_loopback_only(self):
        for url in ("https://example.com", "http://example.com", "http://user:pw@localhost", "http://localhost/api"):
            with self.assertRaises(ValueError):
                benchmark.base_url(url)
        self.assertEqual(benchmark.base_url("http://127.0.0.1:7861/"), "http://127.0.0.1:7861")

    def test_mock_run_records_parameters_and_hashes_but_not_prompt(self):
        args = SimpleNamespace(execute=True, url="http://localhost:7861", mode="legacy", width=512,
                               height=512, steps=20, cfg=5, seed=42, server_pid=123, timeout=60)
        def request(base, path, payload=None, binary=False):
            if path == "/api/pizarra/status": return {"enabled": False}
            if path == "/api/models": return [{"active": True, "label": "model", "value": "/private/model", "family": "sd15"}]
            if path == "/api/generate/txt2img": return {"job_id": "job"}
            if path == "/api/jobs/job": return {"status": "completed", "result_images": ["/outputs/result/image.png"]}
            if path == "/api/engine": return {"scheduler": "DDIMScheduler"}
            if path.startswith("/outputs/"): return b"fake image"
            raise AssertionError(path)
        monitor = SimpleNamespace(start=lambda: None, finish=lambda: {"rss_process_peak_sampled_bytes": 10})
        with patch.object(benchmark, "request", side_effect=request), patch.object(benchmark, "ProcessMonitor", return_value=monitor):
            result = benchmark.run(args)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["scheduler"], "DDIMScheduler")
        self.assertEqual(result["parameters"]["seed"], 42)
        self.assertEqual(len(result["result_sha256"]), 1)
        self.assertNotIn(benchmark.PROMPT, str(result))
        self.assertNotIn("/private/model", str(result))

    def test_compare_rejects_differing_scheduler_or_parameters(self):
        base = {"model_identifier_sha256": "abc", "family": "sd15", "scheduler": "DDIMScheduler", "active_loras": [],
                "parameters": {"seed": 42}, "prompt_id": "fixed", "status": "completed",
                "wall_seconds": 1.0, "observations": {}, "result_sha256": ["abc"]}
        a, b = {**base, "mode": "legacy"}, {**base, "mode": "pizarra"}
        self.assertTrue(benchmark.compare(a, b)["same_result_file_sha256"])
        for changed in ({"scheduler": None}, {"parameters": {"seed": 17}}, {"family": "sdxl"}):
            with self.assertRaises(ValueError):
                benchmark.compare(a, {**b, **changed})


if __name__ == "__main__":
    unittest.main()
