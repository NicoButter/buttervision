import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core.pizarra.metrics import PizarraMetrics
from core.engine_memory import bind_pizarra_metrics, lifecycle, measured_phase


class MetricsTests(unittest.TestCase):
    def test_disabled_preserves_unknown_as_null(self):
        metric = PizarraMetrics("job", "sd15", enabled=False)
        metric.start()
        result = metric.finish("completed")
        self.assertIsNone(result["metrics"]["duration_seconds"])
        self.assertIsNone(result["metrics"]["vram_device_total_bytes"])
        self.assertIsNone(result["metrics"]["gpu_execution_seconds"])
        self.assertIsNone(result["metrics"]["pizarra_storage"])

    def test_cpu_ram_and_phase_samples_are_scoped(self):
        process = Mock()
        process.cpu_times.return_value = SimpleNamespace(user=4.0, system=1.0)
        process.memory_info.return_value = SimpleNamespace(rss=1234)
        process.num_threads.return_value = 4
        with patch("core.pizarra.metrics.psutil.Process", return_value=process), patch("core.pizarra.metrics.psutil.cpu_percent", return_value=15.5), patch("core.pizarra.metrics.psutil.virtual_memory", return_value=SimpleNamespace(available=888)), patch("core.pizarra.metrics.torch.cuda.is_available", return_value=False):
            metric = PizarraMetrics("job", "sd15")
            metric.start()
            with metric.phase("preparation"):
                time.sleep(0.001)
            result = metric.finish("completed", {"written_bytes": 3})
        values = result["metrics"]
        self.assertEqual(values["ram_system_available_bytes"], 888)
        self.assertEqual(values["rss_process_peak_sampled_bytes"], 1234)
        self.assertEqual(values["cpu_system_percent"], 15.5)
        self.assertEqual(values["process_threads"], 4)
        self.assertGreater(values["phases_seconds"]["preparation"], 0)
        self.assertIsNone(values["vram_process_peak_sampled_bytes"])
        self.assertEqual(values["pizarra_storage"]["written_bytes"], 3)
        self.assertNotIn("prompt", str(result))

    def test_cuda_process_and_device_metrics_are_not_conflated(self):
        with patch("core.pizarra.metrics.torch.cuda.is_available", return_value=True), patch("core.pizarra.metrics.torch.cuda.is_initialized", return_value=True), patch("core.pizarra.metrics.torch.cuda.mem_get_info", return_value=(30, 100)), patch("core.pizarra.metrics.torch.cuda.memory_allocated", return_value=12), patch("core.pizarra.metrics.torch.cuda.memory_reserved", return_value=20), patch("core.pizarra.metrics.torch.cuda.max_memory_allocated", return_value=70), patch("core.pizarra.metrics.torch.cuda.synchronize", side_effect=AssertionError("must not synchronize")), patch("core.pizarra.metrics.torch.cuda.reset_peak_memory_stats", side_effect=AssertionError("must not reset")):
            metric = PizarraMetrics("job", "sdxl")
            metric.start()
            result = metric.finish("completed")
        values = result["metrics"]
        self.assertEqual(values["vram_device_free_bytes"], 30)
        self.assertEqual(values["vram_device_total_bytes"], 100)
        self.assertEqual(values["vram_process_peak_sampled_bytes"], 12)
        self.assertEqual(values["vram_process_peak_lifetime_bytes"], 70)
        self.assertIsNone(values["gpu_execution_seconds"])

    def test_context_scopes_model_loading_and_pipeline_phase(self):
        metric = PizarraMetrics("job", "sd15")
        @lifecycle("txt2img", "load")
        def load():
            return "loaded"
        with patch("core.engine_memory.diagnose"), patch("core.pizarra.metrics.torch.cuda.is_available", return_value=False):
            metric.start()
            with bind_pizarra_metrics(metric):
                self.assertEqual(load(), "loaded")
                with measured_phase("inference_and_decoding"):
                    pass
            with measured_phase("not_in_job"):
                pass
            result = metric.finish("completed")
        phases = result["metrics"]["phases_seconds"]
        self.assertIn("model_loading", phases)
        self.assertIn("inference_and_decoding", phases)
        self.assertNotIn("not_in_job", phases)


if __name__ == "__main__":
    unittest.main()
