"""Mocked PIZARRA-to-Diffusers inference adapter checks; no model is loaded."""
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from core.model_catalog import SDXL, SD15
from core.pizarra.inference_adapter import (
    InferenceExecutionError,
    InferencePlanError,
    PizarraInferenceAdapter,
)

GIB = 1024**3


def resources(*, ram=16, vram=16, cpu_offload=True, cuda=True):
    return {
        "ram_total_bytes": 16 * GIB,
        "ram_available_bytes": ram * GIB,
        "process_rss_bytes": 512 * 1024**2,
        "cgroup_limit_bytes": None,
        "cgroup_current_bytes": None,
        "cgroup_available_bytes": None,
        "cuda_available": cuda,
        "vram_available_bytes": vram * GIB if cuda else None,
        "vram_total_bytes": 16 * GIB if cuda else None,
        "accelerate_available": True,
        "low_cpu_mem_usage_available": True,
        "safetensors_mmap_available": True,
        "sequential_cpu_offload_available": True,
        "cpu_offload_configured": cpu_offload,
        "group_disk_offload_api_available": True,
        "diffusers_version": "0.36.0",
        "disk": {"configured": False, "valid": False, "free_bytes": 0, "reason": ""},
    }


class StorageStub:
    def __init__(self, *, available=True):
        self.config = SimpleNamespace(enabled=True)
        self.available = available
        self._stats = {"read_bytes": 20, "written_bytes": 30, "read_seconds": 0.1,
                       "write_seconds": 0.2, "free_bytes": 1000, "managed_bytes": 200,
                       "quota_remaining_bytes": 800}

    def validate(self):
        return {"available": self.available, "free_bytes": 1000}

    def stats(self):
        return dict(self._stats)


class InferenceAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name) / "juggernaut-xl.safetensors"
        self.model.write_bytes(b"local-mock-checkpoint")
        with self.model.open("r+b") as checkpoint_file:
            checkpoint_file.truncate(7 * GIB)  # sparse test fixture; no checkpoint is copied
        self.storage = StorageStub()
        self.approved = patch("core.pizarra.inference_adapter.approved_path", return_value=self.model)
        self.family = patch("core.pizarra.inference_adapter.checkpoint_family", return_value=SDXL)
        self.resources = patch("core.pizarra.inference_adapter.collect_resources", return_value=resources())
        self.budget = patch("core.pizarra.inference_adapter.inspect_budget", return_value={
            "ram": {"target_budget_bytes": 16 * GIB},
            "gpu": {"target_budget_bytes": 16 * GIB},
            "pizarra": {"available_bytes": 1000, "quota_remaining_bytes": 800},
        })
        self.approved.start()
        self.family.start()
        self.resources.start()
        self.budget.start()
        self.addCleanup(self.approved.stop)
        self.addCleanup(self.family.stop)
        self.addCleanup(self.resources.stop)
        self.addCleanup(self.budget.stop)

    def adapter(self, **kwargs):
        return PizarraInferenceAdapter(str(self.model), self.storage, **kwargs)

    def test_auto_plans_standard_and_cpu_offload(self):
        standard = self.adapter().plan()[1]
        self.assertEqual(standard["strategy"], "standard")
        self.assertEqual(standard["family"], SDXL)
        self.assertFalse(standard["storage_direct_offload_allowed"])

        self.resources.stop()
        self.resources = patch("core.pizarra.inference_adapter.collect_resources",
                               return_value=resources(ram=12, vram=2))
        self.resources.start()
        self.addCleanup(self.resources.stop)
        low_memory = self.adapter().plan()[1]
        self.assertEqual(low_memory["strategy"], "cpu_offload")
        self.assertEqual(low_memory["selected_profile"], "LOW_RAM")

    def test_rejects_unregistered_family_and_unavailable_volume(self):
        with patch("core.pizarra.inference_adapter.checkpoint_family", return_value=SD15):
            with self.assertRaises(InferencePlanError):
                self.adapter().plan()
        self.storage.available = False
        with self.assertRaisesRegex(InferencePlanError, "volumen PIZARRA"):
            self.adapter().plan()

    def test_refuses_disk_offload_without_a_quota_enforced_directory_capability(self):
        with self.assertRaisesRegex(InferencePlanError, "escritura mutable directa"):
            self.adapter(requested_strategy="DISK_OFFLOAD_EXPERIMENTAL").plan()

    def test_rejects_insufficient_memory_without_loading(self):
        self.resources.stop()
        self.resources = patch("core.pizarra.inference_adapter.collect_resources",
                               return_value=resources(ram=1, vram=1, cpu_offload=False))
        self.resources.start()
        self.addCleanup(self.resources.stop)
        with self.assertRaises(InferencePlanError) as raised:
            self.adapter().execute("short prompt")
        self.assertFalse(raised.exception.plan.get("admitted", True))

    def test_execute_measures_strategy_and_releases_pipeline(self):
        pipe = Mock()
        pipe.generate_image.return_value = [object()]
        self._patch_execution(pipe)
        images, record = self.adapter().execute("short prompt")
        self.assertEqual(len(images), 1)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["strategy"], "standard")
        self.assertIn("weight_loading", record["timings_seconds"])
        self.assertIsNone(record["timings_seconds"]["decoding_separate"])
        self.assertFalse(record["pizarra_storage_api"]["direct_diffusers_writes"])
        pipe.cleanup.assert_called_once()
        self.assertEqual(pipe.load_pipeline.call_count, 1)

    def test_memory_failure_is_recorded_and_cleanup_runs(self):
        pipe = Mock()
        pipe.generate_image.side_effect = MemoryError("simulated allocation")
        self._patch_execution(pipe)
        with self.assertRaises(InferenceExecutionError) as raised:
            self.adapter().execute("short prompt")
        self.assertEqual(raised.exception.category, "memory_allocation")
        self.assertEqual(raised.exception.record["status"], "failed")
        pipe.cleanup.assert_called_once()

    def _patch_execution(self, pipe):
        metrics = Mock()
        metrics.phase.side_effect = lambda _name: nullcontext()
        metrics.finish.return_value = {"metrics": {
            "rss_process_initial_bytes": 100,
            "rss_process_bytes": 90,
            "rss_process_final_bytes": 90,
            "rss_process_peak_sampled_bytes": 120,
            "vram_device_free_initial_bytes": 500,
            "vram_device_free_bytes": 400,
            "vram_device_free_final_bytes": 400,
            "vram_device_total_bytes": 1024,
            "vram_process_allocated_bytes": 100,
            "vram_process_reserved_bytes": 120,
            "vram_process_peak_sampled_bytes": 110,
            "vram_process_peak_lifetime_bytes": 111,
            "phases_seconds": {"weight_loading": 1.0, "inference_and_decoding": 2.0,
                                "cleanup": 0.1},
            "duration_seconds": 3.2,
        }}
        self.metrics_patch = patch("core.pizarra.inference_adapter.PizarraMetrics", return_value=metrics)
        self.pipeline_patch = patch("core.pizarra.inference_adapter.SDXLPipeline", return_value=pipe)
        self.metrics_patch.start()
        self.pipeline_patch.start()
        self.addCleanup(self.metrics_patch.stop)
        self.addCleanup(self.pipeline_patch.stop)


if __name__ == "__main__":
    unittest.main()
