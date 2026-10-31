"""Administrative persistence and service integration; no real inference."""
import json
import sqlite3
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import config
from api.generation import EngineBusyError, GenerationJob, GenerationService, PizarraUnavailableError
from core.pizarra.job_store import JobStore
from core.pizarra.storage import StorageUnavailable


def make_job():
    return GenerationJob(id=str(uuid.uuid4()), mode="txt2img", payload={"prompt": "private-unique-prompt", "reference_image": "private-photo"})


class JobStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "local" / "jobs.sqlite3"
        self.store = JobStore(self.path)

    def test_restart_marks_running_and_queued_interrupted_without_resuming(self):
        running, queued = make_job(), make_job()
        for job in (running, queued):
            self.store.create(job.as_dict(), engine="sd15", model_identifier="model.safetensors")
        running.update(status="running", progress=0.2, message="Generating")
        self.store.update(running.as_dict())
        recovered = JobStore(self.path, owner_alive=lambda pid, started: False)
        for job in (running, queued):
            row = recovered.get(job.id)
            self.assertEqual(row["status"], "interrupted")
            self.assertEqual(row["failure_category"], "process_interrupted")
            self.assertIsNotNone(row["finished_at"])
            self.assertIsNone(row["checkpoint_reference"])
        self.assertIsNotNone(recovered.get(running.id)["started_at"])

    def test_completed_result_survives_recovery_with_metrics(self):
        job = make_job()
        self.store.create(job.as_dict(), engine="sd15", model_identifier="model.safetensors")
        job.update(status="running", progress=0.2, message="Generating")
        self.store.update(job.as_dict())
        job.update(status="completed", progress=1, message="Completed", result_images=["/outputs/job/image.png"])
        metrics = {"job_id": job.id, "metrics": {"vram_process_peak_sampled_bytes": None}}
        self.store.finalize(job.as_dict(), metrics)
        recovered = JobStore(self.path)
        row = recovered.get(job.id)
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["result_images"], ["/outputs/job/image.png"])
        self.assertEqual(row["metrics_reference"], f"sqlite:{job.id}")
        self.assertEqual(recovered.get_metrics(job.id), metrics)
        self.assertIsNone(row["failure_category"])

    def test_live_owner_is_not_marked_interrupted_by_second_store(self):
        job = make_job()
        self.store.create(job.as_dict(), engine="sd15", model_identifier="model")
        job.update(status="running", progress=0.1, message="Running")
        self.store.update(job.as_dict())
        other = JobStore(self.path, owner_alive=lambda pid, started: True)
        self.assertEqual(other.get(job.id)["status"], "running")

    def test_symlinked_or_foreign_database_is_rejected_without_modification(self):
        target = Path(self.temp.name) / "foreign.sqlite3"
        target.write_bytes(b"foreign")
        target.chmod(0o600)
        link = self.path.parent / "link.sqlite3"
        link.symlink_to(target)
        with self.assertRaises(ValueError):
            JobStore(link)
        self.assertEqual(target.read_bytes(), b"foreign")
        existing = self.path.parent / "foreign.sqlite3"
        with sqlite3.connect(existing) as db:
            db.execute("CREATE TABLE private_data(value TEXT)")
        existing.chmod(0o600)
        with self.assertRaises(ValueError):
            JobStore(existing)
        with sqlite3.connect(existing) as db:
            self.assertEqual(db.execute("SELECT name FROM sqlite_master WHERE name='private_data'").fetchone()[0], "private_data")

    def test_failed_metrics_write_rolls_back_terminal_state(self):
        job = make_job()
        self.store.create(job.as_dict(), engine="sd15", model_identifier="model")
        job.update(status="running", progress=0.1, message="Running")
        self.store.update(job.as_dict())
        job.update(status="completed", progress=1, message="Completed")
        with self.assertRaises(ValueError):
            self.store.finalize(job.as_dict(), {"bad": float("nan")})
        self.assertEqual(self.store.get(job.id)["status"], "running")
        self.assertIsNone(self.store.get_metrics(job.id))

    def test_payload_does_not_enter_database_or_metrics(self):
        job = make_job()
        self.store.create(job.as_dict(), engine="sd15", model_identifier="model")
        self.store.set_metrics(job.id, {"job_id": job.id, "metrics": {"value": None}})
        with sqlite3.connect(self.path) as db:
            content = str(db.execute("SELECT * FROM jobs").fetchall()) + str(db.execute("SELECT * FROM job_metrics").fetchall())
        self.assertNotIn("private-unique-prompt", content)
        self.assertNotIn("private-photo", content)
        self.assertNotIn("payload", self.store.get(job.id))
        self.assertEqual(self.path.stat().st_mode & 0o077, 0)


class ServicePizarraTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.config = SimpleNamespace(enabled=True, metrics_enabled=True, root=root / "unmounted", quota_bytes=10000, min_free_bytes=100)
        self.storage = Mock()
        self.storage.stats.return_value = {"available": True, "total_bytes": 10000, "free_bytes": 9000,
                                           "managed_bytes": 1000, "quota_bytes": 5000, "min_free_bytes": 100,
                                           "read_bytes": 0, "written_bytes": 0, "read_seconds": 0.0, "write_seconds": 0.0}
        self.sd = Mock(model_id="model.safetensors", sd_pipeline=None, loaded_loras={}, available_loras={})
        models = Mock()
        models.resolve_model_path.return_value = "model.safetensors"
        models.list_local_model_infos.return_value = []
        self.patchers = [
            patch.object(config, "PROJECT_ROOT", root),
            patch.object(config.model_config, "model_id", "model.safetensors"),
            patch("core.pizarra.config.PizarraConfig.from_env", return_value=self.config),
            patch("core.pizarra.storage.PizarraStorage", return_value=self.storage),
            patch("api.generation.ModelManager", return_value=models),
            patch("api.generation.ButterVisionPipeline", return_value=self.sd),
            patch("api.generation.checkpoint_family", return_value="sd15"),
            patch("api.generation.collect_memory"),
        ]
        for item in self.patchers:
            item.start()
            self.addCleanup(item.stop)
        self.service = GenerationService()

    def wait_idle(self):
        self.assertTrue(self.service.worker_lock.acquire(timeout=5))
        self.service.worker_lock.release()

    @staticmethod
    def complete(job):
        job.update(status="completed", progress=1, message="Completed", result_images=["/outputs/synthetic/image.png"])

    def test_disabled_does_not_create_ledger_or_storage(self):
        temp = Path(self.temp.name) / "disabled"
        conf = SimpleNamespace(enabled=False)
        with patch.object(config, "PROJECT_ROOT", temp), patch("core.pizarra.config.PizarraConfig.from_env", return_value=conf), patch("core.pizarra.storage.PizarraStorage") as factory:
            service = GenerationService()
        self.assertIsNone(service.pizarra_store)
        self.assertIsNone(service.pizarra_storage)
        self.assertFalse(temp.exists())
        factory.assert_not_called()

    def test_enabled_job_records_terminal_and_does_not_store_payload(self):
        with patch.object(self.service, "_run_txt2img", side_effect=self.complete):
            job = self.service.create_job("txt2img", {"prompt": "private-unique-prompt", "reference_image": "private-photo"})
            self.wait_idle()
        self.assertEqual(job.status, "completed")
        self.assertEqual(job.payload, {})
        row = self.service.pizarra_store.get(job.id)
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["result_images"], ["/outputs/synthetic/image.png"])
        self.assertEqual(row["engine"], "sd15")
        self.assertTrue(row["started_at"])
        self.assertTrue(row["finished_at"])
        report = self.service.get_pizarra_metrics(job.id)
        self.assertEqual(report["status"], "completed")
        self.assertNotIn("private-unique-prompt", json.dumps(row))
        self.assertNotIn("private-photo", json.dumps(report))
        self.storage.create_workspace.assert_called_once_with(job.id)

    def test_persistent_completed_job_is_read_by_restarted_service_without_hdd(self):
        with patch.object(self.service, "_run_txt2img", side_effect=self.complete):
            original = self.service.create_job("txt2img", {"prompt": "x"})
            self.wait_idle()
        self.storage.validate.side_effect = StorageUnavailable("unmounted")
        restarted = GenerationService()
        job = restarted.get_job(original.id)
        self.assertEqual(job.status, "completed")
        self.assertEqual(job.result_images, ["/outputs/synthetic/image.png"])
        self.assertEqual(job.payload, {})
        self.assertEqual(restarted.get_pizarra_metrics(original.id)["status"], "completed")

    def test_unmounted_bank_rejected_without_worker_and_failed_ledger_recorded(self):
        self.storage.validate.side_effect = StorageUnavailable("unmounted")
        with patch.object(self.service, "_run_txt2img") as run:
            with self.assertRaises(PizarraUnavailableError):
                self.service.create_job("txt2img", {"prompt": "x"})
        run.assert_not_called()
        self.assertFalse(self.service.worker_lock.locked())
        with sqlite3.connect(self.service.pizarra_store.path) as db:
            row = db.execute("SELECT status, failure_category FROM jobs").fetchone()
        self.assertEqual(row, ("failed", "storage_unavailable"))

    def test_lost_storage_fails_job_after_existing_inference_and_preserves_local_diagnostics(self):
        self.storage.validate.side_effect = [None, None, StorageUnavailable("lost"), StorageUnavailable("lost")]
        with patch.object(self.service, "_run_txt2img", side_effect=self.complete):
            job = self.service.create_job("txt2img", {"prompt": "x"})
            self.wait_idle()
        self.assertEqual(job.status, "failed")
        self.assertEqual(job.result_images, ["/outputs/synthetic/image.png"])
        row = self.service.pizarra_store.get(job.id)
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["failure_category"], "storage_unavailable")
        self.assertEqual(self.service.get_pizarra_metrics(job.id)["status"], "failed")

    def test_completed_is_invisible_until_terminal_transaction_finishes(self):
        entered, release = threading.Event(), threading.Event()
        real_finalize = self.service.pizarra_store.finalize
        def finalize(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(4))
            return real_finalize(*args, **kwargs)
        with patch.object(self.service.pizarra_store, "finalize", side_effect=finalize), patch.object(self.service, "_run_txt2img", side_effect=self.complete):
            job = self.service.create_job("txt2img", {"prompt": "x"})
            try:
                self.assertTrue(entered.wait(4))
                self.assertEqual(self.service.get_job(job.id).as_dict()["status"], "running")
                self.assertEqual(self.service.pizarra_store.get(job.id)["status"], "running")
                with self.assertRaises(EngineBusyError):
                    self.service.create_job("txt2img", {"prompt": "second"})
                with self.assertRaises(EngineBusyError):
                    self.service.set_model("other")
            finally:
                release.set()
                self.wait_idle()
        self.assertEqual(job.status, "completed")
        self.assertEqual(self.service.pizarra_store.get(job.id)["status"], "completed")

    def test_ledger_failure_marks_job_failed_and_releases_engine(self):
        with patch.object(self.service.pizarra_store, "finalize", side_effect=sqlite3.OperationalError("full")), patch.object(self.service, "_run_txt2img", side_effect=self.complete):
            job = self.service.create_job("txt2img", {"prompt": "private"})
            self.wait_idle()
        self.assertEqual(job.status, "failed")
        self.assertEqual(self.service.pizarra_store.get(job.id)["failure_category"], "metadata_error")
        self.assertEqual(job.payload, {})


if __name__ == "__main__":
    unittest.main()
