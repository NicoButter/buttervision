"""Storage tests simulate a mount in tempfile; never touch physical disks."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch

from core.pizarra.config import PizarraConfig
from core.pizarra.storage import (
    AlreadyExists, IntegrityError, PizarraStorage, QuotaExceeded,
    StorageUnavailable, UnsafePath,
)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        dev = self.root.stat().st_dev
        self.volume_id = ("73", f"{os.major(dev)}:{os.minor(dev)}")
        self.mounts = {self.root: self.volume_id, Path("/"): ("1", "999:1")}
        def mount(path):
            try:
                return self.mounts[Path(path)]
            except KeyError:
                raise StorageUnavailable("not mounted")
        self.mount_patch = patch("core.pizarra.storage._mount_identity", side_effect=mount)
        self.mount_patch.start()
        self.addCleanup(self.mount_patch.stop)
        self.config = PizarraConfig(enabled=True, root=self.root, quota_bytes=100_000,
                                    min_free_bytes=0, metrics_enabled=True)
        self.storage = PizarraStorage(self.config)
        self.job = str(uuid.uuid4())

    def test_disabled_never_creates_anything(self):
        storage = PizarraStorage(PizarraConfig(enabled=False, root=self.root))
        self.assertEqual(storage.validate(), {"enabled": False, "available": False})
        with self.assertRaises(StorageUnavailable):
            storage.create_workspace(self.job)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_non_mountpoint_missing_and_root_device_are_rejected(self):
        self.mounts.pop(self.root)
        with self.assertRaises(StorageUnavailable):
            self.storage.create_workspace(self.job)
        self.assertEqual(list(self.root.iterdir()), [])
        self.mounts[self.root] = self.volume_id
        self.mounts[Path("/")] = ("1", self.volume_id[1])
        with self.assertRaises(StorageUnavailable):
            self.storage.validate()
        missing = PizarraStorage(PizarraConfig(enabled=True, root=self.root / "missing"))
        with self.assertRaises(StorageUnavailable):
            missing.validate()
        self.assertFalse((self.root / "missing").exists())

    def test_expected_uuid_rejects_wrong_volume_before_writing(self):
        storage = PizarraStorage(PizarraConfig(enabled=True, root=self.root,
                                 quota_bytes=100_000, min_free_bytes=0, volume_uuid="ABCD-1234"))
        with patch("core.pizarra.storage._uuid_device", return_value=-1):
            with self.assertRaises(StorageUnavailable):
                storage.create_workspace(self.job)
        self.assertEqual(list(self.root.iterdir()), [])
        with patch("core.pizarra.storage._uuid_device", return_value=self.root.stat().st_dev):
            self.assertTrue(storage.validate()["available"])

    def test_symlinked_root_or_symlinked_parent_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        link = self.root / "link"
        link.symlink_to(outside, target_is_directory=True)
        storage = PizarraStorage(PizarraConfig(enabled=True, root=link))
        with self.assertRaises(StorageUnavailable):
            storage.create_workspace(self.job)
        self.assertEqual(list(outside.iterdir()), [])

    def test_workspace_commit_read_and_manifest_recovery(self):
        self.storage.create_workspace(self.job)
        ref = self.storage.commit(self.job, "tensors", "tensor.bin", b"small tensor")
        self.assertEqual(ref, "tensors/tensor.bin")
        self.assertEqual(self.storage.read(self.job, ref), b"small tensor")
        counters = self.storage.stats()
        self.assertGreater(counters["metadata_written_bytes"], 0)
        self.assertGreater(counters["tensor_written_bytes"], 0)
        self.assertGreater(counters["tensor_read_bytes"], 0)
        state = self.storage.stats()
        self.assertTrue(state["available"])
        self.assertGreater(state["read_bytes"], 0)
        self.assertGreater(state["written_bytes"], 0)
        self.assertGreater(state["managed_bytes"], 0)
        recovered = PizarraStorage(self.config).recover_manifests()
        self.assertEqual([entry["job_id"] for entry in recovered], [self.job])
        self.assertNotIn("workspace_token", recovered[0])
        with self.assertRaises(AlreadyExists):
            self.storage.commit(self.job, "tensors", "tensor.bin", b"overwrite")
        self.assertEqual(self.storage.read(self.job, ref), b"small tensor")

    def test_disabled_metrics_are_unknown_not_zero(self):
        storage = PizarraStorage(PizarraConfig(enabled=True, root=self.root,
                                 quota_bytes=100_000, min_free_bytes=0, metrics_enabled=False))
        storage.create_workspace(self.job)
        status = storage.stats()
        for key in ("read_bytes", "written_bytes", "read_seconds", "write_seconds",
                    "read_bytes_per_second", "write_bytes_per_second"):
            self.assertIsNone(status[key])

    def test_partial_record_never_published_and_checksum_detects_corruption(self):
        self.storage.create_workspace(self.job)
        token = self.storage.write_temporary(self.job, b"incomplete")
        temp = self.root / "jobs" / self.job / "temporary" / f"pending-{token}.part"
        temp.write_bytes(b"PIZARRA1\n\x00")
        self.assertEqual(self.storage.incomplete(self.job), [{"token": token, "status": "incomplete"}])
        with self.assertRaises(IntegrityError):
            self.storage.commit_temporary(self.job, "tensors", "bad.bin", token)
        self.assertFalse((self.root / "jobs" / self.job / "tensors" / "bad.bin").exists())
        ref = self.storage.commit(self.job, "tensors", "ok.bin", b"integrity")
        target = self.root / "jobs" / self.job / ref
        content = target.read_bytes()
        target.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        with self.assertRaises(IntegrityError):
            self.storage.read(self.job, ref)

    def test_invalid_manifest_not_recovered(self):
        self.storage.create_workspace(self.job)
        another = str(uuid.uuid4())
        self.storage.create_workspace(another)
        manifest = self.root / "jobs" / another / "manifest.json"
        payload = json.loads(manifest.read_text())
        payload["job_id"] = self.job
        manifest.write_text(json.dumps(payload))
        self.assertEqual([item["job_id"] for item in self.storage.recover_manifests()], [self.job])

    def test_quota_and_free_margin_are_checked_before_writing_payload(self):
        self.storage.create_workspace(self.job)
        used = self.storage.stats()["managed_bytes"]
        limited = PizarraStorage(PizarraConfig(enabled=True, root=self.root,
                                  quota_bytes=used + 400, min_free_bytes=0))
        with self.assertRaises(QuotaExceeded):
            limited.commit(self.job, "tensors", "large.bin", b"x" * 500)
        self.assertEqual(limited.incomplete(self.job), [])
        reserve = PizarraStorage(PizarraConfig(enabled=True, root=self.root,
                                  quota_bytes=10**9, min_free_bytes=10**20))
        with self.assertRaises(QuotaExceeded):
            reserve.commit(self.job, "tensors", "no-space.bin", b"a")

    def test_concurrent_storages_share_quota_lock(self):
        self.storage.create_workspace(self.job)
        current = self.storage.stats()["managed_bytes"]
        cfg = PizarraConfig(enabled=True, root=self.root, quota_bytes=current + 1100, min_free_bytes=0)
        def write(i):
            try:
                return PizarraStorage(cfg).commit(self.job, "tensors", f"tensor-{i}.bin", b"x" * 500)
            except QuotaExceeded:
                return None
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(write, range(6)))
        self.assertEqual(sum(result is not None for result in results), 1)
        self.assertLessEqual(PizarraStorage(cfg).stats()["managed_bytes"], cfg.quota_bytes)

    def test_cleanup_only_our_leases_and_never_confirmed_or_foreign_files(self):
        self.storage.create_workspace(self.job)
        ref = self.storage.commit(self.job, "tensors", "keep.bin", b"keep")
        token = self.storage.write_temporary(self.job, b"temp")
        temp_dir = self.root / "jobs" / self.job / "temporary"
        foreign = temp_dir / "foreign.txt"
        foreign.write_text("do not delete")
        with patch("core.pizarra.storage.time.time", return_value=time.time() + 100):
            self.assertEqual(self.storage.cleanup_abandoned(self.job, older_than_seconds=10), 1)
        self.assertFalse((temp_dir / f"pending-{token}.part").exists())
        self.assertTrue(foreign.exists())
        self.assertEqual(self.storage.read(self.job, ref), b"keep")

    def test_escape_references_and_symlink_directories_are_rejected(self):
        self.storage.create_workspace(self.job)
        for name in ("../foreign", "/tmp/file", ".hidden", "a/../../b"):
            with self.assertRaises(UnsafePath):
                self.storage.commit(self.job, "tensors", name, b"test")
        with tempfile.TemporaryDirectory() as other:
            target = self.root / "jobs" / self.job / "tensors"
            target.rmdir()
            target.symlink_to(other, target_is_directory=True)
            with self.assertRaises(UnsafePath):
                self.storage.commit(self.job, "tensors", "escape.bin", b"test")
            self.assertEqual(list(Path(other).iterdir()), [])

    def test_mount_loss_is_detected_before_commit(self):
        self.storage.create_workspace(self.job)
        token = self.storage.write_temporary(self.job, b"pending")
        self.mounts.pop(self.root)
        with self.assertRaises(StorageUnavailable):
            self.storage.commit_temporary(self.job, "tensors", "result.bin", token)
        self.assertFalse((self.root / "jobs" / self.job / "tensors" / "result.bin").exists())

    def test_unrelated_lock_file_is_never_overwritten(self):
        lock = self.root / ".pizarra.lock"
        lock.write_bytes(b"another application")
        with self.assertRaises(UnsafePath):
            self.storage.create_workspace(self.job)
        self.assertEqual(lock.read_bytes(), b"another application")


class ConfigTests(unittest.TestCase):
    def test_auto_requires_recognized_authorization(self):
        env = {"BUTTERVISION_PIZARRA_MODE": "AUTO", "BUTTERVISION_PIZARRA_ENABLED": "AUTO"}
        with patch.dict(os.environ, env, clear=True), patch("core.pizarra.config._auto_authorized", return_value=False):
            self.assertFalse(PizarraConfig.from_env().enabled)
        with patch.dict(os.environ, env, clear=True), patch("core.pizarra.config._auto_authorized", return_value=True):
            self.assertTrue(PizarraConfig.from_env().enabled)

    def test_defaults_and_parsing(self):
        with patch("core.pizarra.config._auto_authorized", return_value=False), patch.dict(os.environ, {}, clear=True):
            cfg = PizarraConfig.from_env()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.root, Path("/mnt/pizarra"))
        self.assertEqual(cfg.quota_bytes, 400_000_000_000)
        with patch.dict(os.environ, {"BUTTERVISION_PIZARRA_ENABLED": "true", "BUTTERVISION_PIZARRA_QUOTA_GB": "1.5"}, clear=True):
            cfg = PizarraConfig.from_env()
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.quota_bytes, 1_500_000_000)

    def test_invalid_values_fail_closed(self):
        for env in ({"BUTTERVISION_PIZARRA_ENABLED": "maybe"}, {"BUTTERVISION_PIZARRA_ROOT": "relative"},
                    {"BUTTERVISION_PIZARRA_QUOTA_GB": "nan"}, {"BUTTERVISION_PIZARRA_MIN_FREE_GB": "-1"}):
            enabled = {"BUTTERVISION_PIZARRA_ENABLED": "true", **env}
            with self.subTest(env=env), patch.dict(os.environ, enabled, clear=True):
                with self.assertRaises(ValueError):
                    PizarraConfig.from_env()

    def test_disabled_ignores_unused_invalid_settings(self):
        with patch.dict(os.environ, {"BUTTERVISION_PIZARRA_ENABLED": "false",
                                    "BUTTERVISION_PIZARRA_ROOT": "relative",
                                    "BUTTERVISION_PIZARRA_QUOTA_GB": "not-a-number",
                                    "BUTTERVISION_PIZARRA_METRICS_ENABLED": "invalid"}, clear=True):
            self.assertFalse(PizarraConfig.from_env().enabled)


if __name__ == "__main__":
    unittest.main()
