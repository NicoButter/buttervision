"""CPU-only tensor persistence and out-of-core operator tests."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import uuid

import numpy as np

from core.pizarra.config import PizarraConfig
from core.pizarra.storage import IntegrityError, PizarraStorage, QuotaExceeded, StorageUnavailable
from core.pizarra.tensor_store import BufferLimiter, BufferPolicy, TensorStore
from core.pizarra.tensor_ops import matmul, reduce_sum, scale


class TensorFixture:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        device = self.root.stat().st_dev
        self.mounts = {self.root: ("test-mount", f"{os.major(device)}:{os.minor(device)}"),
                       Path("/"): ("root", "999:999")}
        def mounted(path):
            try:
                return self.mounts[Path(path)]
            except KeyError as error:
                raise StorageUnavailable("not mounted") from error
        from unittest.mock import patch
        self.mount_patch = patch("core.pizarra.storage._mount_identity", side_effect=mounted)
        self.mount_patch.start()
        self.addCleanup(self.mount_patch.stop)
        config = PizarraConfig(enabled=True, root=self.root, quota_bytes=1_000_000, min_free_bytes=0)
        self.storage = PizarraStorage(config)
        self.job = str(uuid.uuid4())
        self.storage.create_workspace(self.job)
        self.policy = BufferPolicy(max_fragment_bytes=128, max_simultaneous_buffers=6,
                                   result_partial_bytes=128, safety_margin_bytes=128,
                                   total_budget_bytes=1_024)
        self.store = TensorStore(self.storage, self.policy)

class TensorStoreTests(TensorFixture, unittest.TestCase):
    def test_streams_blocks_reopens_and_preserves_dtype_and_order(self):
        source = np.arange(35, dtype=np.float32).reshape(5, 7)
        ref = self.store.write_tensor(self.job, source.shape, "float32",
                                      (source.reshape(-1)[i:i + 9] for i in range(0, source.size, 9)),
                                      block_elements=8)
        self.assertEqual(ref.block_count, 5)
        reopened = TensorStore(self.storage, self.policy).open_tensor(self.job, ref.reference)
        result = np.concatenate(list(self.store.read_blocks(reopened))).reshape(reopened.shape)
        np.testing.assert_array_equal(result, source)
        half = source.astype(np.float16)
        half_ref = self.store.write_tensor(self.job, half.shape, "float16",
                                           (half.reshape(-1)[i:i + 5] for i in range(0, half.size, 5)),
                                           block_elements=7)
        np.testing.assert_array_equal(np.concatenate(list(self.store.iter_chunks(half_ref))), half.reshape(-1))

    def test_2d_region_reads_across_linear_block_boundaries(self):
        source = np.arange(42, dtype=np.float32).reshape(6, 7)
        flat = source.reshape(-1)
        ref = self.store.write_tensor(self.job, source.shape, "float32",
                                      (flat[i:i + 24] for i in range(0, flat.size, 24)), block_elements=5)
        np.testing.assert_array_equal(self.store.read_block(ref, 1, 5, 2, 7), source[1:5, 2:7])

    def test_checksum_corruption_is_rejected(self):
        source = np.arange(16, dtype=np.float32)
        ref = self.store.write_tensor(self.job, (16,), "float32", [source[:8], source[8:]], block_elements=8)
        path = self.root / "jobs" / self.job / "tensors" / self.store._block_name(ref.tensor_id, 0)
        content = bytearray(path.read_bytes())
        content[-1] ^= 0xFF
        path.write_bytes(content)
        with self.assertRaises(IntegrityError):
            list(self.store.read_blocks(ref))

    def test_interrupted_tensor_can_be_reopened_and_resumed(self):
        tensor_id = "a" * 32
        with self.assertRaisesRegex(ValueError, "no completan"):
            self.store.write_tensor(self.job, (8,), "float32",
                                    [np.array([1, 2], dtype=np.float32)], block_elements=4,
                                    tensor_id=tensor_id)
        restarted = TensorStore(PizarraStorage(self.storage.config), self.policy)
        ref = restarted.open_tensor(self.job, f"tensor:{tensor_id}")
        self.assertEqual(restarted.missing_blocks(ref), [0, 1])
        with self.assertRaisesRegex(IntegrityError, "Falta el bloque"):
            list(restarted.read_blocks(ref))
        restarted.write_block(ref, 0, np.array([1, 2, 3, 4], dtype=np.float32))
        restarted.write_block(ref, 1, np.array([5, 6, 7, 8], dtype=np.float32))
        self.assertEqual(restarted.missing_blocks(ref), [])
        np.testing.assert_array_equal(np.concatenate(list(restarted.read_blocks(ref))), np.arange(1, 9, dtype=np.float32))

    def test_shapes_dtype_buffer_limits_and_quota_are_enforced(self):
        with self.assertRaises(ValueError):
            self.store.write_tensor(self.job, (0,), "float32", [])
        with self.assertRaises(ValueError):
            self.store.write_tensor(self.job, (2,), "float64", [])
        with self.assertRaises(ValueError):
            self.store.write_tensor(self.job, (100,), "float32", [np.zeros(100, dtype=np.float32)])
        with self.assertRaises(ValueError):
            self.store.open_tensor(self.job, "tensor:../../outside")
        limiter = BufferLimiter(self.policy)
        with self.assertRaises(MemoryError):
            with limiter.reserve(self.policy.max_fragment_bytes + 1):
                pass
        with limiter.reserve(64):
            self.assertEqual(limiter.snapshot()["managed_current_bytes"], 64)
        self.assertEqual(limiter.snapshot()["managed_current_bytes"], 0)
        current = self.storage.stats()["managed_bytes"]
        self.storage.config = PizarraConfig(enabled=True, root=self.root,
                                            quota_bytes=current + 128, min_free_bytes=0)
        with self.assertRaises(QuotaExceeded):
            self.store.write_tensor(self.job, (32,), "float32",
                                    [np.arange(32, dtype=np.float32)], block_elements=32)


class TensorOpsTests(TensorFixture, unittest.TestCase):
    def test_tensor_larger_than_buffer_budget_streams_and_processes(self):
        elements = 4_096
        blocks = (np.ones(min(32, elements - offset), dtype=np.float32)
                  for offset in range(0, elements, 32))
        ref = self.store.write_tensor(self.job, (elements,), "float32", blocks, block_elements=32)
        self.assertGreater(elements * 4, self.policy.total_budget_bytes)
        total, read_metrics = reduce_sum(self.store, ref)
        result_ref, transform_metrics = scale(self.store, ref, 2.0)
        transformed_total, _ = reduce_sum(self.store, result_ref)
        self.assertEqual(total, float(elements))
        self.assertEqual(transformed_total, float(elements * 2))
        self.assertLessEqual(read_metrics["managed_peak_bytes"], self.policy.max_fragment_bytes)
        self.assertLessEqual(transform_metrics["managed_peak_buffers"], self.policy.max_simultaneous_buffers)
        self.assertLessEqual(transform_metrics["managed_partial_peak_bytes"], self.policy.result_partial_bytes)

    def test_scale_and_incremental_sum_match_numpy(self):
        source = np.linspace(-2, 3, 35, dtype=np.float32).reshape(5, 7)
        flat = source.reshape(-1)
        ref = self.store.write_tensor(self.job, source.shape, "float32", (flat[i:i + 12] for i in range(0, flat.size, 12)), block_elements=6)
        result_ref, metrics = scale(self.store, ref, 2.5)
        result = np.concatenate(list(self.store.read_blocks(result_ref))).reshape(source.shape)
        np.testing.assert_allclose(result, source * 2.5, rtol=0, atol=1e-6)
        total, sum_metrics = reduce_sum(self.store, ref)
        self.assertAlmostEqual(total, float(np.sum(source, dtype=np.float64)), places=10)
        self.assertLessEqual(metrics["managed_peak_bytes"], self.policy.max_fragment_bytes * self.policy.max_simultaneous_buffers)
        self.assertIn("rss_process_after_bytes", sum_metrics)

    def test_tiled_matmul_irregular_dimensions_matches_numpy(self):
        rng = np.random.default_rng(3)
        left = rng.normal(size=(5, 7)).astype(np.float32)
        right = rng.normal(size=(7, 6)).astype(np.float32)
        left_flat, right_flat = left.reshape(-1), right.reshape(-1)
        left_ref = self.store.write_tensor(self.job, left.shape, "float32", (left_flat[i:i + 24] for i in range(0, left_flat.size, 24)), block_elements=8)
        right_ref = self.store.write_tensor(self.job, right.shape, "float32", (right_flat[i:i + 24] for i in range(0, right_flat.size, 24)), block_elements=9)
        result_ref, metrics = matmul(self.store, left_ref, right_ref, block_size=3)
        result = np.concatenate(list(self.store.read_blocks(result_ref))).reshape(5, 6)
        np.testing.assert_allclose(result, left @ right, rtol=1e-5, atol=1e-5)
        self.assertLessEqual(metrics["managed_peak_buffers"], self.policy.max_simultaneous_buffers)

    def test_float16_matmul_uses_float32_result_and_cpu_only(self):
        left = np.arange(15, dtype=np.float16).reshape(3, 5)
        right = np.arange(20, dtype=np.float16).reshape(5, 4)
        lref = self.store.write_tensor(self.job, left.shape, "float16", [left.reshape(-1)], block_elements=6)
        rref = self.store.write_tensor(self.job, right.shape, "float16", [right.reshape(-1)], block_elements=7)
        result_ref, metrics = matmul(self.store, lref, rref, block_size=2)
        self.assertEqual(result_ref.dtype, "float32")
        actual = np.concatenate(list(self.store.read_blocks(result_ref))).reshape(3, 4)
        np.testing.assert_allclose(actual, left.astype(np.float32) @ right.astype(np.float32), rtol=0, atol=1e-5)
        self.assertIsNone(metrics["vram"]["device_total_bytes"])
        self.assertGreaterEqual(metrics["managed_partial_peak_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
