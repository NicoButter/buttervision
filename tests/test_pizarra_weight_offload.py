import os
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch

import torch

from core.pizarra.config import PizarraConfig
from core.pizarra.storage import PizarraStorage, StorageUnavailable, UnsafePath, QuotaExceeded
from core.pizarra.tensor_store import BufferPolicy
from core.pizarra.weight_offload import WeightOffload


class OffloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        dev = self.root.stat().st_dev
        def mounted(path):
            if Path(path) == self.root:
                return ('test', f'{os.major(dev)}:{os.minor(dev)}')
            if Path(path) == Path('/'):
                return ('root', '999:999')
            raise StorageUnavailable('unmounted')
        self.mount = patch('core.pizarra.storage._mount_identity', side_effect=mounted)
        self.mount.start()
        self.addCleanup(self.mount.stop)
        self.storage = PizarraStorage(PizarraConfig(enabled=True, root=self.root, quota_bytes=1000000, min_free_bytes=0))
        self.adapter = WeightOffload(self.storage, policy=BufferPolicy(128, 6, 128, 128, 1024))

    def test_roundtrip_and_cleanup(self):
        expected = {'block.a': torch.arange(100, dtype=torch.float32),
                    'block.b': torch.arange(30, dtype=torch.float16).reshape(5, 6)}
        weights = {k: v.clone() for k, v in expected.items()}
        session = self.adapter.offload(weights)
        self.assertEqual(weights, {})
        self.assertGreater(session['logical_released_bytes'], 0)
        self.assertGreater(session['write_bytes'], 0)
        self.assertGreater(session['verify_read_bytes'], 0)
        self.assertGreater(session['before']['rss_bytes'], 0)
        restored, metrics = self.adapter.restore(session)
        for name in expected:
            self.assertTrue(torch.equal(expected[name], restored[name]))
            self.assertEqual(expected[name].dtype, restored[name].dtype)
            self.assertEqual(expected[name].shape, restored[name].shape)
        self.assertGreater(metrics['read_bytes'], 0)
        self.assertGreater(self.storage.cleanup_offload(session['job_id'], session['token']), 0)
        self.assertEqual(self.storage.cleanup_offload(session['job_id'], session['token']), 0)

    def test_foreign_and_wrong_token_never_delete(self):
        weights = {'a': torch.ones(4)}
        session = self.adapter.offload(weights)
        job = self.root / 'jobs' / session['job_id']
        with self.assertRaises(UnsafePath):
            self.storage.cleanup_offload(session['job_id'], str(uuid.uuid4()))
        foreign = job / 'tensors' / 'foreign'
        foreign.write_bytes(b'foreign')
        with self.assertRaises(UnsafePath):
            self.storage.cleanup_offload(session['job_id'], session['token'])
        self.assertEqual(foreign.read_bytes(), b'foreign')
        foreign.unlink()
        self.storage.cleanup_offload(session['job_id'], session['token'])

    def test_failure_preserves_source_and_quota(self):
        self.storage.config = PizarraConfig(enabled=True, root=self.root, quota_bytes=1300, min_free_bytes=0)
        weights = {'a': torch.ones(1000)}
        with self.assertRaises(QuotaExceeded):
            self.adapter.offload(weights)
        self.assertTrue(torch.equal(weights['a'], torch.ones(1000)))

    def test_interrupted_cleanup_is_retryable(self):
        session = self.adapter.offload({'a': torch.arange(100, dtype=torch.float32)})
        original = os.unlink
        calls = []
        def interrupted(name, *args, **kwargs):
            calls.append(name)
            if len(calls) == 2:
                raise OSError('interrupted')
            return original(name, *args, **kwargs)
        with patch('core.pizarra.storage.os.unlink', side_effect=interrupted):
            with self.assertRaises(StorageUnavailable):
                self.storage.cleanup_offload(session['job_id'], session['token'])
        self.assertGreater(self.storage.cleanup_offload(session['job_id'], session['token']), 0)
        self.assertEqual(self.storage.cleanup_offload(session['job_id'], session['token']), 0)

    def test_failed_verification_keeps_source(self):
        weights = {'a': torch.arange(10, dtype=torch.float32)}
        with patch.object(self.adapter, '_verify', side_effect=ValueError('corrupt')):
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                self.adapter.offload(weights)
        self.assertTrue(torch.equal(weights['a'], torch.arange(10, dtype=torch.float32)))

    def test_symlink_is_not_removed(self):
        session = self.adapter.offload({'a': torch.ones(4)})
        directory = self.root / 'jobs' / session['job_id'] / 'tensors'
        link = directory / 'foreign'
        link.symlink_to(self.root / 'outside')
        with self.assertRaises(UnsafePath):
            self.storage.cleanup_offload(session['job_id'], session['token'])
        self.assertTrue(link.is_symlink())
        link.unlink()
        self.storage.cleanup_offload(session['job_id'], session['token'])

    def test_mount_change_rejected(self):
        weights = {'a': torch.ones(4)}
        session = self.adapter.offload(weights)
        with patch('core.pizarra.storage._mount_identity', side_effect=StorageUnavailable('unmounted')):
            with self.assertRaises(StorageUnavailable):
                self.storage.cleanup_offload(session['job_id'], session['token'])
        self.assertGreater(self.storage.cleanup_offload(session['job_id'], session['token']), 0)

if __name__ == '__main__':
    unittest.main()
