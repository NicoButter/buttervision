import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from core.pizarra.config import PizarraConfig
from core.pizarra.storage import PizarraStorage, StorageUnavailable
from core.sdxl_progressive_loader import load_progressive
from scripts.pizarra_unet_probe import check_scope, execute_experiment, main
from test_sdxl_progressive import ProgressiveTests


class UNetProbeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ProgressiveTests('test_complete_real_load_values_dtype_and_unchanged_file')
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
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
        mock = patch('core.pizarra.storage._mount_identity', side_effect=mounted)
        mock.start()
        self.addCleanup(mock.stop)
        self.storage = PizarraStorage(PizarraConfig(enabled=True, root=self.root, quota_bytes=10000000, min_free_bytes=0))
        self.pipe = load_progressive(self.fixture.path, self.fixture.root, dtype=torch.float32)
        self.events = []

    def event(self, name, **values):
        self.events.append((name, values))

    def test_progressive_unet_block_and_full_forward(self):
        block = self.pipe.unet.down_blocks[0].resnets[0]
        execute_experiment(self.pipe.unet, self.storage, full_forward=True, emit_event=self.event)
        self.assertIs(block, self.pipe.unet.down_blocks[0].resnets[0])
        events = dict(self.events)
        self.assertTrue(events['released']['all_meta'])
        self.assertTrue(events['released']['same_module'])
        self.assertTrue(events['block_equivalence']['exact'])
        self.assertTrue(events['unet_equivalence']['exact'])
        self.assertGreater(events['persisted']['pizarra_write_bytes'], 0)
        self.assertGreater(events['restored']['pizarra_total_read_bytes'], 0)
        self.assertIn('cleaned', events)

    def test_restore_failure_preserves_session(self):
        with patch('core.pizarra.module_offload.ModuleOffload.restore', side_effect=RuntimeError('restore failed')):
            with self.assertRaisesRegex(RuntimeError, 'restore failed'):
                execute_experiment(self.pipe.unet, self.storage, emit_event=self.event)
        events = dict(self.events)
        self.assertIn('recovery_failed_session_preserved', events)
        self.assertNotIn('cleaned', events)
        session = events['session_preserved']
        self.assertTrue((self.root / 'jobs' / session['job_id']).exists())
        self.storage.cleanup_offload(session['job_id'], session['token'])

    def test_failure_after_release_restores_and_preserves_session(self):
        def failing_event(name, **values):
            self.event(name, **values)
            if name == 'released':
                raise RuntimeError('injected failure')
        with self.assertRaisesRegex(RuntimeError, 'injected failure'):
            execute_experiment(self.pipe.unet, self.storage, emit_event=failing_event)
        events = dict(self.events)
        self.assertIn('exception_recovery_restored', events)
        self.assertNotIn('cleaned', events)
        self.assertFalse(any(t.is_meta for t in self.pipe.unet.down_blocks[0].resnets[0].parameters()))
        session = events['session_preserved']
        self.storage.cleanup_offload(session['job_id'], session['token'])

    def test_hooks_rejected_before_session(self):
        handle = self.pipe.unet.register_forward_pre_hook(lambda *args: None)
        try:
            with self.assertRaisesRegex(ValueError, 'Hooks'):
                execute_experiment(self.pipe.unet, self.storage, emit_event=self.event)
            self.assertEqual(self.events, [])
        finally:
            handle.remove()


class ScopeTests(unittest.TestCase):
    def test_rejects_worker_outside_scope(self):
        with self.assertRaises((RuntimeError, OSError)):
            check_scope('pizarra-unet-unrelated.scope', 1024**3)

    def test_scope_requires_finite_memory_and_zero_swap(self):
        unit = 'pizarra-unet-test.scope'
        def run(maximum, swap):
            values = {'/proc/self/cgroup': '0::/user.slice/' + unit,
                      '/proc/self/mountinfo': '', '/sys/test/memory.max': maximum,
                      '/sys/test/memory.swap.max': swap}
            with patch.object(Path, 'read_text', lambda path, *a, **k: values[str(path)]), \
                 patch('scripts.pizarra_unet_probe.cgroup_memory_path', return_value=Path('/sys/test/memory.current')):
                return check_scope(unit, 1024)
        self.assertEqual(run('1024', '0')['memory_max_bytes'], 1024)
        for maximum, swap in [('max', '0'), ('1025', '0'), ('1024', 'max')]:
            with self.assertRaises(RuntimeError):
                run(maximum, swap)

    def test_cli_requires_opt_in(self):
        with self.assertRaises(SystemExit):
            main([])
