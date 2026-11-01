import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import weakref
from unittest.mock import patch

import torch

from core.pizarra.config import PizarraConfig
from core.pizarra.storage import PizarraStorage, StorageUnavailable
from scripts import pizarra_weight_offload_probe as probe


class DiagnosticsTests(unittest.TestCase):
    def test_cgroup_mount_root_and_versions(self):
        v2 = '25 1 0:29 /user.slice /sys/fs/cgroup rw - cgroup2 cgroup rw\n'
        self.assertEqual(probe.cgroup_memory_path('0::/user.slice/app.scope\n', v2),
                         Path('/sys/fs/cgroup/app.scope/memory.current'))
        self.assertIsNone(probe.cgroup_memory_path('0::/other\n', v2))
        v1 = '25 1 0:29 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n'
        self.assertEqual(probe.cgroup_memory_path('4:cpu,memory:/test\n', v1),
                         Path('/sys/fs/cgroup/memory/test/memory.usage_in_bytes'))
        self.assertEqual(probe.cgroup_memory_path('0::/v2\n4:memory:/test\n', v2 + v1),
                         Path('/sys/fs/cgroup/memory/test/memory.usage_in_bytes'))
        escaped = '25 1 0:29 / /sys/fs/cgroup\\040space rw - cgroup2 cgroup rw\n'
        self.assertEqual(probe.cgroup_memory_path('0::/test\n', escaped),
                         Path('/sys/fs/cgroup space/test/memory.current'))

    def test_measurements_and_weakrefs(self):
        snapshot = probe.diagnostic_memory()
        self.assertGreater(snapshot['rss_bytes'], 0)
        self.assertEqual(set(snapshot), {'rss_bytes', 'uss_bytes', 'anonymous_bytes', 'peak_rss_bytes', 'cgroup_current_bytes'})
        tensor = torch.ones(8)
        refs = [weakref.ref(tensor)]
        self.assertEqual(probe.weakref_counts(refs)['alive'], 1)
        del tensor
        self.assertEqual(probe.weakref_counts(refs)['collected'], 1)

    def test_trim_unavailable(self):
        with patch.object(probe.platform, 'libc_ver', return_value=('other', '')):
            self.assertEqual(probe.trim_glibc(), {'available': False, 'result': None})

    def test_probe_roundtrip_compact_output_and_optional_trim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev = root.stat().st_dev
            def mounted(path):
                if Path(path) == root:
                    return ('test', f'{os.major(dev)}:{os.minor(dev)}')
                if Path(path) == Path('/'):
                    return ('root', '999:999')
                raise StorageUnavailable('unmounted')
            storage = PizarraStorage(PizarraConfig(enabled=True, root=root,
                                                   quota_bytes=1_000_000, min_free_bytes=0))
            for trim in (False, True):
                output = io.StringIO()
                argv = ['probe', '--execute', '--model', '/dummy.safetensors']
                if trim:
                    argv.append('--malloc-trim')
                with patch('core.pizarra.storage._mount_identity', side_effect=mounted), \
                     patch('core.pizarra.storage.PizarraStorage', return_value=storage), \
                     patch('core.model_catalog.approved_path', return_value=Path('/dummy.safetensors')), \
                     patch('core.model_catalog.checkpoint_family', return_value='sdxl'), \
                     patch('core.model_catalog.SDXL', 'sdxl'), \
                     patch('core.pizarra.weight_offload.load_unet_block', side_effect=lambda *_: {'weight': torch.arange(32.)}), \
                     patch.object(probe, 'trim_glibc', return_value={'available': True, 'result': 1}) as trimmer, \
                     patch.object(sys, 'argv', argv), contextlib.redirect_stdout(output):
                    self.assertEqual(probe.main(), 0)
                events = [json.loads(line) for line in output.getvalue().splitlines()]
                by_name = {event['event']: event for event in events}
                self.assertEqual(set(by_name), {'before_load', 'loaded', 'session', 'released', 'after_gc', 'restored', 'cleaned'} |
                                 ({'after_malloc_trim'} if trim else set()))
                self.assertEqual(trimmer.call_count, int(trim))
                self.assertEqual(by_name['released']['weakrefs']['alive'], 0)
                self.assertGreater(by_name['released']['logical_released_bytes'], 0)
                self.assertEqual(by_name['restored']['integrity'], 'sha256 exact bytes')
                self.assertGreater(by_name['cleaned']['records_removed'], 0)
                self.assertEqual(sum('token' in event for event in events), 1)
                self.assertNotIn('index', output.getvalue())
                for phase in ('before_load', 'loaded', 'released', 'after_gc', 'restored'):
                    self.assertGreater(by_name[phase]['memory']['rss_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
