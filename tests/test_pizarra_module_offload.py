import unittest
from unittest.mock import patch
import torch

from test_pizarra_weight_offload import OffloadTests
from core.pizarra.module_offload import ModuleOffload


class ModuleTests(OffloadTests):
    def test_active_block_roundtrip(self):
        torch.manual_seed(42)
        root = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Conv2d(2, 4, 3, padding=1), torch.nn.GroupNorm(2, 4), torch.nn.SiLU()))
        x = torch.randn(1, 2, 8, 8)
        with torch.inference_mode():
            expected = root[0](x)
        adapter = ModuleOffload(root, '0', self.storage)
        session = adapter.persist()
        adapter.release()
        self.assertTrue(all(p.is_meta for p in root[0].parameters()))
        actual = adapter.run(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with patch.object(adapter.backend, 'offload', side_effect=AssertionError('rewritten')):
            adapter.release()
            torch.testing.assert_close(adapter.run(x), expected, rtol=0, atol=0)
        self.assertGreater(session['write_bytes'], 0)
        adapter.cleanup()

    def test_hook_and_shared_storage_rejected(self):
        root = torch.nn.Sequential(torch.nn.Linear(4, 4))
        handle = root.register_forward_pre_hook(lambda *args: None)
        with self.assertRaisesRegex(ValueError, 'Hooks'):
            ModuleOffload(root, '0', self.storage)
        handle.remove()
        root.append(root[0])
        with self.assertRaisesRegex(ValueError, 'compartido'):
            ModuleOffload(root, '0', self.storage)

    def test_modified_weights_cannot_be_released(self):
        root = torch.nn.Sequential(torch.nn.Linear(4, 4))
        adapter = ModuleOffload(root, '0', self.storage)
        adapter.persist()
        with torch.no_grad():
            root[0].weight.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'cambió'):
            adapter.release()
        self.assertFalse(root[0].weight.is_meta)
        adapter.cleanup()

class DiffusersBlockTests(OffloadTests):
    def test_resnet_block_execution(self):
        from diffusers.models.resnet import ResnetBlock2D
        torch.manual_seed(7)
        block = ResnetBlock2D(in_channels=32, out_channels=32, temb_channels=128, groups=8).eval()
        root = torch.nn.Sequential(block)
        x, temb = torch.randn(1, 32, 8, 8), torch.randn(1, 128)
        with torch.inference_mode():
            expected = block(x, temb)
        adapter = ModuleOffload(root, '0', self.storage)
        adapter.persist()
        adapter.release()
        actual = adapter.run(x, temb)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        adapter.cleanup()
