from unittest.mock import patch
import torch

from core.pizarra.automatic_offload import AutomaticOffload
from test_pizarra_weight_offload import OffloadTests


class AutomaticTests(OffloadTests):
    def test_roundtrip_and_failures(self):
        torch.manual_seed(3)
        root = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Conv2d(2, 4, 3, padding=1), torch.nn.SiLU()), torch.nn.Conv2d(4, 2, 1)).eval()
        x = torch.randn(1, 2, 8, 8)
        with torch.inference_mode():
            expected = root(x)
            original_forward = root[0].forward
            controller = AutomaticOffload(root, '0', self.storage)
            controller.install()
            with patch.object(controller.adapter.backend, 'offload', side_effect=AssertionError('rewrite')):
                for _ in range(2):
                    torch.testing.assert_close(root(x), expected, rtol=0, atol=0)
                    self.assertTrue(all(p.is_meta for p in root[0].parameters()))
            with patch.object(controller.adapter, 'restore', side_effect=RuntimeError('restore')):
                with self.assertRaisesRegex(RuntimeError, 'restore'):
                    root(x)
            self.assertEqual(controller.adapter.state, 'released')
            with patch.object(root[0][0], 'forward', side_effect=RuntimeError('forward')):
                with self.assertRaisesRegex(RuntimeError, 'forward'):
                    root(x)
            self.assertEqual(controller.adapter.state, 'released')
            torch.testing.assert_close(root(x), expected, rtol=0, atol=0)
            controller.close()
            self.assertEqual(root[0].forward, original_forward)
            torch.testing.assert_close(root(x), expected, rtol=0, atol=0)

    def test_concurrent_access_and_foreign_hooks_rejected(self):
        root = torch.nn.Sequential(torch.nn.Linear(2, 2)).eval()
        controller = AutomaticOffload(root, '0', self.storage)
        with torch.inference_mode():
            controller.install()
            controller.lock.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, 'concurrente'):
                    root(torch.ones(1, 2))
            finally:
                controller.lock.release()
            hook = root[0].register_forward_hook(lambda *args: None)
            try:
                with self.assertRaisesRegex(ValueError, 'Hooks'):
                    root(torch.ones(1, 2))
            finally:
                hook.remove()
            controller.close()

    def test_restrictions(self):
        root = torch.nn.Sequential(torch.nn.Linear(2, 2)).eval()
        controller = AutomaticOffload(root, '0', self.storage)
        with torch.inference_mode():
            controller.install()
            with self.assertRaisesRegex(RuntimeError, 'directo'):
                root[0](torch.ones(1, 2))
            root.train()
            with self.assertRaisesRegex(RuntimeError, 'evaluación'):
                root(torch.ones(1, 2))
            root.eval()
        with self.assertRaisesRegex(RuntimeError, 'inference_mode'):
            root(torch.ones(1, 2))
        controller.close()
