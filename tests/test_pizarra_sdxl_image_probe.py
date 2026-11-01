from types import SimpleNamespace
from unittest.mock import patch
import torch
from PIL import Image

from scripts.pizarra_sdxl_image_probe import generate
from test_pizarra_weight_offload import OffloadTests


class TinyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_blocks = torch.nn.ModuleList([torch.nn.Module()])
        self.down_blocks[0].resnets = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

    def forward(self, x):
        return self.down_blocks[0].resnets[0](x)


class TinyPipeline:
    def __init__(self, fail=False):
        self.unet = TinyUNet().eval()
        self.vae = SimpleNamespace(decode=lambda x: x)
        self.fail = fail

    def encode_prompt(self):
        return torch.ones(1, 2)

    def __call__(self, **kwargs):
        x = self.unet(self.encode_prompt())
        if self.fail:
            raise RuntimeError('generation failed')
        self.vae.decode(x)
        return SimpleNamespace(images=[Image.new('RGB', (kwargs['width'], kwargs['height']))])


class ImageProbeTests(OffloadTests):
    def test_image_and_cleanup(self):
        pipe = TinyPipeline()
        events = []
        output = self.root / 'image.png'
        generate(pipe, self.storage, output, 64, emit_event=lambda name, **kw: events.append((name, kw)))
        with Image.open(output) as image:
            self.assertEqual(image.size, (64, 64))
        self.assertFalse(any(p.is_meta for p in pipe.unet.parameters()))
        self.assertIn('cleaned', [name for name, _ in events])
        end = next(data for name, data in events if name == 'generation_end')
        self.assertEqual((end['unet_forwards'], end['restores'], end['additional_write_bytes']), (1, 1, 0))
        with self.assertRaises(FileExistsError):
            generate(pipe, self.storage, output, 64, emit_event=lambda *a, **k: None)
        with Image.open(output) as image:
            self.assertEqual(image.size, (64, 64))

    def test_failure_recovers_and_preserves_session(self):
        pipe = TinyPipeline(fail=True)
        events = []
        output = self.root / 'failure.png'
        with self.assertRaisesRegex(RuntimeError, 'generation failed'):
            generate(pipe, self.storage, output, 64, emit_event=lambda name, **kw: events.append((name, kw)))
        self.assertFalse(output.exists())
        self.assertFalse(any(p.is_meta for p in pipe.unet.parameters()))
        session = next(data for name, data in events if name == 'session_preserved')
        self.assertGreater(self.storage.cleanup_offload(session['job_id'], session['token']), 0)
