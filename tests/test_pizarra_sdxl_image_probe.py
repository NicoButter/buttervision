from types import SimpleNamespace
from unittest.mock import patch
import torch
from PIL import Image

from scripts.pizarra_sdxl_image_probe import generate, prepare_embeddings, PROMPT
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
        self.text_encoder = torch.nn.Linear(2, 2)
        self.text_encoder_2 = torch.nn.Linear(2, 2)

    def encode_prompt(self, **kwargs):
        assert kwargs["prompt"] == PROMPT
        assert self.text_encoder is not None and self.text_encoder_2 is not None
        return torch.ones(1, 2), None, torch.ones(1, 2), None

    def __call__(self, **kwargs):
        assert self.text_encoder is None and self.text_encoder_2 is None
        assert kwargs["pooled_prompt_embeds"] is not None
        assert "prompt" not in kwargs
        x = self.unet(kwargs["prompt_embeds"])
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
        names = [name for name, _ in events]
        self.assertLess(names.index('memory_before_prompt_encoding'), names.index('memory_after_prompt_encoding'))
        self.assertLess(names.index('memory_after_prompt_encoding'), names.index('memory_after_encoder_release'))
        self.assertLess(names.index('memory_after_encoder_release'), names.index('memory_before_restore'))
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

    def test_constructor_failure_removes_reservation(self):
        output = self.root / 'constructor.png'
        with patch('core.pizarra.automatic_offload.AutomaticOffload', side_effect=RuntimeError('constructor')):
            with self.assertRaisesRegex(RuntimeError, 'constructor'):
                generate(TinyPipeline(), self.storage, output, 64, emit_event=lambda *a, **k: None)
        self.assertFalse(output.exists())

    def test_encoder_release_reduces_resident_memory(self):
        import gc
        import psutil
        import weakref
        pipe = TinyPipeline()
        pipe.text_encoder = torch.nn.Linear(4096, 4096, bias=False)
        pipe.text_encoder_2 = torch.nn.Linear(4096, 4096, bias=False)
        references = [weakref.ref(pipe.text_encoder), weakref.ref(pipe.text_encoder_2)]
        gc.collect()
        before = psutil.Process().memory_info().rss
        embeddings = prepare_embeddings(pipe, lambda *a, **k: None)
        after = psutil.Process().memory_info().rss
        self.assertTrue(all(ref() is None for ref in references))
        self.assertGreater(before - after, 64 * 1024**2)
        self.assertIsNotNone(embeddings['pooled_prompt_embeds'])

    def test_installed_diffusers_generates_with_embeddings_and_no_encoders(self):
        from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
        unet = UNet2DConditionModel(
            sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
            block_out_channels=(32,), down_block_types=('CrossAttnDownBlock2D',),
            up_block_types=('CrossAttnUpBlock2D',), cross_attention_dim=16,
            attention_head_dim=4, norm_num_groups=8, addition_embed_type='text_time',
            addition_time_embed_dim=4, projection_class_embeddings_input_dim=32)
        vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
                            block_out_channels=(32,), down_block_types=('DownEncoderBlock2D',),
                            up_block_types=('UpDecoderBlock2D',), layers_per_block=1, norm_num_groups=8)
        pipe = StableDiffusionXLPipeline(vae=vae, text_encoder=None, text_encoder_2=None,
                                        tokenizer=None, tokenizer_2=None, unet=unet,
                                        scheduler=EulerDiscreteScheduler())
        pipe.set_progress_bar_config(disable=True)
        result = pipe(prompt_embeds=torch.ones(1, 4, 16), pooled_prompt_embeds=torch.ones(1, 8),
                      height=8, width=8, num_inference_steps=1, guidance_scale=1.0,
                      generator=torch.Generator().manual_seed(507), output_type='pil')
        self.assertEqual(result.images[0].size, (8, 8))
