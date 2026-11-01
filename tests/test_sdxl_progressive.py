"""Small real safetensors and complete models; no production checkpoint loading."""
import gc
import hashlib
import json
import os
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
from diffusers.loaders import single_file_utils as official
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from core.sdxl_progressive_loader import (
    TensorRef, ProgressiveLoadError, read_index, conversion_plan, empty_models,
    load_progressive, assert_initialized, progressive_estimate, validate_source,
)


def original_fixture(models):
    """Independent inverse for the tiny base architecture, including conv attention."""
    result = {}
    resnet = {'norm1': 'in_layers.0', 'conv1': 'in_layers.2',
              'time_emb_proj': 'emb_layers.1', 'norm2': 'out_layers.0',
              'conv2': 'out_layers.3', 'conv_shortcut': 'skip_connection'}
    for key, value in models['unet'].state_dict().items():
        old = key
        for target, source in official.DIFFUSERS_TO_LDM_MAPPING['unet']['layers'].items():
            if key == target:
                old = source
        for target, source in official.DIFFUSERS_TO_LDM_MAPPING['unet']['addition_embed_type'].items():
            if key == target:
                old = source
        for kind, stem, indices in [('down_blocks.0', 'input_blocks', [1]),
                                    ('up_blocks.0', 'output_blocks', [0, 1])]:
            for j, index in enumerate(indices):
                for component, number in [('resnets', 0), ('attentions', 1)]:
                    prefix = f'{kind}.{component}.{j}.'
                    if key.startswith(prefix):
                        tail = key[len(prefix):]
                        if component == 'resnets':
                            a, b = tail.split('.', 1)
                            tail = resnet.get(a, a) + '.' + b
                        old = f'{stem}.{index}.{number}.{tail}'
        for component, j, index in [('resnets', 0, 0), ('attentions', 0, 1), ('resnets', 1, 2)]:
            prefix = f'mid_block.{component}.{j}.'
            if key.startswith(prefix):
                tail = key[len(prefix):]
                if component == 'resnets':
                    a, b = tail.split('.', 1)
                    tail = resnet.get(a, a) + '.' + b
                old = f'middle_block.{index}.{tail}'
        result['model.diffusion_model.' + old] = value.clone()
    for key, value in models['vae'].state_dict().items():
        old = official.DIFFUSERS_TO_LDM_MAPPING['vae'].get(key, key)
        for target, source in official.DIFFUSERS_TO_LDM_MAPPING['vae'].items():
            if key == target:
                old = source
        old = old.replace('encoder.down_blocks.0.resnets.', 'encoder.down.0.block.')
        old = old.replace('decoder.up_blocks.0.resnets.', 'decoder.up.0.block.')
        old = old.replace('mid_block.resnets.0.', 'mid.block_1.').replace('mid_block.resnets.1.', 'mid.block_2.')
        old = old.replace('conv_shortcut.', 'nin_shortcut.')
        if 'mid_block.attentions.0.' in old:
            old = old.replace('mid_block.attentions.0.', 'mid.attn_1.')
            for a, b in [('group_norm', 'norm'), ('to_q', 'q'), ('to_k', 'k'), ('to_v', 'v'), ('to_out.0', 'proj_out')]:
                old = old.replace(a, b)
            if value.ndim == 2:
                value = value[:, :, None, None]
        result['first_stage_model.' + old] = value.clone()
    for key, value in models['text_encoder'].state_dict().items():
        result['conditioner.embedders.0.transformer.' + key] = value.clone()
    state = models['text_encoder_2'].state_dict()
    prefix = 'conditioner.embedders.1.model.'
    for target, source in official.DIFFUSERS_TO_LDM_MAPPING['openclip']['layers'].items():
        value = state[target]
        result[prefix + source] = value.T.contiguous() if source == 'text_projection' else value.clone()
    for key, value in state.items():
        if not key.startswith('text_model.encoder.layers.'):
            continue
        old = key.replace('text_model.encoder.layers.', 'transformer.resblocks.')
        for a, b in [('layer_norm1', 'ln_1'), ('layer_norm2', 'ln_2'), ('.fc1.', '.c_fc.'),
                     ('.fc2.', '.c_proj.'), ('.self_attn', '.attn')]:
            old = old.replace(a, b)
        if any('.' + q + '_proj.' in old for q in 'qkv'):
            if '.q_proj.' not in old:
                continue
            packed = torch.cat([state[key.replace('.q_proj.', '.' + q + '_proj.')] for q in 'qkv'])
            old = old.replace('.q_proj.weight', '.in_proj_weight').replace('.q_proj.bias', '.in_proj_bias')
            result[prefix + old] = packed
        else:
            result[prefix + old] = value.clone()
    return result


class ProgressiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {'BUTTERVISION_CHECKPOINT_DIRS': str(self.root)})
        self.env.start()
        torch.manual_seed(7)
        clip = CLIPTextConfig(vocab_size=8, hidden_size=8, intermediate_size=16,
                              num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8, projection_dim=8)
        self.models = {
            'unet': UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
                down_block_types=('CrossAttnDownBlock2D',), up_block_types=('CrossAttnUpBlock2D',),
                block_out_channels=(8,), layers_per_block=1, cross_attention_dim=2048,
                attention_head_dim=2, norm_num_groups=4, addition_embed_type='text_time',
                addition_time_embed_dim=2, projection_class_embeddings_input_dim=20),
            'vae': AutoencoderKL(block_out_channels=(8,), norm_num_groups=4, layers_per_block=1),
            'text_encoder': CLIPTextModel(clip),
            'text_encoder_2': CLIPTextModelWithProjection(clip),
        }
        for name, model in self.models.items():
            if name.startswith('text'):
                model.config.save_pretrained(self.root / name)
            else:
                model.save_config(self.root / name)
        EulerDiscreteScheduler().save_pretrained(self.root / 'scheduler')
        vocab = self.root / 'vocab.json'; vocab.write_text(json.dumps({'<|startoftext|>': 0, '<|endoftext|>': 1}))
        merges = self.root / 'merges.txt'; merges.write_text('#version: 0.2\n')
        tok = CLIPTokenizer(str(vocab), str(merges))
        for name in ('tokenizer', 'tokenizer_2'):
            tok.save_pretrained(self.root / name)
        self.weights = original_fixture(self.models)
        self.path = self.root / 'tiny.safetensors'
        save_file(self.weights, self.path)

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_complete_real_load_values_dtype_and_unchanged_file(self):
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with patch('torch.load', side_effect=AssertionError('monolithic torch.load')), \
             patch('diffusers.StableDiffusionXLPipeline.from_single_file', side_effect=AssertionError('single_file')):
            pipe = load_progressive(self.path, self.root)
        loaded = {name: getattr(pipe, name) for name in self.models}
        assert_initialized(loaded)
        for name, original in self.models.items():
            for key, value in original.state_dict().items():
                actual = loaded[name].state_dict()[key]
                dtype = torch.float16
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(actual, value.to(dtype), rtol=0, atol=0)
        self.assertIsNotNone(pipe.scheduler)
        self.assertIsNotNone(pipe.tokenizer_2)
        self.assertEqual(pipe.progressive_load_metrics['uninitialized'], 0)
        self.assertEqual(before, hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_only_one_original_tensor_alive_and_owned_destination(self):
        opened = safe_open
        source_refs = []

        class TrackedFile:
            def __init__(self, *args, **kwargs):
                self.handle = opened(*args, **kwargs)

            def __enter__(self):
                self.handle.__enter__()
                return self

            def __exit__(self, *args):
                return self.handle.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self.handle, name)

            def get_tensor(inner, key):
                self.assertTrue(all(ref() is None for ref in source_refs))
                tensor = inner.handle.get_tensor(key)
                source_refs.append(weakref.ref(tensor))
                return tensor

        with patch('core.sdxl_progressive_loader.safe_open', TrackedFile):
            pipe = load_progressive(self.path, self.root)
        self.assertTrue(all(ref() is None for ref in source_refs))
        self.assertEqual(len(source_refs), sum(len(m.state_dict()) for m in self.models.values()))
        # Retaining the final pipeline did not retain any source mappings/tensors.
        self.assertIsNotNone(pipe.unet)

    def test_engine_integration_uses_progressive_and_cleans_up(self):
        from core.sdxl_pipeline import SDXLPipeline
        (self.root / 'model_index.json').write_text(json.dumps({'_class_name': 'StableDiffusionXLPipeline'}))
        with patch.dict(os.environ, {'BUTTERVISION_SDXL_CONFIG': str(self.root)}), \
             patch('core.sdxl_pipeline.available_ram', return_value=100 * 1024**3), \
             patch('diffusers.StableDiffusionXLPipeline.from_single_file', side_effect=AssertionError('legacy')):
            backend = SDXLPipeline(str(self.path))
            pipe = backend.load_pipeline()
        self.assertTrue(backend.load_plan['progressive_load'])
        self.assertEqual(backend.load_state, 'loaded')
        assert_initialized({name: getattr(pipe, name) for name in self.models})
        ref = weakref.ref(pipe)
        del pipe
        backend.cleanup()
        gc.collect()
        self.assertIsNone(ref())
        self.assertIsNone(backend.pipe)

    def test_symbolic_conversion_matches_official_numerically(self):
        models = empty_models(self.root)
        plans = conversion_plan(read_index(self.path), models)
        references = {
            'unet': official.convert_ldm_unet_checkpoint(self.weights, models['unet'].config),
            'vae': official.convert_ldm_vae_checkpoint(self.weights, models['vae'].config),
            'text_encoder': official.convert_ldm_clip_checkpoint(self.weights, 'conditioner.embedders.0.transformer.'),
            'text_encoder_2': official.convert_open_clip_checkpoint(models['text_encoder_2'], self.weights, 'conditioner.embedders.1.model.'),
        }
        for name, plan in plans.items():
            for key, ref in plan.items():
                torch.testing.assert_close(ref.apply(self.weights[ref.key]), references[name][key], rtol=0, atol=0)
        self.assertTrue(any(ref.operations for ref in plans['text_encoder_2'].values()))
        self.assertTrue(any(ref.operations for ref in plans['vae'].values()))

    def test_tensor_index_does_not_read_tensor_values(self):
        index = read_index(self.path)
        self.assertTrue(all(isinstance(x, TensorRef) for x in index.values()))
        with safe_open(self.path, framework='pt') as handle:
            key = next(iter(index))
            torch.testing.assert_close(handle.get_tensor(key), self.weights[key])

    def test_missing_each_component_rejected_before_materialization(self):
        for prefix in ('model.diffusion_model.', 'first_stage_model.',
                       'conditioner.embedders.0.', 'conditioner.embedders.1.'):
            weights = dict(self.weights)
            weights.pop(next(k for k in weights if k.startswith(prefix)))
            save_file(weights, self.path)
            with self.assertRaises(ProgressiveLoadError):
                conversion_plan(read_index(self.path), empty_models(self.root))

    def test_shape_and_dtype_mismatch(self):
        key = 'model.diffusion_model.input_blocks.0.0.weight'
        for value in (torch.ones(1), torch.ones(self.weights[key].shape, dtype=torch.int64)):
            weights = dict(self.weights); weights[key] = value
            save_file(weights, self.path)
            with self.assertRaises(ProgressiveLoadError):
                conversion_plan(read_index(self.path), empty_models(self.root))

    def test_meta_parameters_and_buffers(self):
        for buffer in (False, True):
            model = torch.nn.Module()
            if buffer:
                model.register_buffer('pending', torch.empty(1, device='meta'))
            else:
                model.register_parameter('pending', torch.nn.Parameter(torch.empty(1, device='meta')))
            with self.assertRaises(ProgressiveLoadError):
                assert_initialized({'model': model})

    def test_cancellation_and_oom_release_partial_models(self):
        for fail in ('cancel', 'oom'):
            models = empty_models(self.root)
            refs = [weakref.ref(model) for model in models.values()]
            with patch('core.sdxl_progressive_loader.empty_models', side_effect=lambda _: models):
                if fail == 'cancel':
                    checks = []
                    def cancelled():
                        checks.append(True)
                        return len(checks) > 2
                    with self.assertRaises(InterruptedError):
                        load_progressive(self.path, self.root, cancelled=cancelled)
                    self.assertEqual(len(checks), 3)
                else:
                    from accelerate.utils import set_module_tensor_to_device
                    calls = []
                    def fail_after_first(*args, **kwargs):
                        if calls:
                            raise MemoryError('fixture OOM after first tensor')
                        calls.append(True)
                        return set_module_tensor_to_device(*args, **kwargs)
                    with patch('accelerate.utils.set_module_tensor_to_device', side_effect=fail_after_first):
                        with self.assertRaises(MemoryError):
                            load_progressive(self.path, self.root)
                    self.assertEqual(len(calls), 1)
            models.clear(); gc.collect()
            self.assertTrue(all(ref() is None for ref in refs))

    def test_buffer_limit_corruption_and_unsupported(self):
        with self.assertRaisesRegex(ProgressiveLoadError, 'Buffer'):
            load_progressive(self.path, self.root, max_buffer_bytes=1)
        bad = self.root / 'bad.ckpt'; bad.write_bytes(b'bad')
        with self.assertRaises(ProgressiveLoadError):
            validate_source(bad)
        self.path.write_bytes(b'corrupt')
        with self.assertRaises(Exception):
            read_index(self.path)

    def test_progressive_planner_preserves_inference_requirements(self):
        from core.sdxl_memory import plan_strategy, GIB
        resources = {'ram_available_bytes': 9 * GIB, 'vram_available_bytes': 3 * GIB,
                     'cuda_available': True, 'accelerate_available': True,
                     'safetensors_mmap_available': True, 'low_cpu_mem_usage_available': True,
                     'sequential_cpu_offload_available': True, 'cpu_offload_configured': True}
        old = plan_strategy(7 * GIB, resources)
        new = plan_strategy(7 * GIB, resources, progressive={'resident_bytes': 6 * GIB, 'temporary_bytes': GIB // 4})
        self.assertTrue(new['admitted'])
        self.assertEqual(old['vram_low_required_bytes'], new['vram_low_required_bytes'])
        resources['ram_available_bytes'] = GIB
        self.assertFalse(plan_strategy(7 * GIB, resources, progressive={'resident_bytes': 6 * GIB, 'temporary_bytes': GIB // 4})['admitted'])

    def test_load_only_scope(self):
        from scripts.sdxl_isolated_probe import build_scope_command
        from types import SimpleNamespace
        with patch('scripts.sdxl_isolated_probe.psutil.virtual_memory', return_value=SimpleNamespace(total=16 * 1024**3)):
            command = build_scope_command(self.path, load_only=True)
        self.assertIn('--load-only', command)
        self.assertIn('--scope', command)
        self.assertIn('--collect', command)
        self.assertIn('--property=MemoryMax=10G', command)
        self.assertIn('--property=MemoryHigh=9G', command)
        self.assertIn('--property=MemorySwapMax=0', command)
        self.assertIn('timeout', command)
        self.assertIn('3600s', command)
        self.assertNotIn('--wait', command)


if __name__ == '__main__':
    unittest.main()
