"""Experimental routing tests; no model weights or inference are run."""
import time
from unittest.mock import patch
from PIL import Image
import config
from api.generation import GenerationJob
from core.model_catalog import SDXL
from test_model_manager_v2_3 import OfflineFixture


class ExperimentalImageTests(OfflineFixture):
    def setUp(self):
        super().setUp()
        self.engine = self.service()
        self.engine.family = SDXL
        self.engine.sd_manager.model_id = str(self.xl)
        self.payload = dict(engine_mode='pizarra_experimental', prompt='test', negative_prompt='',
                            steps=1, cfg_scale=1.0, width=64, height=64, seed=507, batch_size=1)

    def test_restrictions(self):
        cases = [({'width': 512}, '64'), ({'steps': 2}, 'paso'),
                 ({'negative_prompt': 'x'}, 'negativo'), ({'reference_image': 'x'}, 'referencias'),
                 ({'device': 'cuda'}, 'adicionales')]
        for changes, message in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                self.engine.create_job('txt2img', {**self.payload, **changes})
        with self.assertRaisesRegex(ValueError, 'txt2img'):
            self.engine.create_job('face_reference', self.payload)
        self.engine.lora_registry['x'] = {}
        with self.assertRaisesRegex(ValueError, 'LoRA'):
            self.engine.create_job('txt2img', self.payload)
        self.engine.lora_registry.clear()
        self.engine.family = 'sd15'
        with self.assertRaisesRegex(ValueError, 'SDXL'):
            self.engine.create_job('txt2img', self.payload)

    def test_route_and_normal_history(self):
        with patch.object(config, 'OUTPUTS_DIR', self.root / 'outputs'), \
             patch('core.pizarra.sdxl_image_service.run_image', return_value=Image.new('RGB', (64, 64))) as run:
            job = self.engine.create_job('txt2img', self.payload)
            for _ in range(500):
                if job.status in ('completed', 'failed'):
                    break
                time.sleep(.01)
            self.assertEqual(job.status, 'completed', job.error)
            self.assertEqual(run.call_args.args[1:], ('test', 64, 507))
            self.assertEqual(self.engine.history(), job.result_images)

    def test_error_reaches_job(self):
        with patch('core.pizarra.sdxl_image_service.run_image', side_effect=RuntimeError('volumen no disponible')):
            job = self.engine.create_job('txt2img', self.payload)
            for _ in range(500):
                if job.status == 'failed':
                    break
                time.sleep(.01)
            self.assertEqual(job.status, 'failed')
            self.assertIn('volumen no disponible', job.error)
