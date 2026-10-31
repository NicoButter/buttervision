"""Local model manager regression tests. No checkpoint weights or inference are loaded."""
import json
import gc
import os
import tempfile
import threading
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

import api.app as api_app
import config
from api.generation import EngineBusyError, GenerationJob, GenerationService
from core.model_catalog import SD15, SDXL, UNKNOWN, approved_path, checkpoint_family, entries, lora_family
from core.sdxl_pipeline import SDXLPipeline, check_budget, local_sdxl_config
from scripts.sdxl_isolated_probe import build_scope_command


def checkpoint(path, family):
    dim = 2048 if family == SDXL else 768
    weights = {
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": torch.zeros((2, dim)),
        "first_stage_model.encoder.conv_in.weight": torch.zeros(1),
        ("conditioner.embedders.0.transformer.weight" if family == SDXL else "cond_stage_model.transformer.weight"): torch.zeros(1),
    }
    if family == SDXL:
        weights["conditioner.embedders.1.model.weight"] = torch.zeros(1)
    save_file(weights, str(path))


def lora(path, family, metadata=None):
    dim = 2048 if family == SDXL else 768
    save_file({"lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight": torch.zeros((2, dim))},
              str(path), metadata=metadata)


class OfflineFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = self.root / "checkpoints"
        self.models.mkdir()
        self.loras = self.root / "loras"
        self.loras.mkdir()
        self.sd = self.models / "sd.safetensors"
        self.xl = self.models / "xl.safetensors"
        checkpoint(self.sd, SD15)
        checkpoint(self.xl, SDXL)
        self.one = self.loras / "one.safetensors"
        self.two = self.loras / "two.safetensors"
        self.other = self.loras / "other.safetensors"
        lora(self.one, SD15)
        lora(self.two, SD15)
        lora(self.other, SDXL)
        for target, value in (("SD_MODELS_DIR", self.models), ("LORA_DIR", self.loras)):
            p = patch.object(config, target, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(config.model_config, "model_id", str(self.sd))
        p.start()
        self.addCleanup(p.stop)
        for target in ("requests.sessions.Session.request", "core.model_manager.snapshot_download"):
            p = patch(target, side_effect=AssertionError("network/download forbidden"))
            p.start()
            self.addCleanup(p.stop)
        p = patch.dict(os.environ, {"BUTTERVISION_CHECKPOINT_DIRS": "", "BUTTERVISION_LORA_DIRS": ""})
        p.start()
        self.addCleanup(p.stop)

    def service(self):
        return GenerationService()


class CatalogTests(OfflineFixture):
    def test_architecture_uses_structure_not_filename(self):
        self.assertEqual(checkpoint_family(self.sd), SD15)
        self.assertEqual(checkpoint_family(self.xl), SDXL)
        self.assertEqual(lora_family(self.one), SD15)
        self.assertEqual(lora_family(self.other), SDXL)
        renamed = self.root / "definitely_sdxl.safetensors"
        checkpoint(renamed, SD15)
        self.assertEqual(checkpoint_family(renamed), SD15)
        invalid = self.root / "bad.safetensors"
        invalid.write_text("not a safetensors file")
        self.assertEqual(checkpoint_family(invalid), UNKNOWN)
        self.assertEqual(lora_family(invalid), UNKNOWN)

    def test_conflicting_lora_evidence_is_rejected(self):
        bad = self.loras / "conflict.safetensors"
        lora(bad, SD15, {"ss_base_model_version": "sdxl_base_v1-0"})
        self.assertEqual(lora_family(bad), UNKNOWN)

    def test_diffusers_configs_require_both_encoders_for_sdxl(self):
        model = self.root / "diffusers"
        (model / "unet").mkdir(parents=True)
        (model / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionXLPipeline"}))
        (model / "unet/config.json").write_text(json.dumps({"cross_attention_dim": 2048}))
        self.assertEqual(checkpoint_family(model), UNKNOWN)
        for filename in ("text_encoder_2/config.json", "tokenizer_2/tokenizer_config.json"):
            file = model / filename
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("{}")
        self.assertEqual(checkpoint_family(model), SDXL)

    def test_registered_forge_directory_and_symlink_escape(self):
        forge = self.root / "forge"
        forge.mkdir()
        existing = forge / "not-copied.safetensors"
        checkpoint(existing, SDXL)
        outside = self.root / "outside.safetensors"
        checkpoint(outside, SD15)
        (forge / "escape.safetensors").symlink_to(outside)
        self.assertIsNone(approved_path(str(existing), "checkpoint"))
        with patch.dict(os.environ, {"BUTTERVISION_CHECKPOINT_DIRS": str(forge)}):
            self.assertIn(existing, entries("checkpoint"))
            self.assertIsNone(approved_path(str(forge / "escape.safetensors"), "checkpoint"))
            self.assertEqual(approved_path(str(existing), "checkpoint"), existing)
            self.assertFalse((self.models / existing.name).exists())

    def test_default_checkpoint_real_header(self):
        path = Path(__file__).resolve().parents[1] / "models/stable-diffusion/cyberrealistic_final.safetensors"
        if path.exists():
            self.assertEqual(checkpoint_family(path), SD15)


class ManagerTests(OfflineFixture):
    def test_selection_state_is_not_reported_as_loaded_or_ready(self):
        service = self.service()
        self.assertEqual(service.model_state, "selected")
        self.assertEqual(service.engine_state()["model_state"], "selected")
        with patch("api.generation.local_sdxl_config", return_value="/local/config"):
            service.set_model(str(self.xl))
        self.assertEqual(service.model_state, "selected")
        self.assertIsNone(service.sd_manager.pipe)
        active = next(model for model in service.list_models() if model["active"])
        self.assertEqual(active["state"], "selected")
        self.assertEqual(next(model for model in service.list_models() if model["value"] == str(self.sd))["state"], "detected")

    def test_txt2img_publishes_loading_generating_and_loaded_states(self):
        service = self.service()
        states = []
        service.sd_manager.load_sd_pipeline = Mock(side_effect=lambda: states.append(service.model_state))
        service.sd_manager.generate_image = Mock(side_effect=lambda **kwargs: states.append(service.model_state) or [])
        service._save_generation = Mock(return_value=([], config.OUTPUTS_DIR / "mock"))
        service._format_info = Mock(return_value="mock")
        job = GenerationJob(id="test", mode="txt2img", payload={"prompt": "controlled"})
        service._run_txt2img(job)
        self.assertEqual(states, ["loading", "generating"])
        self.assertEqual(service.model_state, "loaded")

    def test_startup_uses_sdxl_backend_when_active_model_is_sdxl(self):
        with patch.object(config.model_config, "model_id", str(self.xl)):
            service = self.service()
        self.assertEqual(service.family, SDXL)
        self.assertIsInstance(service.sd_manager, SDXLPipeline)
        self.assertIsNone(service.sd_manager.pipe)

    def test_unknown_or_unregistered_checkpoint_preserves_previous_state(self):
        service = self.service()
        service.sd_manager.cleanup = Mock()
        invalid = self.models / "not-a-model.safetensors"
        invalid.write_bytes(b"bad")
        for target in (invalid, self.root / "unregistered.safetensors"):
            with self.assertRaises(ValueError):
                service.set_model(str(target))
        self.assertEqual(service.family, SD15)
        self.assertEqual(service.sd_manager.model_id, str(self.sd))
        service.sd_manager.cleanup.assert_not_called()

    def test_architecture_roundtrip_releases_old_manager(self):
        service = self.service()
        old = service.sd_manager
        old.cleanup = Mock()
        xl = Mock(model_id=str(self.xl), pipe=None)
        new_sd = Mock(model_id=str(self.sd), sd_pipeline=None)
        with patch("api.generation.local_sdxl_config", return_value="/local/config"), patch("api.generation.SDXLPipeline", return_value=xl), patch("api.generation.ButterVisionPipeline", return_value=new_sd):
            service.set_model(str(self.xl))
            old.cleanup.assert_called_once()
            self.assertIs(service.sd_manager, xl)
            self.assertEqual(service.family, SDXL)
            service.set_model(str(self.sd))
        xl.cleanup.assert_called_once()
        self.assertIs(service.sd_manager, new_sd)
        self.assertEqual(service.family, SD15)

    def test_missing_sdxl_configuration_does_not_release_previous_model(self):
        service = self.service()
        service.sd_manager.cleanup = Mock()
        with patch("api.generation.local_sdxl_config", side_effect=ValueError("missing")):
            with self.assertRaises(ValueError):
                service.set_model(str(self.xl))
        service.sd_manager.cleanup.assert_not_called()
        self.assertEqual(service.family, SD15)

    def test_sdxl_face_blocked_and_memory_failure_precedes_loading(self):
        service = self.service()
        service.family = SDXL
        service.sd_manager = Mock(model_id=str(self.xl), pipe=None)
        with self.assertRaisesRegex(ValueError, "Face Reference"):
            service.create_job("face_reference", {"prompt": "x"})
        with patch("api.generation.check_budget", side_effect=ValueError("RAM insuficiente")):
            with self.assertRaisesRegex(ValueError, "RAM"):
                service.create_job("txt2img", {"prompt": "x", "width": 768, "height": 768})
        service.sd_manager.load_pipeline.assert_not_called()
        self.assertFalse(service.worker_lock.locked())
        with patch.object(api_app, "get_service", return_value=service), patch.object(api_app, "validate_reference_image"):
            response = TestClient(api_app.app).post("/api/generate/face-reference", json={"prompt": "x", "reference_image": "test"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("SDXL", response.json()["detail"])

    def test_sdxl_resolution_is_not_clamped_to_sd15(self):
        service = self.service()
        service.family = SDXL
        with patch.object(service, "resolutions", return_value=[512, 640, 768]):
            result = service._normalized_common({"prompt": "x", "width": 640, "height": 768})
        self.assertEqual((result["width"], result["height"]), (640, 768))

    def test_multiple_loras_and_weight_updates_do_not_replace_each_other(self):
        service = self.service()
        pipe = Mock()
        active = []
        def set_active(names, adapter_weights):
            active[:] = names
        pipe.set_adapters.side_effect = set_active
        pipe.get_active_adapters.side_effect = lambda: list(active)
        pipe.delete_adapters.side_effect = lambda name: active.remove(name)
        service.sd_manager.sd_pipeline = pipe
        service.sd_manager.load_sd_pipeline = Mock(return_value=pipe)
        for path, weight in ((self.one, 0.4), (self.two, 0.7)):
            service.set_lora(str(path), True, weight)
        self.assertEqual(len(active), 2)
        self.assertEqual(pipe.set_adapters.call_args.kwargs["adapter_weights"], [0.4, 0.7])
        self.assertEqual(sum(item["active"] for item in service.list_loras()), 2)
        service.set_lora(str(self.one), True, 1.2)
        self.assertEqual(len(active), 2)
        self.assertEqual(pipe.set_adapters.call_args.kwargs["adapter_weights"], [1.2, 0.7])
        self.assertEqual(pipe.load_lora_weights.call_count, 2)
        service.set_lora(str(self.one), False)
        self.assertEqual(len(active), 1)
        self.assertEqual(sum(item["active"] for item in service.list_loras()), 1)
        self.assertTrue(all(call.kwargs["local_files_only"] for call in pipe.load_lora_weights.call_args_list))

    def test_adjusted_legacy_lora_keeps_its_weight_when_another_is_activated(self):
        service = self.service()
        pipe = Mock()
        active = ["mi_cara"]
        pipe.set_adapters.side_effect = lambda names, adapter_weights: active.__setitem__(slice(None), names)
        pipe.get_active_adapters.side_effect = lambda: list(active)
        service.sd_manager.sd_pipeline = pipe
        service.sd_manager.load_sd_pipeline = Mock(return_value=pipe)
        service.sd_manager.loaded_loras = {"mi_cara": 0.8}
        service.sd_manager.available_loras = {"mi_cara": str(self.one)}
        service.set_lora(str(self.one), True, 1.2)
        service.set_lora(str(self.two), True, 0.5)
        self.assertEqual(pipe.set_adapters.call_args.kwargs["adapter_weights"], [1.2, 0.5])

    def test_legacy_lora_is_visible_while_generation_is_running(self):
        service = self.service()
        pipe = Mock()
        entered, finish = threading.Event(), threading.Event()
        def load():
            service.sd_manager.sd_pipeline = pipe
            service.sd_manager.loaded_loras = {"mi_cara": 0.8}
            service.sd_manager.available_loras = {"mi_cara": str(self.one)}
            return pipe
        def generate(**kwargs):
            entered.set()
            finish.wait(3)
            return []
        service.sd_manager.load_sd_pipeline = load
        service.sd_manager.generate_image = generate
        service._save_generation = Mock(return_value=([], config.OUTPUTS_DIR / "mock"))
        job = service.create_job("txt2img", {"prompt": "x"})
        try:
            self.assertTrue(entered.wait(3))
            item = next(item for item in service.list_loras() if item["value"] == str(self.one))
            self.assertTrue(item["active"])
        finally:
            finish.set()
            self.assertTrue(service.worker_lock.acquire(timeout=3))
            service.worker_lock.release()
        self.assertEqual(job.status, "completed")

    def test_incompatible_lora_and_busy_mutations_do_not_load(self):
        service = self.service()
        service.sd_manager.load_sd_pipeline = Mock()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            service.set_lora(str(self.other), True)
        service.sd_manager.load_sd_pipeline.assert_not_called()
        service.worker_lock.acquire()
        try:
            for operation in (lambda: service.set_lora(str(self.one), True), lambda: service.set_model(str(self.xl)), lambda: service.create_job("txt2img", {})):
                with self.assertRaises(EngineBusyError):
                    operation()
            with patch.object(api_app, "get_service", return_value=service):
                response = TestClient(api_app.app).post("/api/loras/active", json={"path": str(self.one), "enabled": True})
            self.assertEqual(response.status_code, 409)
        finally:
            service.worker_lock.release()

    def test_lora_failure_cleans_partial_pipeline_and_allows_retry(self):
        service = self.service()
        pipe = Mock()
        pipe.load_lora_weights.side_effect = RuntimeError("load")
        service.sd_manager.sd_pipeline = pipe
        service.sd_manager.load_sd_pipeline = Mock(return_value=pipe)
        service.sd_manager.cleanup = Mock()
        with self.assertRaises(RuntimeError):
            service.set_lora(str(self.one), True)
        service.sd_manager.cleanup.assert_called_once()
        self.assertEqual(service.lora_registry, {})
        self.assertFalse(service.worker_lock.locked())

    def test_lora_failure_releases_local_pipeline_reference_before_cleanup(self):
        service = self.service()
        class FailingPipe:
            def load_lora_weights(self, *args, **kwargs):
                raise RuntimeError("load")
        service.sd_manager.sd_pipeline = FailingPipe()
        reference = weakref.ref(service.sd_manager.sd_pipeline)
        service.sd_manager.load_sd_pipeline = lambda: service.sd_manager.sd_pipeline
        observed = []
        def cleanup():
            service.sd_manager.sd_pipeline = None
            gc.collect()
            observed.append(reference() is None)
        service.sd_manager.cleanup = cleanup
        with self.assertRaises(RuntimeError):
            service.set_lora(str(self.one), True)
        self.assertEqual(observed, [True])

    def test_loaded_lora_remains_visible_and_can_be_disabled_if_file_disappears(self):
        service = self.service()
        pipe = Mock()
        active = []
        pipe.set_adapters.side_effect = lambda names, adapter_weights: active.__setitem__(slice(None), names)
        pipe.get_active_adapters.side_effect = lambda: list(active)
        pipe.delete_adapters.side_effect = lambda name: active.remove(name)
        service.sd_manager.sd_pipeline = pipe
        service.sd_manager.load_sd_pipeline = Mock(return_value=pipe)
        service.set_lora(str(self.one), True)
        self.one.unlink()
        missing = next(item for item in service.list_loras() if item["value"] == str(self.one))
        self.assertTrue(missing["active"])
        self.assertFalse(missing["compatible"])
        service.set_lora(str(self.one), False)
        self.assertEqual(active, [])


class BackendTests(OfflineFixture):
    def test_budget_blocks_before_factor_load(self):
        with patch("core.sdxl_pipeline.available_ram", return_value=0), patch("core.sdxl_pipeline.StableDiffusionXLPipeline.from_single_file") as load:
            with self.assertRaisesRegex(ValueError, "RAM"):
                SDXLPipeline(str(self.xl)).load_pipeline()
        load.assert_not_called()

    def test_disabled_offload_requires_room_for_weights_on_gpu(self):
        with patch("core.sdxl_pipeline.available_ram", return_value=100 * 1024**3), patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.mem_get_info", return_value=(1024**3, 4 * 1024**3)), patch.object(config.model_config, "enable_cpu_offload", False):
            with self.assertRaisesRegex(ValueError, "VRAM"):
                check_budget(str(self.xl))

    def test_low_cpu_memory_budget_is_checkpoint_plus_explicit_margin(self):
        from core.sdxl_pipeline import inspect_load_budget
        with patch("core.sdxl_pipeline.available_ram", return_value=16 * 1024**3), patch("core.sdxl_pipeline.is_accelerate_available", return_value=True), patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.mem_get_info", return_value=(8 * 1024**3, 8 * 1024**3)):
            budget = inspect_load_budget(self.xl)
        self.assertTrue(budget["admitted"])
        self.assertEqual(budget["estimated_required_ram_bytes"], self.xl.stat().st_size + 2 * 1024**3)
        self.assertTrue(budget["low_cpu_mem_usage_available"])
        self.assertEqual(budget["ram_weight_multiplier"], 1)
        self.assertEqual(budget["strategy"], "diffusers_low_cpu_mem_usage_sequential_cpu_offload")

    def test_cpu_fallback_accounts_for_float32_weight_expansion(self):
        from core.sdxl_pipeline import inspect_load_budget
        with patch("core.sdxl_pipeline.available_ram", return_value=16 * 1024**3), patch("core.sdxl_pipeline.is_accelerate_available", return_value=True), patch("torch.cuda.is_available", return_value=False):
            budget = inspect_load_budget(self.xl)
        self.assertEqual(budget["ram_weight_multiplier"], 2)
        self.assertEqual(budget["estimated_required_ram_bytes"], 2 * self.xl.stat().st_size + 2 * 1024**3)

    def test_low_cpu_memory_budget_requires_accelerate(self):
        with patch("core.sdxl_pipeline.available_ram", return_value=16 * 1024**3), patch("core.sdxl_pipeline.is_accelerate_available", return_value=False), patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "Accelerate"):
                check_budget(self.xl)

    def test_probe_scope_is_bounded_and_rejects_unregistered_checkpoint(self):
        with patch("scripts.sdxl_isolated_probe.psutil.virtual_memory", return_value=SimpleNamespace(total=16 * 1024**3)):
            command = build_scope_command(str(self.xl), memory_max_gib=10, timeout_seconds=300)
        self.assertIn("--property=MemoryHigh=9G", command)
        self.assertIn("--property=MemoryMax=10G", command)
        self.assertIn("--property=MemorySwapMax=0", command)
        self.assertIn("300s", command)
        self.assertIn("--worker", command)
        with self.assertRaisesRegex(ValueError, "catálogo local"):
            build_scope_command(str(self.root / "outside.safetensors"), memory_max_gib=10)

    def test_probe_scope_leaves_host_headroom(self):
        with patch("scripts.sdxl_isolated_probe.psutil.virtual_memory", return_value=SimpleNamespace(total=16 * 1024**3)):
            with self.assertRaisesRegex(ValueError, "dejar 4 GiB"):
                build_scope_command(str(self.xl), memory_max_gib=13)

    def test_sdxl_offload_precision_local_config_and_cleanup(self):
        pipe = Mock()
        pipe.components = {}
        with patch("torch.cuda.is_available", return_value=True), patch("core.sdxl_pipeline.check_budget"), patch("core.sdxl_pipeline.local_sdxl_config", return_value="/local/sdxl"), patch("core.sdxl_pipeline.StableDiffusionXLPipeline.from_single_file", return_value=pipe) as load:
            backend = SDXLPipeline(str(self.xl))
            self.assertIs(backend.load_pipeline(), pipe)
            self.assertIs(backend.load_pipeline(), pipe)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(load.call_args.kwargs["config"], "/local/sdxl")
        self.assertTrue(load.call_args.kwargs["local_files_only"])
        self.assertTrue(load.call_args.kwargs["low_cpu_mem_usage"])
        self.assertIs(load.call_args.kwargs["torch_dtype"], torch.float16)
        pipe.enable_sequential_cpu_offload.assert_called_once()
        pipe.enable_vae_tiling.assert_called_once()
        with patch("core.engine_memory.torch.cuda.is_available", return_value=False):
            backend.cleanup()
        self.assertIsNone(backend.pipe)

    def test_load_failure_releases_partial_pipeline_and_can_retry(self):
        broken, working = Mock(), Mock()
        broken.components = {}
        broken.enable_attention_slicing.side_effect = RuntimeError("load failed")
        working.components = {}
        factory = Mock(side_effect=[broken, working])
        backend = SDXLPipeline(str(self.xl))
        with patch("core.sdxl_pipeline.check_budget"), patch("core.sdxl_pipeline.local_sdxl_config", return_value="/local/sdxl"), patch("core.sdxl_pipeline.StableDiffusionXLPipeline.from_single_file", factory), patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                backend.load_pipeline()
            self.assertIsNone(backend.pipe)
            self.assertEqual(backend.load_state, "error")
            self.assertIs(backend.load_pipeline(), working)
        self.assertEqual(backend.load_state, "loaded")
        backend.cleanup()

    def test_two_tokenizers_are_used_and_passed_independently(self):
        class Tokenizer:
            model_max_length = 5
            def __call__(self, text, add_special_tokens=True, **kwargs):
                words = text.split()
                return SimpleNamespace(input_ids=(["bos"] + words + ["eos"]) if add_special_tokens else words)
            def num_special_tokens_to_add(self, pair=False):
                return 2
            def decode(self, values, **kwargs):
                return " ".join(values)
        pipe = Mock(return_value=SimpleNamespace(images=[]))
        pipe.tokenizer = Tokenizer()
        pipe.tokenizer_2 = Tokenizer()
        pipe.text_encoder = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=5))
        pipe.text_encoder_2 = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4))
        backend = SDXLPipeline(str(self.xl))
        backend.pipe = pipe
        warnings = []
        self.assertEqual(backend.generate_image("a b c d e", prompt_warning_callback=warnings.append, width=768, height=768), [])
        self.assertEqual(pipe.call_args.kwargs["prompt"], "a b c")
        self.assertEqual(pipe.call_args.kwargs["prompt_2"], "a b")
        self.assertEqual(len(warnings), 2)

    def test_sdxl_configuration_does_not_accept_sd15(self):
        with patch.dict(os.environ, {"BUTTERVISION_SDXL_CONFIG": str(self.root / "missing")}), patch("core.sdxl_pipeline.try_to_load_from_cache", return_value=None):
            with self.assertRaisesRegex(ValueError, "SDXL"):
                local_sdxl_config(self.xl)


if __name__ == "__main__":
    unittest.main()
