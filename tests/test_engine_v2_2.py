"""Engine regression suite: all model loading/inference and network are mocked."""
import threading
import tempfile
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from fastapi.testclient import TestClient

import api.app as api_app
import config
from api.generation import EngineBusyError, GenerationJob, GenerationService
from core.advanced_pipeline import ButterVisionPipeline
from core.face_reference_pipeline import SD15FaceReferencePipeline
from core.engine_memory import diagnose, release_owned
from core.prompt_limits import limit_prompts
from core.local_assets import local_sd15_config


class FakeTokenizer:
    model_max_length = 7

    def __call__(self, text, add_special_tokens=True, **kwargs):
        tokens = text.split()
        return SimpleNamespace(input_ids=(["bos"] + tokens + ["eos"]) if add_special_tokens else tokens)

    def decode(self, tokens, **kwargs):
        return " ".join(tokens)

    def num_special_tokens_to_add(self, **kwargs):
        return 2


def mock_pipe():
    pipe = Mock(return_value=SimpleNamespace(images=[]))
    pipe.tokenizer = FakeTokenizer()
    pipe.text_encoder = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=7))
    pipe.scheduler = SimpleNamespace(config={})
    return pipe


class OfflineTests(unittest.TestCase):
    def setUp(self):
        # Fail closed if any test accidentally reaches network/model downloads.
        self.network = patch("requests.sessions.Session.request", side_effect=AssertionError("network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.download = patch("core.model_manager.snapshot_download", side_effect=AssertionError("download forbidden"))
        self.download.start()
        self.addCleanup(self.download.stop)
        self.model_config = patch.object(config.model_config, "model_id", "model.safetensors")
        self.model_config.start()
        self.addCleanup(self.model_config.stop)


class SingletonTests(OfflineTests):
    def test_concurrent_initialization_constructs_one_service(self):
        entered, release = threading.Event(), threading.Event()
        instance = object()

        def construct():
            entered.set()
            self.assertTrue(release.wait(3))
            return instance

        with patch.object(api_app, "service", None), patch("api.generation.GenerationService", side_effect=construct) as factory:
            with ThreadPoolExecutor(max_workers=12) as pool:
                futures = [pool.submit(api_app.get_service) for _ in range(12)]
                self.assertTrue(entered.wait(3))
                release.set()
                self.assertTrue(all(future.result(3) is instance for future in futures))
            self.assertEqual(factory.call_count, 1)

    def test_failed_initialization_can_retry(self):
        instance = object()
        with patch.object(api_app, "service", None), patch("api.generation.GenerationService", side_effect=[RuntimeError("init"), instance]):
            with self.assertRaises(RuntimeError):
                api_app.get_service()
            self.assertIsNone(api_app.service)
            self.assertIs(api_app.get_service(), instance)


class ServiceTests(OfflineTests):
    def setUp(self):
        super().setUp()
        self.sd = Mock(model_id="model.safetensors")
        self.models = Mock()
        self.models.resolve_model_path.side_effect = lambda value: value
        self.models.list_local_model_infos.return_value = []
        with patch("api.generation.ButterVisionPipeline", return_value=self.sd), patch("api.generation.ModelManager", return_value=self.models):
            self.engine = GenerationService()
        self.saves = patch.object(self.engine, "_save_generation", return_value=([], config.OUTPUTS_DIR / "mock"))
        self.saves.start()
        self.addCleanup(self.saves.stop)
        self.finished = threading.Event()
        self.addCleanup(self.finished.set)

    def wait_idle(self):
        self.assertTrue(self.engine.worker_lock.acquire(timeout=3))
        self.engine.worker_lock.release()

    def run_job(self, mode="txt2img", **payload):
        job = self.engine.create_job(mode, {"prompt": "private prompt", **payload})
        self.wait_idle()
        return job

    def block_job(self):
        entered = threading.Event()

        def generate(job):
            entered.set()
            if not self.finished.wait(3):
                raise RuntimeError("test worker timeout")
            job.update(status="completed", progress=1.0, message="Completed")

        blocker = patch.object(self.engine, "_run_txt2img", side_effect=generate)
        blocker.start()
        self.addCleanup(blocker.stop)
        job = self.engine.create_job("txt2img", {"prompt": "private"})
        self.assertTrue(entered.wait(3))
        return job

    def test_concurrent_job_rejected_without_inserting_failed_job(self):
        job = self.block_job()
        with self.assertRaises(EngineBusyError):
            self.engine.create_job("face_reference", {})
        self.assertEqual(list(self.engine.jobs), [job.id])
        self.assertEqual(job.as_dict()["status"], "running")
        self.finished.set()
        self.wait_idle()
        self.assertEqual(job.status, "completed")

    def test_reservation_precedes_thread_start_and_is_released_on_start_failure(self):
        with patch("api.generation.threading.Thread") as thread:
            job = self.engine.create_job("txt2img", {})
            with self.assertRaises(EngineBusyError):
                self.engine.create_job("txt2img", {})
            with self.assertRaises(EngineBusyError):
                self.engine.set_model("other")
            self.engine._run_job(job)  # Execute the captured worker, no model load (empty prompt).
        self.wait_idle()
        self.assertEqual(job.status, "failed")
        with patch("api.generation.threading.Thread.start", side_effect=RuntimeError("start")):
            with self.assertRaises(RuntimeError):
                self.engine.create_job("txt2img", {})
        self.wait_idle()
        self.assertEqual(len(self.engine.jobs), 1)
        with patch("api.generation.GenerationJob", side_effect=MemoryError("allocation")):
            with self.assertRaises(MemoryError):
                self.engine.create_job("txt2img", {})
        self.wait_idle()

    def test_http_409_for_both_generation_modes_and_checkpoint_while_busy(self):
        job = self.block_job()
        client = TestClient(api_app.app)
        with patch.object(api_app, "get_service", return_value=self.engine), patch.object(api_app, "validate_reference_image"):
            for route, payload in (
                ("/api/generate/txt2img", {"prompt": "test"}),
                ("/api/generate/face-reference", {"prompt": "test", "reference_image": "private"}),
                ("/api/models/active", {"model_id": "other.safetensors"}),
            ):
                response = client.post(route, json=payload)
                self.assertEqual(response.status_code, 409, response.text)
                self.assertIn("ocupado", response.json()["detail"])
            self.assertEqual(client.get(f"/api/jobs/{job.id}").json()["status"], "running")
        self.sd.change_model.assert_not_called()
        self.finished.set()
        self.wait_idle()

    def test_checkpoint_switch_holds_same_lock_and_does_not_load_models(self):
        entered = threading.Event()
        def change(model):
            entered.set()
            self.finished.wait(3)
            self.sd.model_id = model
        self.sd.change_model.side_effect = change
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.engine.set_model, "other.safetensors")
            self.assertTrue(entered.wait(3))
            with self.assertRaises(EngineBusyError):
                self.engine.create_job("txt2img", {})
            self.finished.set()
            future.result(3)
        self.sd.generate_image.assert_not_called()
        self.sd.load_sd_pipeline.assert_not_called()
        self.assertEqual(config.model_config.model_id, "other.safetensors")

    def test_same_checkpoint_preserves_loaded_managers(self):
        self.engine.face_manager = Mock()
        self.engine.set_model("model.safetensors")
        self.sd.change_model.assert_not_called()
        self.engine.face_manager.cleanup.assert_not_called()

    def test_checkpoint_change_releases_face_before_changing_normal_model(self):
        events = []
        face = Mock()
        self.engine.face_manager = face
        face.cleanup.side_effect = lambda: events.append("face_released")
        def change(model):
            self.assertIsNone(self.engine.face_manager)
            events.append("checkpoint_changed")
            self.sd.model_id = model
        self.sd.change_model.side_effect = change
        self.engine.set_model("other.safetensors")
        self.assertEqual(events, ["face_released", "checkpoint_changed"])

    def test_job_warnings_survive_status_polling(self):
        def generate(**kwargs):
            kwargs["prompt_warning_callback"]("El prompt se truncará.")
            return []
        self.sd.generate_image.side_effect = generate
        job = self.run_job()
        with patch.object(api_app, "get_service", return_value=self.engine):
            response = TestClient(api_app.app).get(f"/api/jobs/{job.id}")
        self.assertEqual(response.json()["warnings"], ["El prompt se truncará."])
        self.assertEqual(response.json()["status"], "completed")

    def test_missing_checkpoint_preserves_current_model(self):
        self.models.resolve_model_path.return_value = None
        self.models.resolve_model_path.side_effect = None
        with self.assertRaises(ValueError):
            self.engine.set_model("missing")
        self.sd.change_model.assert_not_called()
        self.wait_idle()

    def test_txt2img_to_face_releases_normal_before_face_load_and_reuses_face(self):
        self.assertEqual(self.run_job().status, "completed")
        events = []
        face = Mock(base_model="model.safetensors")
        self.sd.cleanup.side_effect = lambda: events.append("sd_released")
        face.generate.side_effect = lambda **kwargs: events.append("face_generate") or []
        with patch("api.generation.SD15FaceReferencePipeline", return_value=face) as factory, patch("api.generation.decode_reference_image", return_value=object()):
            first = self.run_job("face_reference", reference_image="private photo")
            second = self.run_job("face_reference", reference_image="private photo")
        self.assertEqual(events, ["sd_released", "face_generate", "face_generate"])
        self.assertEqual(factory.call_count, 1)
        face.cleanup.assert_not_called()
        self.assertEqual(first.status, "completed")
        self.assertEqual(second.payload, {})
        self.assertEqual(self.engine.active_pipeline, "face_reference")

    def test_face_to_txt2img_drops_face_manager_before_normal_load(self):
        events = []
        face = Mock(base_model="model.safetensors")
        with patch("api.generation.SD15FaceReferencePipeline", return_value=face), patch("api.generation.decode_reference_image", return_value=object()):
            self.run_job("face_reference", reference_image="photo")
        face.cleanup.side_effect = lambda: events.append("face_released")
        def generate(**kwargs):
            self.assertIsNone(self.engine.face_manager)
            events.append("normal_generate")
            return []
        self.sd.generate_image.side_effect = generate
        self.run_job()
        self.run_job()
        self.assertEqual(events, ["face_released", "normal_generate", "normal_generate"])
        self.assertEqual(self.sd.cleanup.call_count, 1)
        self.assertEqual(self.engine.active_pipeline, "txt2img")

    def test_inference_oom_recovers_and_preserves_terminal_job_without_private_data(self):
        self.sd.generate_image.side_effect = [torch.OutOfMemoryError("private prompt photo"), []]
        with self.assertLogs("uvicorn.error.engine", level="INFO") as logs:
            failed = self.run_job()
            success = self.run_job()
        self.assertEqual(failed.status, "failed")
        self.assertIn("Memoria insuficiente", failed.error)
        self.assertEqual(success.status, "completed")
        self.assertEqual(self.engine.get_job(failed.id).status, "failed")
        self.assertEqual(failed.payload, {})
        self.assertNotIn("private", str(failed.as_dict()))
        self.assertNotIn("private", " ".join(logs.output))
        self.assertIn("memory_allocation_error", " ".join(logs.output))
        self.sd.cleanup.assert_called_once()

    def test_face_failure_disposes_manager_and_can_retry(self):
        broken = Mock(base_model="model.safetensors")
        broken.generate.side_effect = RuntimeError("private")
        healthy = Mock(base_model="model.safetensors")
        with patch("api.generation.SD15FaceReferencePipeline", side_effect=[broken, healthy]), patch("api.generation.decode_reference_image", return_value=object()):
            failed = self.run_job("face_reference", reference_image="photo")
            self.assertIsNone(self.engine.face_manager)
            recovered = self.run_job("face_reference", reference_image="photo")
        broken.cleanup.assert_called_once()
        self.assertEqual(failed.status, "failed")
        self.assertEqual(recovered.status, "completed")

    def test_inference_traceback_does_not_retain_resources_during_cleanup(self):
        class TemporaryResource:
            pass
        references = []
        observed_during_cleanup = []
        def fail(**kwargs):
            transient = TemporaryResource()
            references.append(weakref.ref(transient))
            raise RuntimeError("inference")
        self.sd.generate_image.side_effect = fail
        self.sd.cleanup.side_effect = lambda: observed_during_cleanup.append(references[0]())
        self.assertEqual(self.run_job().status, "failed")
        self.assertEqual(observed_during_cleanup, [None])
        self.sd.cleanup.assert_called_once()

    def test_job_snapshots_are_valid_and_independent(self):
        job = GenerationJob(id="test", mode="txt2img", payload={})
        def update():
            for _ in range(300):
                job.update(status="running", progress=0.5, message="running")
                job.update(status="completed", progress=1, message="completed")
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(update)
            for _ in range(300):
                snapshot = job.as_dict()
                if snapshot["status"] in ("running", "completed"):
                    self.assertEqual(snapshot["status"], snapshot["message"])
                snapshot["warnings"].append("external")
            future.result(3)
        self.assertEqual(job.warnings, [])


class PipelineTests(OfflineTests):
    def setUp(self):
        super().setUp()
        for name in ("core.advanced_pipeline.local_sd15_config", "core.face_reference_pipeline.local_sd15_config"):
            mocked = patch(name, return_value="/mock/config")
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_missing_local_configuration_fails_without_downloading(self):
        with patch("core.local_assets.try_to_load_from_cache", return_value=None) as cache:
            with self.assertRaisesRegex(RuntimeError, "No se descargarán"):
                local_sd15_config("/missing/model.safetensors")
        self.assertTrue(cache.called)

    def test_existing_cached_configuration_is_passed_as_local_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json", "tokenizer/tokenizer_config.json", "unet/config.json", "vae/config.json"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            with patch("core.local_assets.try_to_load_from_cache", return_value=str(root / "model_index.json")):
                self.assertEqual(local_sd15_config("/missing/model.safetensors"), folder)

    def test_diagnostics_report_ram_vram_and_state_without_payload(self):
        with patch("core.engine_memory.psutil.virtual_memory", return_value=SimpleNamespace(used=10, available=20)), patch("core.engine_memory.torch.cuda.is_available", return_value=True), patch("core.engine_memory.torch.cuda.mem_get_info", return_value=(30, 50)), patch("core.engine_memory.torch.cuda.memory_allocated", return_value=12), patch("core.engine_memory.torch.cuda.memory_reserved", return_value=15), self.assertLogs("uvicorn.error.engine", level="INFO") as logs:
            diagnose("job_start", "txt2img", "running")
        output = " ".join(logs.output)
        for expected in ("'ram_used': 10", "'ram_available': 20", "'vram_used': 20", "'vram_available': 30", "txt2img", "running"):
            self.assertIn(expected, output)

    def test_release_failure_still_drops_auxiliary_models(self):
        class BrokenPipe:
            def remove_all_hooks(self):
                raise RuntimeError("detach")
        owner = SimpleNamespace(pipe=BrokenPipe(), image_encoder=torch.nn.Linear(1, 1))
        encoder = weakref.ref(owner.image_encoder)
        with self.assertRaises(RuntimeError):
            release_owned(owner, ("pipe", "image_encoder"))
        self.assertIsNone(owner.pipe)
        self.assertIsNone(owner.image_encoder)
        self.assertIsNone(encoder())

    def test_cleanup_removes_child_offload_hooks_and_auxiliary_references_before_cache(self):
        class Pipe:
            def __init__(self, module):
                self.components = {"unet": module}
            def remove_all_hooks(self):
                pass  # Simulate Diffusers version which misses child-only hooks.
        owner = SimpleNamespace(pipe=None, image_encoder=None)
        module = torch.nn.Sequential(torch.nn.Linear(1, 1))
        child = module[0]
        add_hook_to_module(child, AlignDevicesHook(execution_device="cpu", offload=True))
        owner.pipe = Pipe(module)
        owner.image_encoder = torch.nn.Linear(1, 1)
        references = [weakref.ref(value) for value in (owner.pipe, module, owner.image_encoder)]
        del module
        def cache():
            self.assertTrue(all(ref() is None for ref in references))
            self.assertFalse(hasattr(child, "_hf_hook"))
            self.assertFalse(hasattr(child, "_old_forward"))
        with patch("accelerate.utils.modeling.clear_device_cache"), patch("core.engine_memory.torch.cuda.is_available", return_value=True), patch("core.engine_memory.torch.cuda.empty_cache", side_effect=cache) as empty:
            release_owned(owner, ("pipe", "image_encoder"))
            empty.assert_called_once()
        self.assertIsNone(owner.pipe)
        self.assertIsNone(owner.image_encoder)

    def test_face_cleanup_releases_pipeline_and_clip_encoder(self):
        face = SD15FaceReferencePipeline("local.safetensors", device="cpu")
        face.pipe = SimpleNamespace(components={})
        face.image_encoder = torch.nn.Linear(1, 1)
        encoder = weakref.ref(face.image_encoder)
        face.cleanup()
        face.cleanup()  # Idempotent.
        self.assertIsNone(face.pipe)
        self.assertIsNone(face.image_encoder)
        self.assertIsNone(encoder())

    def test_normal_cleanup_releases_all_shared_pipeline_components(self):
        normal = ButterVisionPipeline("local.safetensors", device="cpu")
        shared = torch.nn.Linear(1, 1)
        normal.sd_pipeline = SimpleNamespace(components={"unet": shared})
        normal.img2img_pipeline = SimpleNamespace(components={"unet": shared})
        normal.animatediff_pipeline = SimpleNamespace(components={})
        reference = weakref.ref(shared)
        del shared
        normal.loaded_loras["local"] = 0.5
        normal.cleanup()
        self.assertIsNone(reference())
        self.assertIsNone(normal.sd_pipeline)
        self.assertIsNone(normal.img2img_pipeline)
        self.assertIsNone(normal.animatediff_pipeline)
        self.assertEqual(normal.loaded_loras, {})

    def test_sd_load_is_local_only_and_reused_with_existing_dtype(self):
        for source in ("local.safetensors", "local-directory"):
            with self.subTest(source=source):
                normal = ButterVisionPipeline(source, device="cpu", enable_optimizations=False)
                normal.available_loras = {}
                pipe = mock_pipe()
                with patch.object(normal, "_resolve_model_source", return_value=source), patch.object(normal, "_prepare_pipeline_device", side_effect=lambda p: p), patch("core.advanced_pipeline.DDIMScheduler.from_config"), patch("core.advanced_pipeline.StableDiffusionPipeline.from_single_file", return_value=pipe) as single, patch("core.advanced_pipeline.StableDiffusionPipeline.from_pretrained", return_value=pipe) as pretrained:
                    self.assertIs(normal.load_sd_pipeline(), pipe)
                    self.assertIs(normal.load_sd_pipeline(), pipe)
                loader = single if source.endswith("safetensors") else pretrained
                loader.assert_called_once()
                self.assertTrue(loader.call_args.kwargs["local_files_only"])
                if source.endswith("safetensors"):
                    self.assertEqual(loader.call_args.kwargs["config"], "/mock/config")
                self.assertIs(loader.call_args.kwargs["torch_dtype"], torch.float32)

    def test_face_load_local_only_reuses_clip_and_ip_adapter(self):
        face = SD15FaceReferencePipeline("local.safetensors", device="cpu")
        pipe = mock_pipe()
        with patch("diffusers.StableDiffusionPipeline.from_single_file", return_value=pipe) as factory, patch("diffusers.DDIMScheduler.from_config"), patch.object(face, "_load_clip_vision", return_value=object()) as clip, patch.object(face, "_load_ip_adapter_state_dict", return_value={}), patch.object(pipe, "to", return_value=pipe):
            self.assertIs(face._load_pipeline(), pipe)
            self.assertIs(face._load_pipeline(), pipe)
        factory.assert_called_once()
        clip.assert_called_once()
        self.assertTrue(factory.call_args.kwargs["local_files_only"])
        self.assertEqual(factory.call_args.kwargs["config"], "/mock/config")
        self.assertTrue(pipe.load_ip_adapter.call_args.kwargs["local_files_only"])
        self.assertIs(factory.call_args.kwargs["dtype"], torch.float32)

    def test_partial_load_remains_owned_for_failure_cleanup(self):
        normal = ButterVisionPipeline("local.safetensors", device="cpu")
        pipe = SimpleNamespace(components={}, scheduler=SimpleNamespace(config={}))
        with patch.object(normal, "_resolve_model_source", return_value="local.safetensors"), patch("core.advanced_pipeline.StableDiffusionPipeline.from_single_file", return_value=pipe), patch("core.advanced_pipeline.DDIMScheduler.from_config", side_effect=RuntimeError("scheduler")):
            with self.assertRaises(RuntimeError):
                normal.load_sd_pipeline()
        self.assertIs(normal.sd_pipeline, pipe)
        normal.cleanup()
        self.assertIsNone(normal.sd_pipeline)

    def test_encoder_returns_to_cpu_when_embedding_calculation_fails(self):
        face = SD15FaceReferencePipeline("local.safetensors", device="cpu")
        face.image_encoder = Mock()
        with patch.object(face, "_encode_image", side_effect=RuntimeError("embedding")):
            with self.assertRaises(RuntimeError):
                face._encode_ip_adapter_image(None, None, 1, True)
        face.image_encoder.to.assert_called_once_with("cpu")

    def test_prompt_truncation_warns_without_text_and_short_prompts_unchanged(self):
        pipe = mock_pipe()
        warnings = []
        with self.assertLogs("uvicorn.error.engine", level="WARNING") as logs:
            positive, negative = limit_prompts(pipe, "private a b c d e f", "photo a b c d e", warnings.append)
        self.assertEqual(positive, "private a b c d")
        self.assertEqual(negative, "photo a b c d")
        self.assertEqual(len(warnings), 2)
        self.assertTrue(all("no se procesará" in warning for warning in warnings))
        self.assertNotIn("private", " ".join(logs.output))
        self.assertNotIn("photo", " ".join(logs.output))
        self.assertEqual(limit_prompts(pipe, "a b", ""), ("a b", ""))

    def test_txt2img_passes_only_bounded_text_to_mocked_inference(self):
        normal = ButterVisionPipeline("local.safetensors", device="cpu")
        pipe = mock_pipe()
        warnings = []
        with patch.object(normal, "load_sd_pipeline", return_value=pipe):
            normal.generate_image("a b c d e f g", prompt_warning_callback=warnings.append)
        self.assertEqual(pipe.call_args.kwargs["prompt"], "a b c d e")
        self.assertEqual(len(warnings), 1)

    def test_face_passes_only_bounded_text_to_mocked_inference(self):
        face = SD15FaceReferencePipeline("local.safetensors", device="cpu")
        pipe = mock_pipe()
        warnings = []
        with patch.object(face, "ensure_assets") as assets, patch.object(face, "_load_pipeline", return_value=pipe), patch.object(face, "_encode_ip_adapter_image", return_value=[]):
            face.generate(object(), "a b c d e f g", prompt_warning_callback=warnings.append)
        assets.assert_called_once_with(allow_download=False)
        self.assertEqual(pipe.call_args.kwargs["prompt"], "a b c d e")
        self.assertEqual(len(warnings), 1)

    def test_existing_cuda_precision_and_offload_policies_are_preserved(self):
        with patch("torch.cuda.is_available", return_value=True):
            normal = ButterVisionPipeline("local.safetensors", device="cuda")
            face = SD15FaceReferencePipeline("local.safetensors", device="cuda")
        pipe = mock_pipe()
        normal.available_loras = {}
        with patch.object(normal, "_resolve_model_source", return_value="local.safetensors"), patch.object(normal, "_prepare_pipeline_device", side_effect=lambda p: p), patch("core.advanced_pipeline.DDIMScheduler.from_config"), patch("diffusers.StableDiffusionPipeline.from_single_file", return_value=pipe) as factory:
            normal.load_sd_pipeline()
        self.assertIs(factory.call_args.kwargs["torch_dtype"], torch.float32)
        with patch("diffusers.StableDiffusionPipeline.from_single_file", return_value=pipe) as factory, patch("diffusers.DDIMScheduler.from_config"), patch.object(face, "_load_clip_vision", return_value=object()), patch.object(face, "_load_ip_adapter_state_dict", return_value={}):
            face._load_pipeline()
        self.assertIs(factory.call_args.kwargs["dtype"], torch.float16)
        pipe.enable_sequential_cpu_offload.assert_called_once()
        pipe.enable_xformers_memory_efficient_attention.assert_not_called()
        with patch.object(normal, "_apply_vram_optimizations", return_value=pipe):
            normal._prepare_pipeline_device(pipe)
        pipe.to.assert_not_called()


if __name__ == "__main__":
    unittest.main()
