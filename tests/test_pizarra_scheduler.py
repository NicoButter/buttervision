"""CPU-only PIZARRA scheduler and observational-budget regression tests."""
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch

import config
from core.pizarra.config import PizarraConfig
from core.pizarra.memory_budget import inspect_budget
from core.pizarra.scheduler import Scheduler, Task, prepare_block_matmul, reconstruct_block_matmul
from core.pizarra.storage import PizarraStorage


class MemoryBudgetTests(unittest.TestCase):
    def test_observational_budget_uses_measured_values_and_configured_margins(self):
        storage = Mock()
        storage.stats.return_value = {"total_bytes": 12000, "free_bytes": 5000,
                                      "managed_bytes": 1000, "quota_remaining_bytes": 4000,
                                      "available_bytes": 3500}
        process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=200))
        other = SimpleNamespace(info={"pid": os.getpid() + 100, "memory_info": SimpleNamespace(rss=350)})
        with patch("core.pizarra.memory_budget.psutil.virtual_memory", return_value=SimpleNamespace(total=2000, used=1000, available=1000)), patch("core.pizarra.memory_budget.psutil.Process", return_value=process), patch("core.pizarra.memory_budget.psutil.process_iter", return_value=[other]), patch("core.pizarra.memory_budget._cgroup_available", return_value=400), patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.mem_get_info", return_value=(3000, 4000)), patch("torch.cuda.memory_allocated", return_value=500), patch("torch.cuda.memory_reserved", return_value=700):
            before = (config.model_config.use_fp16, config.model_config.enable_cpu_offload, config.model_config.max_width)
            result = inspect_budget("txt2img", storage=storage, gpu_margin_fraction=0.1, ram_margin_fraction=0.2)
            after = (config.model_config.use_fp16, config.model_config.enable_cpu_offload, config.model_config.max_width)
        self.assertEqual(before, after)
        self.assertTrue(result["observational"])
        self.assertEqual(result["gpu"]["target_budget_bytes"], 2700)
        self.assertEqual(result["gpu"]["device_used_not_torch_reserved_bytes"], 300)
        self.assertEqual(result["ram"]["available_bytes"], 400)
        self.assertEqual(result["ram"]["target_budget_bytes"], 320)
        self.assertEqual(result["ram"]["visible_other_processes_rss_sum_bytes"], 350)
        self.assertEqual(result["pizarra"]["available_bytes"], 3500)

    def test_unavailable_measurements_remain_null(self):
        with patch("torch.cuda.is_available", return_value=False):
            value = inspect_budget("face_reference")
        self.assertIsNone(value["gpu"]["device_total_bytes"])
        self.assertIsNone(value["gpu"]["target_budget_bytes"])
        self.assertIsNone(value["pizarra"]["free_bytes"])
        with self.assertRaises(ValueError):
            inspect_budget("txt2img", ram_margin_fraction=1.5)
        with self.assertRaises(ValueError):
            inspect_budget("txt2img", gpu_margin_fraction=float("nan"))


class FakeStorage:
    def __init__(self):
        self.records = {}
        self.writes = []
        self.lock = threading.Lock()

    def commit(self, job_id, area, name, data):
        with self.lock:
            key = (job_id, f"{area}/{name}")
            if key in self.records:
                raise FileExistsError(name)
            self.records[key] = data
            self.writes.append(key)
            return key[1]

    def read(self, job_id, reference):
        return self.records[(job_id, reference)]


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "task-metadata.sqlite3"
        self.storage = FakeStorage()
        self.job = str(uuid.uuid4())

    def test_dependency_order_recovery_and_no_duplicate_execution(self):
        log = []
        def first(values):
            log.append("first")
            return b"first output"
        def second(values):
            log.append("second")
            self.assertEqual(values, [b"first output"])
            return b"second output"
        operations = {"first": first, "second": second}
        scheduler = Scheduler(self.db, self.storage, operations=operations)
        scheduler.submit(self.job, [
            Task("later", self.job, "second", dependencies=["earlier"], input_references=["@task:earlier"]),
            Task("earlier", self.job, "first", memory_budget={"gpu_bytes": 30}),
        ])
        tasks = scheduler.run_ready(self.job)
        self.assertEqual(log, ["first", "second"])
        self.assertEqual([task.status for task in tasks], ["completed", "completed"])
        restored = Scheduler(self.db, self.storage, operations=operations)
        self.assertEqual([task.status for task in restored.tasks(self.job)], ["completed", "completed"])
        restored.run_ready(self.job)
        self.assertEqual(log, ["first", "second"])
        self.assertNotIn(b"first output", self.db.read_bytes())

    def test_dependency_validation_cycle_and_duplicate_are_atomic(self):
        scheduler = Scheduler(self.db, self.storage)
        with self.assertRaisesRegex(ValueError, "inexistente"):
            scheduler.submit(self.job, [Task("a", self.job, "matrix_add", dependencies=["missing"])])
        with self.assertRaisesRegex(ValueError, "ciclo"):
            scheduler.submit(self.job, [Task("a", self.job, "matrix_add", dependencies=["b"]), Task("b", self.job, "matrix_add", dependencies=["a"])])
        with self.assertRaisesRegex(ValueError, "referencia"):
            scheduler.submit(self.job, [Task("a", self.job, "matrix_add", input_references=["@task:b"])])
        self.assertEqual(scheduler.tasks(self.job), [])
        scheduler.submit(self.job, [Task("a", self.job, "matrix_add")])
        with self.assertRaisesRegex(ValueError, "duplicado"):
            scheduler.submit(self.job, [Task("a", self.job, "matrix_add")])
        self.assertEqual(len(scheduler.tasks(self.job)), 1)

    def test_failure_leaves_no_confirmed_task_output_or_private_exception_text(self):
        def fail(values):
            raise RuntimeError("private prompt or input")
        scheduler = Scheduler(self.db, self.storage, operations={"fail": fail})
        scheduler.submit(self.job, [Task("first", self.job, "fail"), Task("dependent", self.job, "matrix_add", dependencies=["first"])])
        tasks = scheduler.run_ready(self.job)
        self.assertEqual([task.status for task in tasks], ["failed", "pending"])
        self.assertEqual(tasks[0].failure_category, "RuntimeError")
        self.assertEqual(tasks[0].output_references, [])
        self.assertEqual(self.storage.writes, [])
        self.assertNotIn(b"private prompt", self.db.read_bytes())

    def test_storage_error_marks_failure_without_publishing_a_result(self):
        storage = FakeStorage()
        storage.commit = Mock(side_effect=OSError("device unavailable"))
        scheduler = Scheduler(self.db, storage, operations={"work": lambda values: b"result"})
        scheduler.submit(self.job, [Task("first", self.job, "work")])
        task = scheduler.run_ready(self.job)[0]
        self.assertEqual(task.status, "failed")
        self.assertEqual(task.output_references, [])
        self.assertEqual(task.failure_category, "OSError")

    def test_large_valid_dependency_chain_does_not_exceed_recursion_depth(self):
        scheduler = Scheduler(self.db, self.storage)
        tasks = [Task(f"t{i}", self.job, "matrix_add", dependencies=[f"t{i-1}"] if i else []) for i in range(1500)]
        scheduler.submit(self.job, tasks)
        self.assertEqual(len(scheduler.tasks(self.job)), 1500)

    def test_dead_running_owner_is_interrupted_and_never_reexecuted(self):
        calls = []
        scheduler = Scheduler(self.db, self.storage, operations={"test": lambda values: calls.append(1) or b"x"})
        scheduler.submit(self.job, [Task("a", self.job, "test")])
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE pizarra_tasks SET status='running', owner_pid=12345, owner_start=0 WHERE task_id='a'")
        restored = Scheduler(self.db, self.storage, operations={"test": lambda values: calls.append(1) or b"x"}, owner_alive=lambda pid, start: False)
        self.assertEqual(restored.tasks(self.job)[0].status, "interrupted")
        restored.run_ready(self.job)
        self.assertEqual(calls, [])
        self.assertEqual(self.storage.writes, [])

    def test_two_scheduler_instances_do_not_repeat_a_running_task(self):
        started, release = threading.Event(), threading.Event()
        calls = []
        def work(values):
            calls.append(1)
            started.set()
            if not release.wait(3):
                raise RuntimeError("timeout")
            return b"done"
        scheduler = Scheduler(self.db, self.storage, operations={"test": work})
        scheduler.submit(self.job, [Task("a", self.job, "test")])
        worker = threading.Thread(target=scheduler.run_ready, args=(self.job,))
        worker.start()
        try:
            self.assertTrue(started.wait(3))
            second = Scheduler(self.db, self.storage, operations={"test": work})
            self.assertEqual(second.tasks(self.job)[0].status, "running")
            second.run_ready(self.job)
        finally:
            release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(calls, [1])
        self.assertEqual(second.tasks(self.job)[0].status, "completed")

    def test_block_matrix_reconstructs_using_atomic_storage_and_survives_restart(self):
        mount = Path(self.tmp.name) / "volume"
        mount.mkdir()
        dev = mount.stat().st_dev
        def mounted(path):
            if Path(path) == Path("/"):
                return ("root", "999:999")
            if Path(path) == mount:
                return ("fake-mount", f"{os.major(dev)}:{os.minor(dev)}")
            raise AssertionError(f"Unexpected mount: {path}")
        with patch("core.pizarra.storage._mount_identity", side_effect=mounted):
            storage = PizarraStorage(PizarraConfig(enabled=True, root=mount, quota_bytes=2_000_000, min_free_bytes=0))
            storage.create_workspace(self.job)
            scheduler = Scheduler(self.db, storage)
            left = [[1, 2, 3], [4, 5, 6], [-1, 0.5, 2]]
            right = [[7, 8, 1], [9, 10, 2], [11, 12, -1]]
            plan = prepare_block_matmul(scheduler, self.job, left, right, block_size=2)
            with self.assertRaisesRegex(ValueError, "incompletos"):
                reconstruct_block_matmul(scheduler, self.job, plan)
            self.assertTrue(all(task.status == "completed" for task in scheduler.run_ready(self.job)))
            restarted = Scheduler(self.db, storage)
            result = reconstruct_block_matmul(restarted, self.job, plan)
            expected = [[sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
            for i, line in enumerate(expected):
                for j, value in enumerate(line):
                    self.assertAlmostEqual(result[i][j], value, places=10)
            before = storage.stats()["managed_bytes"]
            restarted.run_ready(self.job)
            self.assertEqual(storage.stats()["managed_bytes"], before)

    def test_demonstration_rejects_excessive_task_graph_before_writing(self):
        scheduler = Scheduler(self.db, self.storage)
        matrix = [[1] * 32 for _ in range(32)]
        with self.assertRaisesRegex(ValueError, "Demasiadas"):
            prepare_block_matmul(scheduler, self.job, matrix, matrix, block_size=1)
        self.assertEqual(self.storage.writes, [])


if __name__ == "__main__":
    unittest.main()
