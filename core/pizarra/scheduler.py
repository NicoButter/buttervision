"""Serial, CPU-only experimental task scheduler, independent of diffusion.

Tasks produce exactly one immutable, storage-confirmed output. SQLite stores
administrative state and references, never tensor contents. A task claimed by a
process that dies becomes interrupted, never automatically retried. This avoids
executing twice after a crash between the output commit and the SQLite update.
"""
import json
import math
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import psutil

_TASK_ID = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
_MAX_DIMENSION = 256  # This demonstrator only permits small CPU matrices.


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _job_id(value):
    try:
        result = str(uuid.UUID(value))
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("job_id debe ser UUID canónico") from error
    if result != value:
        raise ValueError("job_id debe ser UUID canónico")
    return result


def _task_id(value):
    if not isinstance(value, str) or not _TASK_ID.fullmatch(value):
        raise ValueError("task_id inválido")
    return value


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _matrix(value):
    if (not isinstance(value, list) or not value or len(value) > _MAX_DIMENSION or
            not isinstance(value[0], list) or not value[0] or len(value[0]) > _MAX_DIMENSION):
        raise ValueError("Matriz vacía, inválida o demasiado grande")
    width = len(value[0])
    result = []
    for row in value:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("La matriz debe ser rectangular")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in row):
            raise ValueError("La matriz debe contener números finitos")
        result.append([float(x) for x in row])
    return result


def _encode_matrix(matrix):
    return _json(matrix).encode("utf-8")


def _decode_matrix(data):
    if not isinstance(data, bytes):
        raise ValueError("El tensor de entrada debe ser bytes JSON")
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("Tensor JSON demasiado grande")
    try:
        return _matrix(json.loads(data.decode("utf-8")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Tensor JSON inválido") from error


def _multiply(inputs):
    if len(inputs) != 2:
        raise ValueError("matrix_multiply requiere dos entradas")
    left, right = map(_decode_matrix, inputs)
    if len(left[0]) != len(right):
        raise ValueError("Dimensiones incompatibles")
    result = [[math.fsum(left[i][k] * right[k][j] for k in range(len(right)))
               for j in range(len(right[0]))] for i in range(len(left))]
    return _encode_matrix(_matrix(result))


def _add(inputs):
    if not inputs:
        raise ValueError("matrix_add requiere al menos una entrada")
    matrices = [_decode_matrix(value) for value in inputs]
    rows, cols = len(matrices[0]), len(matrices[0][0])
    if any(len(item) != rows or len(item[0]) != cols for item in matrices):
        raise ValueError("Dimensiones incompatibles")
    result = [[math.fsum(item[i][j] for item in matrices) for j in range(cols)] for i in range(rows)]
    return _encode_matrix(_matrix(result))


@dataclass
class Task:
    task_id: str
    job_id: str
    operation: str
    dependencies: list = field(default_factory=list)
    input_references: list = field(default_factory=list)
    output_references: list = field(default_factory=list)
    memory_budget: dict = field(default_factory=dict)
    status: str = "pending"
    failure_category: str = None
    updated_at: str = None


def _owner_alive(pid, start):
    if not pid:
        return False
    try:
        process = psutil.Process(pid)
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return False
        # A reused PID is not the process that claimed the task.
        return start is None or abs(process.create_time() - start) < 0.01
    except (psutil.NoSuchProcess, ValueError):
        return False
    except (psutil.AccessDenied, OSError):
        return True  # Unknown is not proof that another worker died.


class Scheduler:
    """Run independent prepared tasks serially, persisting state locally.

    ``database_path`` is an application-controlled local SQLite path, separate
    from the removable storage. ``storage`` must expose ``commit`` and ``read``.
    Each ``job_id`` workspace must already have been created by the caller.
    Custom handlers are trusted Python CPU executors; there are no GPU workers.
    """
    def __init__(self, database_path, storage, *, operations=None, owner_alive=None):
        self.path = str(Path(database_path))
        self.storage = storage
        self.operations = {"matrix_multiply": _multiply, "matrix_add": _add}
        if operations:
            self.operations.update(operations)
        self._run_lock = threading.Lock()
        self._owner_alive = owner_alive or _owner_alive
        try:
            self._owner_start = psutil.Process().create_time()
        except (psutil.Error, OSError):
            self._owner_start = None
        import os
        self._owner_pid = os.getpid()
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS pizarra_tasks (
                job_id TEXT NOT NULL, task_id TEXT NOT NULL, operation TEXT NOT NULL,
                dependencies TEXT NOT NULL, input_references TEXT NOT NULL,
                output_references TEXT NOT NULL, memory_budget TEXT NOT NULL,
                status TEXT NOT NULL, failure_category TEXT, updated_at TEXT NOT NULL,
                owner_pid INTEGER, owner_start REAL, PRIMARY KEY(job_id, task_id)
            )""")
        self.recover_abandoned()

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def recover_abandoned(self):
        """Mark tasks of proven-dead owners interrupted; never execute them."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = conn.execute("SELECT job_id, task_id, owner_pid, owner_start FROM pizarra_tasks WHERE status='running'").fetchall()
                for row in rows:
                    if not self._owner_alive(row["owner_pid"], row["owner_start"]):
                        conn.execute("UPDATE pizarra_tasks SET status='interrupted', failure_category='owner_interrupted', updated_at=? WHERE job_id=? AND task_id=? AND status='running'",
                                     (_now(), row["job_id"], row["task_id"]))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def tasks(self, job_id):
        _job_id(job_id)
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM pizarra_tasks WHERE job_id=? ORDER BY rowid", (job_id,)).fetchall()
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row):
        return Task(task_id=row["task_id"], job_id=row["job_id"], operation=row["operation"],
                    dependencies=json.loads(row["dependencies"]), input_references=json.loads(row["input_references"]),
                    output_references=json.loads(row["output_references"]), memory_budget=json.loads(row["memory_budget"]),
                    status=row["status"], failure_category=row["failure_category"], updated_at=row["updated_at"])

    def submit(self, job_id, tasks):
        """Atomically register a DAG; reject duplicates, unknown ops and cycles."""
        _job_id(job_id)
        tasks = list(tasks)
        if not tasks:
            raise ValueError("No hay tareas para registrar")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = {row["task_id"]: json.loads(row["dependencies"]) for row in conn.execute(
                    "SELECT task_id, dependencies FROM pizarra_tasks WHERE job_id=?", (job_id,))}
                graph = dict(existing)
                for task in tasks:
                    if not isinstance(task, Task) or task.job_id != job_id:
                        raise ValueError("Tarea inválida o de otro trabajo")
                    task_id = _task_id(task.task_id)
                    if task_id in graph:
                        raise ValueError("task_id duplicado")
                    if task.status != "pending" or task.output_references or task.failure_category:
                        raise ValueError("Sólo se registran tareas nuevas pendientes")
                    if task.operation not in self.operations:
                        raise ValueError("Operación desconocida")
                    if (not isinstance(task.dependencies, list) or
                            not all(isinstance(dep, str) for dep in task.dependencies) or
                            len(set(task.dependencies)) != len(task.dependencies)):
                        raise ValueError("Dependencias inválidas")
                    if not isinstance(task.input_references, list) or not all(isinstance(ref, str) for ref in task.input_references):
                        raise ValueError("Referencias de entrada inválidas")
                    for dependency in task.dependencies:
                        _task_id(dependency)
                    for reference in task.input_references:
                        if reference.startswith("@task:") and reference[6:] not in task.dependencies:
                            raise ValueError("La referencia de tarea requiere una dependencia explícita")
                    if not isinstance(task.memory_budget, dict):
                        raise ValueError("El presupuesto debe ser un objeto")
                    # Reject non-JSON and non-finite values prior to insertion.
                    _json(task.memory_budget)
                    graph[task_id] = list(task.dependencies)
                if any(dependency not in graph for deps in graph.values() for dependency in deps):
                    raise ValueError("Dependencia inexistente")
                # Iterative topological validation avoids a recursion limit for
                # long but valid chains of tasks.
                remaining = {node: len(deps) for node, deps in graph.items()}
                children = {node: [] for node in graph}
                for node, deps in graph.items():
                    for dependency in deps:
                        children[dependency].append(node)
                ready = [node for node, count in remaining.items() if count == 0]
                visited = 0
                while ready:
                    node = ready.pop()
                    visited += 1
                    for child in children[node]:
                        remaining[child] -= 1
                        if remaining[child] == 0:
                            ready.append(child)
                if visited != len(graph):
                    raise ValueError("El grafo contiene un ciclo")
                for task in tasks:
                    conn.execute("""INSERT INTO pizarra_tasks
                        (job_id, task_id, operation, dependencies, input_references, output_references,
                         memory_budget, status, failure_category, updated_at)
                        VALUES (?, ?, ?, ?, ?, '[]', ?, 'pending', NULL, ?)""",
                                 (job_id, task.task_id, task.operation, _json(task.dependencies),
                                  _json(task.input_references), _json(task.memory_budget), _now()))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _claim_ready(self, job_id):
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = conn.execute("SELECT * FROM pizarra_tasks WHERE job_id=? ORDER BY rowid", (job_id,)).fetchall()
                state = {row["task_id"]: row["status"] for row in rows}
                ready = next((row for row in rows if row["status"] == "pending" and all(
                    state.get(dep) == "completed" for dep in json.loads(row["dependencies"]))), None)
                if ready is None:
                    conn.commit()
                    return None
                result = conn.execute("UPDATE pizarra_tasks SET status='running', owner_pid=?, owner_start=?, updated_at=? WHERE job_id=? AND task_id=? AND status='pending'",
                                      (self._owner_pid, self._owner_start, _now(), job_id, ready["task_id"]))
                conn.commit()
                return self._from_row(ready) if result.rowcount == 1 else None
            except Exception:
                conn.rollback()
                raise

    def _inputs(self, task):
        results = []
        for reference in task.input_references:
            if reference.startswith("@task:"):
                dependency = reference[6:]
                with self._connect() as conn:
                    row = conn.execute("SELECT status, output_references FROM pizarra_tasks WHERE job_id=? AND task_id=?",
                                       (task.job_id, dependency)).fetchone()
                if row is None or row["status"] != "completed":
                    raise RuntimeError("La dependencia no está completa")
                refs = json.loads(row["output_references"])
                if len(refs) != 1:
                    raise RuntimeError("Referencia de salida inválida")
                reference = refs[0]
            results.append(self.storage.read(task.job_id, reference))
        return results

    def run_ready(self, job_id):
        """Execute ready CPU tasks once, sequentially; stop on first failure."""
        _job_id(job_id)
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("El planificador ya está ejecutando tareas")
        try:
            while True:
                task = self._claim_ready(job_id)
                if task is None:
                    return self.tasks(job_id)
                try:
                    output = self.operations[task.operation](self._inputs(task))
                    if not isinstance(output, bytes):
                        raise ValueError("El ejecutor debe devolver bytes")
                    reference = self.storage.commit(job_id, "tensors", f"task_{task.task_id}.bin", output)
                except Exception as error:
                    with self._connect() as conn:
                        conn.execute("UPDATE pizarra_tasks SET status='failed', failure_category=?, updated_at=? WHERE job_id=? AND task_id=? AND status='running'",
                                     (type(error).__name__, _now(), job_id, task.task_id))
                    return self.tasks(job_id)
                # If this administrative update fails the task stays running;
                # recovery marks it interrupted rather than executing it twice.
                with self._connect() as conn:
                    conn.execute("UPDATE pizarra_tasks SET status='completed', output_references=?, updated_at=? WHERE job_id=? AND task_id=? AND status='running'",
                                 (_json([reference]), _now(), job_id, task.task_id))
        finally:
            self._run_lock.release()


def prepare_block_matmul(scheduler, job_id, left, right, *, block_size=2, memory_budget=None):
    """Store small blocks and register their multiplication and reduction DAG."""
    _job_id(job_id)
    left, right = _matrix(left), _matrix(right)
    if len(left[0]) != len(right):
        raise ValueError("Dimensiones incompatibles")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or not 1 <= block_size <= _MAX_DIMENSION:
        raise ValueError("block_size inválido")
    if memory_budget is not None and not isinstance(memory_budget, dict):
        raise ValueError("El presupuesto debe ser un objeto")
    _json(memory_budget or {})
    prefix = uuid.uuid4().hex[:16]
    a_refs, b_refs, tasks, blocks = {}, {}, [], []
    rows, inner, cols = len(left), len(right), len(right[0])
    task_count = math.ceil(rows / block_size) * math.ceil(cols / block_size) * (math.ceil(inner / block_size) + 1)
    if task_count > 4096:
        raise ValueError("Demasiadas tareas para el demostrador CPU")
    budget = memory_budget or {}
    for row in range(0, rows, block_size):
        for col in range(0, cols, block_size):
            partial_ids = []
            for k in range(0, inner, block_size):
                if (row, k) not in a_refs:
                    block = [r[k:k + block_size] for r in left[row:row + block_size]]
                    a_refs[(row, k)] = scheduler.storage.commit(job_id, "tensors", f"{prefix}_a_{row}_{k}.json", _encode_matrix(block))
                if (k, col) not in b_refs:
                    block = [r[col:col + block_size] for r in right[k:k + block_size]]
                    b_refs[(k, col)] = scheduler.storage.commit(job_id, "tensors", f"{prefix}_b_{k}_{col}.json", _encode_matrix(block))
                task_id = f"{prefix}_m_{row}_{col}_{k}"
                partial_ids.append(task_id)
                tasks.append(Task(task_id=task_id, job_id=job_id, operation="matrix_multiply",
                                  input_references=[a_refs[(row, k)], b_refs[(k, col)]], memory_budget=budget))
            output_id = f"{prefix}_sum_{row}_{col}"
            tasks.append(Task(task_id=output_id, job_id=job_id, operation="matrix_add", dependencies=partial_ids,
                              input_references=[f"@task:{tid}" for tid in partial_ids], memory_budget=budget))
            blocks.append({"task_id": output_id, "row": row, "column": col})
    scheduler.submit(job_id, tasks)
    descriptor = {"format_version": 1, "job_id": job_id, "rows": rows, "columns": cols, "blocks": blocks}
    descriptor_ref = scheduler.storage.commit(job_id, "checkpoints", f"{prefix}_matrix_plan.json", _json(descriptor).encode("utf-8"))
    return descriptor_ref


def reconstruct_block_matmul(scheduler, job_id, descriptor_ref):
    """Reassemble only committed completed output blocks, with shape checks."""
    _job_id(job_id)
    try:
        descriptor = json.loads(scheduler.storage.read(job_id, descriptor_ref))
        if descriptor["format_version"] != 1 or descriptor["job_id"] != job_id:
            raise ValueError("Plan incompatible")
        rows, cols = descriptor["rows"], descriptor["columns"]
        if not isinstance(rows, int) or not isinstance(cols, int) or not 1 <= rows <= _MAX_DIMENSION or not 1 <= cols <= _MAX_DIMENSION:
            raise ValueError("Dimensiones inválidas")
        output = [[None for _ in range(cols)] for _ in range(rows)]
        tasks = {task.task_id: task for task in scheduler.tasks(job_id)}
        for block in descriptor["blocks"]:
            task = tasks[block["task_id"]]
            if task.status != "completed" or len(task.output_references) != 1:
                raise ValueError("Hay bloques incompletos")
            matrix = _decode_matrix(scheduler.storage.read(job_id, task.output_references[0]))
            row, col = block["row"], block["column"]
            if not isinstance(row, int) or not isinstance(col, int) or row < 0 or col < 0 or row + len(matrix) > rows or col + len(matrix[0]) > cols:
                raise ValueError("Bloque fuera de límites")
            for i, values in enumerate(matrix):
                for j, value in enumerate(values):
                    if output[row + i][col + j] is not None:
                        raise ValueError("Bloques solapados")
                    output[row + i][col + j] = value
        if any(value is None for line in output for value in line):
            raise ValueError("Faltan bloques")
        return output
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Plan de bloques inválido") from error
