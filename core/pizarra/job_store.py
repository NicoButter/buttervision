"""Local SQLite administrative job ledger; payloads never enter this store."""
import json
import os
import sqlite3
import stat
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import psutil

_APPLICATION_ID = 0x505A3031


def _owner_alive(pid, started):
    if not pid:
        return False
    try:
        process = psutil.Process(pid)
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return False
        return started is None or abs(process.create_time() - started) < 0.01
    except (psutil.NoSuchProcess, ValueError):
        return False
    except (psutil.AccessDenied, OSError):
        # Lack of permission is not evidence of a dead process.
        return True


class JobStore:
    """Persist public job state and structured metrics in a local application DB.

    Connections are short lived to be safe across API and worker threads. This
    ledger does not contain checkpoint state and cannot resume an inference.
    """

    def __init__(self, path, *, owner_alive=None):
        self.path = Path(path)
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.geteuid() or parent.st_mode & 0o077:
            raise ValueError("El directorio local PIZARRA no es privado o seguro")
        self._owner_alive = owner_alive or _owner_alive
        self._owner_pid = os.getpid()
        try:
            self._owner_start = psutil.Process().create_time()
        except (psutil.Error, OSError):
            self._owner_start = None
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            directory = os.open(str(self.path.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            created = True
        except FileExistsError:
            created = False
        self._validate_files()
        with self._connect() as db:
            application_id = db.execute("PRAGMA application_id").fetchone()[0]
            if not created and application_id != _APPLICATION_ID:
                raise ValueError("La base local no pertenece a PIZARRA")
            if created:
                db.execute(f"PRAGMA application_id={_APPLICATION_ID}")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('queued','running','completed','failed','interrupted')),
                progress REAL NOT NULL,
                message TEXT NOT NULL,
                result_images TEXT NOT NULL,
                info TEXT,
                error TEXT,
                warnings TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                engine TEXT,
                model_identifier TEXT,
                failure_category TEXT,
                checkpoint_reference TEXT,
                metrics_reference TEXT,
                owner_pid INTEGER,
                owner_start REAL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS job_metrics (
                job_id TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
                metrics_json TEXT NOT NULL
            )""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            if "owner_pid" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN owner_pid INTEGER")
            if "owner_start" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN owner_start REAL")
            now = datetime.now().isoformat(timespec="seconds")
            rows = db.execute("SELECT job_id, owner_pid, owner_start FROM jobs WHERE status IN ('queued', 'running')").fetchall()
            for row in rows:
                if not self._owner_alive(row["owner_pid"], row["owner_start"]):
                    db.execute("""UPDATE jobs SET status='interrupted', message='Interrumpido por reinicio',
                               error='El proceso terminó antes de confirmar el resultado.',
                               failure_category='process_interrupted', updated_at=?, finished_at=?
                               WHERE job_id=? AND status IN ('queued', 'running')""", (now, now, row["job_id"]))

    def _validate_files(self):
        for suffix in ("", "-wal", "-shm", "-journal"):
            candidate = Path(str(self.path) + suffix)
            try:
                info = candidate.lstat()
            except FileNotFoundError:
                if suffix:
                    continue
                raise
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or info.st_mode & 0o077 or info.st_nlink != 1):
                raise ValueError("Archivo local PIZARRA inseguro")

    @contextmanager
    def _connect(self):
        self._validate_files()
        db = sqlite3.connect(str(self.path), timeout=10)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def create(self, snapshot, *, engine, model_identifier):
        # Explicit allowlist: no payload, images, prompts or arbitrary kwargs.
        with self._connect() as db:
            db.execute("""INSERT INTO jobs (job_id, mode, status, progress, message,
                       result_images, info, error, warnings, created_at, updated_at,
                       engine, model_identifier, owner_pid, owner_start) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                       (snapshot["id"], snapshot["mode"], snapshot["status"], snapshot["progress"],
                        snapshot["message"], json.dumps(snapshot["result_images"]), snapshot["info"],
                        snapshot["error"], json.dumps(snapshot["warnings"]), snapshot["created_at"],
                        snapshot["updated_at"], engine, model_identifier, self._owner_pid, self._owner_start))

    def update(self, snapshot, *, failure_category=None):
        with self._connect() as db:
            self._update(db, snapshot, failure_category)

    def _update(self, db, snapshot, failure_category):
        status = snapshot["status"]
        now = snapshot["updated_at"]
        cursor = db.execute("""UPDATE jobs SET status=?, progress=?, message=?, result_images=?,
                info=?, error=?, warnings=?, updated_at=?,
                started_at=CASE WHEN ?='running' AND started_at IS NULL THEN ? ELSE started_at END,
                finished_at=CASE WHEN ? IN ('completed','failed','interrupted') THEN ? ELSE finished_at END,
                failure_category=? WHERE job_id=?""",
            (status, snapshot["progress"], snapshot["message"], json.dumps(snapshot["result_images"]),
             snapshot["info"], snapshot["error"], json.dumps(snapshot["warnings"]), now,
             status, now, status, now, failure_category, snapshot["id"]))
        if cursor.rowcount != 1:
            raise KeyError("Trabajo persistente no encontrado")

    def get(self, job_id):
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["id"] = result.pop("job_id")
        for key in ("result_images", "warnings"):
            result[key] = json.loads(result[key])
        return result

    def set_metrics(self, job_id, metrics):
        with self._connect() as db:
            self._set_metrics(db, job_id, metrics)

    def _set_metrics(self, db, job_id, metrics):
        encoded = json.dumps(metrics, allow_nan=False, separators=(",", ":"))
        db.execute("""INSERT INTO job_metrics(job_id, metrics_json) VALUES (?, ?)
                          ON CONFLICT(job_id) DO UPDATE SET metrics_json=excluded.metrics_json""", (job_id, encoded))
        db.execute("UPDATE jobs SET metrics_reference=? WHERE job_id=?", (f"sqlite:{job_id}", job_id))

    def finalize(self, snapshot, metrics, *, failure_category=None):
        """Publish terminal state and its metrics in the same transaction."""
        with self._connect() as db:
            self._update(db, snapshot, failure_category)
            if metrics is not None:
                self._set_metrics(db, snapshot["id"], metrics)

    def get_metrics(self, job_id):
        with self._connect() as db:
            row = db.execute("SELECT metrics_json FROM job_metrics WHERE job_id=?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else None
