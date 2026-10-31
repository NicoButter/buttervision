"""Local volume storage: no following links, immutable atomic records and quotas.

The configured root must itself be an independently mounted volume. This
conservative policy deliberately rejects ordinary directories and same-device
bind mounts, so an unplugged drive cannot silently redirect writes to '/'.
"""
import contextlib
import errno
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import threading
import time
import uuid

from .config import PizarraConfig


class PizarraStorageError(RuntimeError):
    pass


class StorageUnavailable(PizarraStorageError):
    pass


class QuotaExceeded(PizarraStorageError):
    pass


class IntegrityError(PizarraStorageError):
    pass


class UnsafePath(PizarraStorageError):
    pass


class AlreadyExists(PizarraStorageError):
    pass


class NotFound(PizarraStorageError):
    pass


_LOCK_NAME = ".pizarra.lock"
_LOCK_MAGIC = b"BUTTERVISION-PIZARRA-LOCK-v1\n"
_RECORD_MAGIC = b"PIZARRA1\n"
_VERSION = 1
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_AREAS = frozenset(("checkpoints", "tensors"))
_OPEN_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _uuid(value):
    try:
        parsed = str(uuid.UUID(value))
    except (ValueError, TypeError, AttributeError) as error:
        raise UnsafePath("Identificador interno inválido") from error
    if parsed != value:
        raise UnsafePath("Identificador interno inválido")
    return parsed


def _name(value):
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value) or ".." in value:
        raise UnsafePath("Nombre de archivo inválido")
    return value


def _unescape_mount(value):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def _mount_identity(path):
    """Return mount ID and device for the exact path, not just its parent."""
    try:
        with open("/proc/self/mountinfo", encoding="utf-8") as source:
            for line in source:
                parts = line.split()
                if len(parts) > 5 and _unescape_mount(parts[4]) == str(path):
                    return parts[0], parts[2]
    except OSError as error:
        raise StorageUnavailable("No se pudo comprobar el volumen montado") from error
    raise StorageUnavailable("La raíz PIZARRA no es un punto de montaje activo")


def _uuid_device(value):
    try:
        device = os.stat(Path("/dev/disk/by-uuid") / value)
    except OSError as error:
        raise StorageUnavailable("El UUID del volumen autorizado no está disponible") from error
    if not stat.S_ISBLK(device.st_mode):
        raise StorageUnavailable("El UUID autorizado no corresponde a un dispositivo de bloque")
    return device.st_rdev


def _open_path_without_symlinks(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or path == Path("/"):
        raise UnsafePath("La raíz PIZARRA debe ser una ruta absoluta segura")
    fd = os.open("/", _OPEN_DIR)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, _OPEN_DIR, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except OSError as error:
        os.close(fd)
        raise StorageUnavailable("La raíz PIZARRA no está accesible o contiene enlaces simbólicos") from error


def _directory(parent_fd, name, *, create=False):
    try:
        if create:
            try:
                os.mkdir(name, 0o700, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except FileExistsError:
                pass
        fd = os.open(name, _OPEN_DIR, dir_fd=parent_fd)
        try:
            if _fd_mount_id(fd) != _fd_mount_id(parent_fd):
                raise UnsafePath("Se detectó otro volumen dentro de PIZARRA")
            return fd
        except Exception:
            os.close(fd)
            raise
    except OSError as error:
        if error.errno in (errno.ENOENT, errno.ELOOP, errno.ENOTDIR):
            raise UnsafePath("Directorio PIZARRA no disponible o inseguro") from error
        raise StorageUnavailable("No se pudo acceder al directorio PIZARRA") from error


def _fd_mount_id(fd):
    try:
        with open(f"/proc/self/fdinfo/{fd}", encoding="ascii") as info:
            for line in info:
                if line.startswith("mnt_id:"):
                    return int(line.split(":", 1)[1])
    except (OSError, ValueError) as error:
        raise StorageUnavailable("No se pudo comprobar el volumen del directorio") from error
    raise StorageUnavailable("No se pudo comprobar el volumen del directorio")


def _write_all(fd, data, account=None):
    view = memoryview(data)
    while view:
        started = time.monotonic()
        count = os.write(fd, view)
        if account:
            account(count, time.monotonic() - started)
        if not count:
            raise OSError("escritura incompleta")
        view = view[count:]


def _read_all(fd, max_bytes=None, account=None):
    result = bytearray()
    while True:
        started = time.monotonic()
        part = os.read(fd, 1024 * 1024)
        if account:
            account(len(part), time.monotonic() - started)
        if not part:
            return bytes(result)
        result.extend(part)
        if max_bytes is not None and len(result) > max_bytes:
            raise IntegrityError("Archivo de metadatos demasiado grande")


def _read_file(parent_fd, name, limit=None, account=None):
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise UnsafePath("Se esperaba un archivo regular")
            return _read_all(fd, limit, account)
        finally:
            os.close(fd)
    except PizarraStorageError:
        raise
    except FileNotFoundError as error:
        raise NotFound("Archivo PIZARRA inexistente") from error
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise UnsafePath("Archivo PIZARRA inseguro") from error
        raise StorageUnavailable("No se pudo leer el archivo PIZARRA") from error


def _count_tree(fd):
    total = 0
    for entry in os.listdir(fd):
        try:
            info = os.stat(entry, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
            elif stat.S_ISDIR(info.st_mode):
                child = os.open(entry, _OPEN_DIR, dir_fd=fd)
                try:
                    # Nested mountpoints are not part of the authorized volume.
                    if (os.fstat(child).st_dev != os.fstat(fd).st_dev or
                            _fd_mount_id(child) != _fd_mount_id(fd)):
                        raise UnsafePath("Se detectó otro volumen dentro de PIZARRA")
                    total += _count_tree(child)
                finally:
                    os.close(child)
            # Symlinks and special files are never followed or removed.
        except FileNotFoundError as error:
            raise StorageUnavailable("El contenido del volumen cambió durante la inspección") from error
    return total


class PizarraStorage:
    def __init__(self, config: PizarraConfig):
        self.config = config
        self._identity = None
        self._thread_lock = threading.RLock()
        self._read_bytes = 0 if config.metrics_enabled else None
        self._written_bytes = 0 if config.metrics_enabled else None
        self._read_seconds = 0.0 if config.metrics_enabled else None
        self._write_seconds = 0.0 if config.metrics_enabled else None

    def _account_read(self, count, elapsed):
        if self.config.metrics_enabled:
            with self._thread_lock:
                self._read_bytes += count
                self._read_seconds += elapsed

    def _account_write(self, count, elapsed):
        if self.config.metrics_enabled:
            with self._thread_lock:
                self._written_bytes += count
                self._write_seconds += elapsed

    @contextlib.contextmanager
    def _root(self):
        if not self.config.enabled:
            raise StorageUnavailable("PIZARRA está deshabilitada")
        try:
            fd = _open_path_without_symlinks(self.config.root)
            try:
                identity = _mount_identity(self.config.root)
                root_identity = _mount_identity(Path("/"))
                dev = os.fstat(fd).st_dev
                if identity[1] == root_identity[1] or identity[1] != f"{os.major(dev)}:{os.minor(dev)}":
                    raise StorageUnavailable("PIZARRA requiere un volumen montado separado del disco raíz")
                if self.config.volume_uuid:
                    if _uuid_device(self.config.volume_uuid) != dev:
                        raise StorageUnavailable("El volumen montado no coincide con el UUID autorizado")
                info = os.stat(self.config.root, follow_symlinks=False)
                if (info.st_dev, info.st_ino) != (dev, os.fstat(fd).st_ino):
                    raise StorageUnavailable("El volumen cambió durante la validación")
                if self._identity is None:
                    self._identity = (identity, info.st_dev, info.st_ino)
                elif self._identity != (identity, info.st_dev, info.st_ino):
                    raise StorageUnavailable("El volumen PIZARRA cambió o fue desmontado")
                yield fd
            finally:
                os.close(fd)
        except PizarraStorageError:
            raise
        except OSError as error:
            raise StorageUnavailable("El volumen PIZARRA dejó de estar disponible") from error

    def _check_root(self, fd):
        identity = _mount_identity(self.config.root)
        path_stat = os.stat(self.config.root, follow_symlinks=False)
        if self._identity != (identity, path_stat.st_dev, path_stat.st_ino) or (
            os.fstat(fd).st_dev, os.fstat(fd).st_ino
        ) != (path_stat.st_dev, path_stat.st_ino):
            raise StorageUnavailable("El volumen PIZARRA cambió durante la operación")

    @contextlib.contextmanager
    def _locked(self):
        # Flock serializes quota checks and publications across processes;
        # RLock serializes threads sharing this instance.
        with self._thread_lock, self._root() as root:
            try:
                os.stat(_LOCK_NAME, dir_fd=root, follow_symlinks=False)
            except FileNotFoundError:
                self._ensure_capacity(root, len(_LOCK_MAGIC))
            try:
                created = False
                try:
                    lock = os.open(_LOCK_NAME, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=root)
                    created = True
                except FileExistsError:
                    lock = os.open(_LOCK_NAME, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root)
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    if not stat.S_ISREG(os.fstat(lock).st_mode):
                        raise UnsafePath("El archivo de bloqueo no es regular")
                    if created:
                        _write_all(lock, _LOCK_MAGIC, self._account_write)
                        os.fsync(lock)
                        os.fsync(root)
                    else:
                        os.lseek(lock, 0, os.SEEK_SET)
                        if _read_all(lock, 1024, self._account_read) != _LOCK_MAGIC:
                            raise UnsafePath("El archivo de bloqueo pertenece a otra aplicación")
                    self._check_root(root)
                    yield root
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    os.close(lock)
            except PizarraStorageError:
                raise
            except OSError as error:
                raise StorageUnavailable("No se pudo completar la operación en el volumen PIZARRA") from error

    def _managed_bytes(self, root):
        total = 0
        try:
            info = os.stat(_LOCK_NAME, dir_fd=root, follow_symlinks=False)
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
        except FileNotFoundError:
            pass
        for name in ("jobs", "cache", "benchmarks"):
            try:
                child = _directory(root, name)
            except UnsafePath as error:
                try:
                    os.stat(name, dir_fd=root, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                raise error
            try:
                if os.fstat(child).st_dev != os.fstat(root).st_dev:
                    raise UnsafePath("Se detectó otro volumen dentro de PIZARRA")
                total += _count_tree(child)
            finally:
                os.close(child)
        return total

    def _ensure_capacity(self, root, size):
        fs = os.fstatvfs(root)
        free = fs.f_bavail * fs.f_frsize
        managed = self._managed_bytes(root)
        if managed + size > self.config.quota_bytes:
            raise QuotaExceeded("Se alcanzó la cuota PIZARRA")
        if free - size < self.config.min_free_bytes:
            raise QuotaExceeded("No hay espacio libre suficiente para mantener la reserva")

    def _snapshot(self, root):
        fs = os.fstatvfs(root)
        free = fs.f_bavail * fs.f_frsize
        managed = self._managed_bytes(root)
        quota_left = max(0, self.config.quota_bytes - managed)
        usable = min(quota_left, max(0, free - self.config.min_free_bytes))
        return {"enabled": True, "available": True, "total_bytes": fs.f_blocks * fs.f_frsize,
                "free_bytes": free, "managed_bytes": managed, "quota_bytes": self.config.quota_bytes,
                "quota_remaining_bytes": quota_left, "min_free_bytes": self.config.min_free_bytes,
                "available_bytes": usable, "read_bytes": self._read_bytes, "written_bytes": self._written_bytes,
                "read_seconds": self._read_seconds, "write_seconds": self._write_seconds,
                "read_bytes_per_second": (self._read_bytes / self._read_seconds if self._read_seconds else None),
                "write_bytes_per_second": (self._written_bytes / self._write_seconds if self._write_seconds else None)}

    def validate(self):
        if not self.config.enabled:
            return {"enabled": False, "available": False}
        with self._thread_lock, self._root() as root:
            return self._snapshot(root)

    def stats(self):
        return self.validate()

    @contextlib.contextmanager
    def _workspace(self, root, job_id):
        job_id = _uuid(job_id)
        jobs = _directory(root, "jobs")
        try:
            job = _directory(jobs, job_id)
            try:
                manifest = json.loads(_read_file(job, "manifest.json", 16_384, self._account_read))
                checksum = manifest.pop("checksum", None)
                if (manifest.get("format_version") != _VERSION or manifest.get("job_id") != job_id or
                    manifest.get("owner") != "buttervision-pizarra" or checksum != hashlib.sha256(_canonical_json(manifest)).hexdigest()):
                    raise IntegrityError("Manifiesto inválido o incompatible")
                _uuid(manifest.get("workspace_token"))
                yield job, manifest
            except (ValueError, TypeError, KeyError) as error:
                raise IntegrityError("Manifiesto PIZARRA inválido") from error
            finally:
                os.close(job)
        finally:
            os.close(jobs)

    def create_workspace(self, job_id):
        job_id = _uuid(job_id)
        manifest = {"format_version": _VERSION, "job_id": job_id, "owner": "buttervision-pizarra",
                    "workspace_token": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat()}
        manifest["checksum"] = hashlib.sha256(_canonical_json(manifest)).hexdigest()
        data = _canonical_json(manifest)
        with self._locked() as root:
            self._ensure_capacity(root, len(data))
            for name in ("jobs", "cache", "benchmarks"):
                fd = _directory(root, name, create=True)
                os.close(fd)
            jobs = _directory(root, "jobs")
            try:
                try:
                    os.mkdir(job_id, 0o700, dir_fd=jobs)
                except FileExistsError as error:
                    raise AlreadyExists("Ya existe un espacio con ese identificador") from error
                os.fsync(jobs)
                job = _directory(jobs, job_id)
                try:
                    for name in ("checkpoints", "tensors", "temporary"):
                        fd = _directory(job, name, create=True)
                        os.close(fd)
                    fd = os.open("manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=job)
                    try:
                        _write_all(fd, data, self._account_write)
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    os.fsync(job)
                    self._ensure_capacity(root, 0)
                    self._check_root(root)
                finally:
                    os.close(job)
            finally:
                os.close(jobs)
        return {"job_id": job_id, "format_version": _VERSION}

    def _record(self, data):
        if not isinstance(data, bytes):
            raise TypeError("El contenido debe ser bytes")
        header = _canonical_json({"format_version": _VERSION, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        return _RECORD_MAGIC + struct.pack(">I", len(header)) + header, data

    def _unpack(self, record):
        pos = len(_RECORD_MAGIC)
        if not record.startswith(_RECORD_MAGIC) or len(record) < pos + 4:
            raise IntegrityError("Registro PIZARRA incompleto")
        length = struct.unpack(">I", record[pos:pos + 4])[0]
        if length > 16_384 or len(record) < pos + 4 + length:
            raise IntegrityError("Cabecera PIZARRA incompleta")
        try:
            header = json.loads(record[pos + 4:pos + 4 + length])
            payload = record[pos + 4 + length:]
            if (header.get("format_version") != _VERSION or header.get("size") != len(payload) or
                    header.get("sha256") != hashlib.sha256(payload).hexdigest()):
                raise IntegrityError("Checksum o longitud PIZARRA inválidos")
            return payload
        except (ValueError, TypeError) as error:
            raise IntegrityError("Cabecera PIZARRA inválida") from error

    def write_temporary(self, job_id, data):
        header, payload = self._record(data)
        token = str(uuid.uuid4())
        marker_name, data_name = f"lease-{token}.json", f"pending-{token}.part"
        with self._locked() as root, self._workspace(root, job_id) as (job, manifest):
            marker = _canonical_json({"version": _VERSION, "job_id": job_id, "id": token,
                                      "workspace_token": manifest["workspace_token"], "created_at": time.time()})
            self._ensure_capacity(root, len(marker) + len(header) + len(payload))
            temp = _directory(job, "temporary")
            try:
                marker_fd = os.open(marker_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=temp)
                try:
                    _write_all(marker_fd, marker, self._account_write)
                    os.fsync(marker_fd)
                finally:
                    os.close(marker_fd)
                os.fsync(temp)
                data_fd = os.open(data_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=temp)
                try:
                    _write_all(data_fd, header, self._account_write)
                    _write_all(data_fd, payload, self._account_write)
                    os.fsync(data_fd)
                finally:
                    os.close(data_fd)
                os.fsync(temp)
                self._ensure_capacity(root, 0)
                self._check_root(root)
            finally:
                os.close(temp)
        return token

    def _lease(self, temp, job_id, manifest, token):
        token = _uuid(token)
        marker_name, data_name = f"lease-{token}.json", f"pending-{token}.part"
        try:
            marker = json.loads(_read_file(temp, marker_name, 4096, self._account_read))
            if (marker.get("version") != _VERSION or marker.get("id") != token or marker.get("job_id") != job_id or
                    marker.get("workspace_token") != manifest["workspace_token"]):
                raise IntegrityError("Identidad de archivo temporal inválida")
        except (ValueError, TypeError) as error:
            raise IntegrityError("Registro temporal inválido") from error
        return marker, marker_name, data_name

    def commit_temporary(self, job_id, area, name, token):
        if area not in _AREAS:
            raise UnsafePath("Área PIZARRA inválida")
        name = _name(name)
        with self._locked() as root, self._workspace(root, job_id) as (job, manifest):
            temp = _directory(job, "temporary")
            target = _directory(job, area)
            try:
                _, marker_name, data_name = self._lease(temp, job_id, manifest, token)
                record = _read_file(temp, data_name, account=self._account_read)
                self._unpack(record)
                self._check_root(root)
                try:
                    # link() is atomic and fails rather than replacing any existing file.
                    os.link(data_name, name, src_dir_fd=temp, dst_dir_fd=target, follow_symlinks=False)
                except FileExistsError as error:
                    raise AlreadyExists("El resultado confirmado ya existe") from error
                os.fsync(target)
                os.unlink(data_name, dir_fd=temp)
                os.unlink(marker_name, dir_fd=temp)
                os.fsync(temp)
                self._check_root(root)
            finally:
                os.close(target)
                os.close(temp)
        return f"{area}/{name}"

    def commit(self, job_id, area, name, data):
        if area not in _AREAS:
            raise UnsafePath("Área PIZARRA inválida")
        _name(name)
        token = self.write_temporary(job_id, data)
        return self.commit_temporary(job_id, area, name, token)

    def read(self, job_id, reference):
        if not isinstance(reference, str):
            raise UnsafePath("Referencia inválida")
        parts = reference.split("/")
        if len(parts) != 2 or parts[0] not in _AREAS:
            raise UnsafePath("Referencia inválida")
        area, name = parts[0], _name(parts[1])
        with self._thread_lock, self._root() as root, self._workspace(root, job_id) as (job, _):
            target = _directory(job, area)
            try:
                record = _read_file(target, name, account=self._account_read)
                data = self._unpack(record)
                self._check_root(root)
            finally:
                os.close(target)
        return data

    def incomplete(self, job_id):
        results = []
        with self._thread_lock, self._root() as root, self._workspace(root, job_id) as (job, manifest):
            temp = _directory(job, "temporary")
            try:
                for name in os.listdir(temp):
                    if not (name.startswith("lease-") and name.endswith(".json")):
                        continue
                    token = name[6:-5]
                    try:
                        _, _, data_name = self._lease(temp, job_id, manifest, token)
                        self._unpack(_read_file(temp, data_name, account=self._account_read))
                        results.append({"token": token, "status": "pending"})
                    except (NotFound, IntegrityError, UnsafePath):
                        results.append({"token": token, "status": "incomplete"})
                self._check_root(root)
            finally:
                os.close(temp)
        return results

    def cleanup_abandoned(self, job_id, *, older_than_seconds):
        """Explicit cleanup of our own leased temporary files only; never results."""
        if older_than_seconds <= 0:
            raise ValueError("La antigüedad debe ser positiva")
        removed = 0
        cutoff = time.time() - older_than_seconds
        with self._locked() as root, self._workspace(root, job_id) as (job, manifest):
            temp = _directory(job, "temporary")
            try:
                for name in os.listdir(temp):
                    if not (name.startswith("lease-") and name.endswith(".json")):
                        continue
                    token = name[6:-5]
                    try:
                        marker, marker_name, data_name = self._lease(temp, job_id, manifest, token)
                        if not isinstance(marker.get("created_at"), (int, float)) or marker["created_at"] > cutoff:
                            continue
                        for filename in (data_name, marker_name):
                            try:
                                info = os.stat(filename, dir_fd=temp, follow_symlinks=False)
                                if not stat.S_ISREG(info.st_mode):
                                    continue
                                os.unlink(filename, dir_fd=temp)
                            except FileNotFoundError:
                                pass
                        removed += 1
                    except (NotFound, IntegrityError, UnsafePath):
                        continue
                os.fsync(temp)
                self._check_root(root)
            finally:
                os.close(temp)
        return removed

    def recover_manifests(self):
        """Return verified manifests only; never infer that inference can resume."""
        result = []
        with self._thread_lock, self._root() as root:
            try:
                jobs = _directory(root, "jobs")
            except UnsafePath as error:
                try:
                    os.stat("jobs", dir_fd=root, follow_symlinks=False)
                except FileNotFoundError:
                    return result
                raise error
            try:
                for name in os.listdir(jobs):
                    try:
                        with self._workspace(root, name) as (_, manifest):
                            result.append({"job_id": manifest["job_id"], "format_version": manifest["format_version"],
                                           "created_at": manifest["created_at"]})
                    except (NotFound, IntegrityError, UnsafePath):
                        continue
                self._check_root(root)
            finally:
                os.close(jobs)
        return result
