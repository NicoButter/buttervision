"""Cross-process guard shared by API inference and the isolated SDXL probe."""
import errno
import os
from pathlib import Path

import config


def acquire_inference_lock():
    """Acquire a non-blocking local lock; return its fd or None when already held."""
    lock_path = Path(config.CACHE_DIR) / "buttervision-inference.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.chmod(lock_path, 0o600)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError as error:
                if error.errno in (errno.EACCES, errno.EDEADLK, errno.EAGAIN):
                    os.close(fd)
                    return None
                raise
        else:
            import fcntl
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in (errno.EACCES, errno.EAGAIN):
                    os.close(fd)
                    return None
                raise
        return fd
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def release_inference_lock(fd):
    if fd is None:
        return
    try:
        if os.name == "nt":
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
