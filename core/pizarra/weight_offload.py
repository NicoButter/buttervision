"""Experimental, CPU-only ownership transfer for a checkpoint UNet block.

The caller must own the tensors exclusively; clearing a dictionary cannot remove
other aliases. The disk copy is kept on any failure after release.
"""
import gc
import hashlib
import json
import resource
import time
import uuid

import numpy as np
import psutil
import torch

from .storage import IntegrityError
from .tensor_store import TensorStore


def memory_snapshot():
    try:
        with open('/sys/fs/cgroup/memory.current', encoding='ascii') as f:
            cgroup = int(f.read())
    except (OSError, ValueError):
        cgroup = None
    return {'rss_bytes': psutil.Process().memory_info().rss,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'cgroup_current_bytes': cgroup}


def load_unet_block(path, prefix):
    """Load only a whole LDM UNet input block, without modifying the checkpoint."""
    import re
    from safetensors import safe_open
    if not re.fullmatch(r'model\.diffusion_model\.input_blocks\.\d+\.', prefix):
        raise ValueError('Prefijo de bloque UNet inválido')
    with safe_open(str(path), framework='pt', device='cpu') as f:
        keys = sorted(k for k in f.keys() if k.startswith(prefix))
        if not keys:
            raise ValueError('Bloque UNet inexistente')
        return {key: f.get_tensor(key).clone() for key in keys}


def _digest(array):
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


class OffloadFailure(RuntimeError):
    def __init__(self, job_id, token):
        super().__init__(f"Offload incompleto; sesión {job_id}, token {token}")
        self.job_id = job_id
        self.token = token


class WeightOffload:
    def __init__(self, storage, *, policy=None):
        self.storage = storage
        self.store = TensorStore(storage, policy)

    def offload(self, weights, *, on_session=None):
        """Verify on-disk bytes before clearing the exclusively-owned mapping."""
        if not weights or not isinstance(weights, dict):
            raise ValueError('Se requiere un diccionario de tensores no vacío')
        for name, tensor in weights.items():
            if (not isinstance(name, str) or not isinstance(tensor, torch.Tensor) or
                    tensor.device.type != 'cpu' or tensor.dtype not in (torch.float16, torch.float32) or
                    not tensor.is_contiguous() or tensor.numel() == 0 or tensor.ndim == 0):
                raise ValueError('Tensor incompatible con TensorStore')
        del tensor
        job, token = str(uuid.uuid4()), str(uuid.uuid4())
        ids = {name: uuid.uuid4().hex for name in weights}
        self.storage.create_workspace(job, offload_token=token, offload_tensor_ids=list(ids.values()))
        if on_session is not None:
            on_session(job, token)
        before = memory_snapshot()
        start_stats = self.storage.stats()
        start = time.monotonic()
        index = {}
        try:
            for name in sorted(weights):
                array = weights[name].detach().numpy()
                flat = array.reshape(-1)
                count = self.store.policy.max_fragment_bytes // array.itemsize
                ref = self.store.write_tensor(job, array.shape, str(array.dtype),
                    (flat[i:i + count] for i in range(0, flat.size, count)), tensor_id=ids[name])
                index[name] = {'ref': ref.reference, 'shape': list(array.shape),
                               'dtype': str(array.dtype), 'sha256': _digest(array), 'bytes': array.nbytes}
                del array, flat
            payload = json.dumps(index, sort_keys=True, separators=(',', ':')).encode()
            self.storage.commit(job, 'tensors', 'offload-index.json', payload)
            write_seconds = time.monotonic() - start
            after_write = self.storage.stats()
            verify_start = time.monotonic()
            self._verify(job, index)
            verify_seconds = time.monotonic() - verify_start
            after_verify = self.storage.stats()
        except Exception as error:
            # Originals remain owned by the caller; expose credentials if cleanup fails.
            try:
                self.storage.cleanup_offload(job, token)
            except Exception:
                raise OffloadFailure(job, token) from error
            raise
        weights.clear()
        gc.collect()
        return {'job_id': job, 'token': token, 'index': index, 'before': before,
                'after_release': memory_snapshot(), 'logical_released_bytes': sum(x['bytes'] for x in index.values()),
                'write_seconds': write_seconds, 'verify_seconds': verify_seconds,
                'write_bytes': after_write['written_bytes'] - start_stats['written_bytes'],
                'verify_read_bytes': after_verify['read_bytes'] - after_write['read_bytes']}

    def _read_index(self, job):
        try:
            return json.loads(self.storage.read(job, 'tensors/offload-index.json'))
        except (ValueError, TypeError) as error:
            raise IntegrityError('Índice de offload inválido') from error

    def _verify(self, job, index):
        if self._read_index(job) != index:
            raise IntegrityError('Índice de offload alterado')
        for item in index.values():
            ref = self.store.open_tensor(job, item['ref'])
            if list(ref.shape) != item['shape'] or ref.dtype != item['dtype']:
                raise IntegrityError('Metadatos de tensor distintos')
            digest = hashlib.sha256()
            size = 0
            for block in self.store.read_blocks(ref):
                digest.update(memoryview(block).cast('B'))
                size += block.nbytes
            if digest.hexdigest() != item['sha256'] or size != item['bytes']:
                raise IntegrityError('Contenido de tensor distinto')

    def restore(self, session):
        job, index = session['job_id'], session['index']
        self._verify(job, index)
        stats = self.storage.stats()
        started = time.monotonic()
        result = {}
        for name, item in index.items():
            ref = self.store.open_tensor(job, item['ref'])
            array = np.empty(ref.shape, dtype=ref.dtype)
            flat = array.reshape(-1)
            offset = 0
            for block in self.store.read_blocks(ref):
                flat[offset:offset + block.size] = block
                offset += block.size
            if _digest(array) != item['sha256']:
                raise IntegrityError('Recuperación de tensor distinta')
            result[name] = torch.from_numpy(array)
        return result, {'read_seconds': time.monotonic() - started,
                        'read_bytes': self.storage.stats()['read_bytes'] - stats['read_bytes'],
                        'after_restore': memory_snapshot()}
