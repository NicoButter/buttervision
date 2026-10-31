"""Versioned, checksummed tensor blocks persisted through PIZARRA storage.

The store never assembles a whole tensor. Each immutable block is an existing
PIZARRA record, so the storage layer retains its mount, UUID, quota, checksum,
symlink and atomic-publication checks.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import re
import threading
import uuid

import numpy as np

from .storage import IntegrityError, NotFound, PizarraStorage, PizarraStorageError

FORMAT_VERSION = 1
_DTYPES = {"float32": np.dtype("<f4"), "float16": np.dtype("<f2")}
_TENSOR_ID = re.compile(r"[0-9a-f]{32}\Z")


@dataclass(frozen=True)
class BufferPolicy:
    """Operational executor buffer budget, independent of process RSS."""

    max_fragment_bytes: int = 8 * 1024 * 1024
    max_simultaneous_buffers: int = 6
    result_partial_bytes: int = 8 * 1024 * 1024
    safety_margin_bytes: int = 8 * 1024 * 1024
    total_budget_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        values = (self.max_fragment_bytes, self.max_simultaneous_buffers,
                  self.result_partial_bytes, self.safety_margin_bytes, self.total_budget_bytes)
        if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in values):
            raise ValueError("La política de buffers requiere límites enteros positivos")
        if self.reserved_bytes > self.total_budget_bytes:
            raise ValueError("Los buffers configurados superan el presupuesto total")

    @property
    def reserved_bytes(self):
        return (self.max_fragment_bytes * self.max_simultaneous_buffers +
                self.result_partial_bytes + self.safety_margin_bytes)

    @classmethod
    def from_budget(cls, total_budget_bytes, *, max_simultaneous_buffers=6,
                    result_partial_bytes=8 * 1024 * 1024, safety_margin_bytes=8 * 1024 * 1024):
        if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0
               for x in (total_budget_bytes, max_simultaneous_buffers)):
            raise ValueError("El presupuesto y el número de buffers deben ser positivos")
        fragment = (total_budget_bytes - result_partial_bytes - safety_margin_bytes) // max_simultaneous_buffers
        return cls(fragment, max_simultaneous_buffers, result_partial_bytes,
                   safety_margin_bytes, total_budget_bytes)


class BufferLimiter:
    """Synchronous backpressure and accounting for executor-owned buffers."""

    def __init__(self, policy):
        self.policy = policy
        self._condition = threading.Condition()
        self._bytes = 0
        self._count = 0
        self._partial_bytes = 0
        self.peak_bytes = 0
        self.peak_count = 0
        self.peak_partial_bytes = 0

    @contextmanager
    def reserve(self, size):
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("Tamaño de buffer inválido")
        if size > self.policy.max_fragment_bytes:
            raise MemoryError("El fragmento excede el límite configurado")
        if size == 0:
            yield
            return
        with self._condition:
            while (self._count >= self.policy.max_simultaneous_buffers or
                   self._bytes + size > self.policy.max_fragment_bytes * self.policy.max_simultaneous_buffers):
                self._condition.wait()
            self._count += 1
            self._bytes += size
            self.peak_count = max(self.peak_count, self._count)
            self.peak_bytes = max(self.peak_bytes, self._bytes)
        try:
            yield
        finally:
            with self._condition:
                self._count -= 1
                self._bytes -= size
                self._condition.notify_all()

    @contextmanager
    def reserve_partial(self, size):
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("Tamaño de resultado parcial inválido")
        if size > self.policy.result_partial_bytes:
            raise MemoryError("El buffer de resultado supera result_partial_bytes")
        if size == 0:
            yield
            return
        with self._condition:
            while self._partial_bytes + size > self.policy.result_partial_bytes:
                self._condition.wait()
            self._partial_bytes += size
            self.peak_partial_bytes = max(self.peak_partial_bytes, self._partial_bytes)
        try:
            yield
        finally:
            with self._condition:
                self._partial_bytes -= size
                self._condition.notify_all()

    def snapshot(self):
        return {"managed_current_bytes": self._bytes, "managed_peak_bytes": self.peak_bytes,
                "managed_current_buffers": self._count, "managed_peak_buffers": self.peak_count,
                "managed_partial_current_bytes": self._partial_bytes,
                "managed_partial_peak_bytes": self.peak_partial_bytes,
                "max_fragment_bytes": self.policy.max_fragment_bytes,
                "max_simultaneous_buffers": self.policy.max_simultaneous_buffers,
                "result_partial_bytes": self.policy.result_partial_bytes,
                "safety_margin_bytes": self.policy.safety_margin_bytes,
                "configured_budget_bytes": self.policy.total_budget_bytes,
                "configured_reserved_bytes": self.policy.reserved_bytes}


@dataclass(frozen=True)
class TensorRef:
    job_id: str
    tensor_id: str
    shape: tuple
    dtype: str
    order: str
    block_elements: int
    block_count: int
    version: int = FORMAT_VERSION

    @property
    def reference(self):
        return f"tensor:{self.tensor_id}"


class TensorStore:
    def __init__(self, storage: PizarraStorage, policy=None):
        if not isinstance(storage, PizarraStorage) and not all(hasattr(storage, name) for name in ("commit", "read")):
            raise TypeError("Se requiere un almacenamiento PIZARRA compatible")
        self.storage = storage
        self.policy = policy or BufferPolicy()

    @staticmethod
    def _shape(shape):
        if not isinstance(shape, (tuple, list)) or not shape or len(shape) > 8:
            raise ValueError("shape debe tener entre 1 y 8 dimensiones")
        result = []
        for dimension in shape:
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
                raise ValueError("Las dimensiones deben ser enteros positivos")
            result.append(dimension)
        return tuple(result)

    def _block_elements(self, dtype, block_elements):
        itemsize = _DTYPES[dtype].itemsize
        max_elements = min(self.policy.max_fragment_bytes, self.policy.result_partial_bytes) // itemsize
        if max_elements < 1:
            raise ValueError("El fragmento configurado es menor que un elemento")
        if block_elements is None:
            return max_elements
        if isinstance(block_elements, bool) or not isinstance(block_elements, int) or not 1 <= block_elements <= max_elements:
            raise ValueError("block_elements supera el presupuesto de fragmento")
        return block_elements

    def _manifest_name(self, tensor_id):
        return f"tensor-{tensor_id}-manifest.json"

    def _block_name(self, tensor_id, index):
        return f"tensor-{tensor_id}-block-{index:08d}.bin"

    def write_tensor(self, job_id, shape, dtype, blocks, *, order="C", block_elements=None,
                     tensor_id=None, limiter=None):
        """Persist consecutive flat blocks from an iterable; input blocks are bounded."""
        shape = self._shape(shape)
        if dtype not in _DTYPES:
            raise ValueError("dtype debe ser float32 o float16")
        if order != "C":
            raise ValueError("Solo se admite orden C")
        block_elements = self._block_elements(dtype, block_elements)
        limiter = limiter or BufferLimiter(self.policy)
        if limiter.policy.max_simultaneous_buffers < 2:
            raise MemoryError("La escritura requiere al menos dos buffers simultáneos")
        total = math.prod(shape)
        block_count = math.ceil(total / block_elements)
        tensor_id = tensor_id or uuid.uuid4().hex
        if not isinstance(tensor_id, str) or not _TENSOR_ID.fullmatch(tensor_id):
            raise ValueError("tensor_id debe ser un identificador hexadecimal de 32 caracteres")
        ref = TensorRef(str(job_id), tensor_id, shape, dtype, order, block_elements, block_count)
        manifest = {"format_version": FORMAT_VERSION, "tensor_id": tensor_id, "shape": list(shape),
                    "dtype": dtype, "order": order, "block_elements": block_elements,
                    "block_count": block_count, "element_count": total}
        self.storage.commit(ref.job_id, "tensors", self._manifest_name(tensor_id), _json_bytes(manifest))

        element_bytes = _DTYPES[dtype].itemsize
        pending = bytearray()
        pending_lease = None
        block_index = 0
        written_elements = 0
        try:
            for source in blocks:
                source_array = None if isinstance(source, (bytes, bytearray, memoryview)) else np.asarray(source)
                source_size = len(source) if source_array is None else source_array.nbytes
                if source_size > self.policy.max_fragment_bytes:
                    raise ValueError("Cada bloque de entrada debe respetar max_fragment_bytes")
                with limiter.reserve(source_size):
                    raw = self._source_bytes(source if source_array is None else source_array, dtype)
                    if len(raw) % element_bytes:
                        raise ValueError("El bloque no contiene un número entero de elementos")
                    if written_elements + len(raw) // element_bytes > total:
                        raise ValueError("Los bloques exceden el tamaño declarado")
                    cursor = 0
                    while cursor < len(raw):
                        if not pending:
                            pending_lease = limiter.reserve_partial(block_elements * element_bytes)
                            pending_lease.__enter__()
                        count = min(block_elements * element_bytes - len(pending), len(raw) - cursor)
                        pending.extend(raw[cursor:cursor + count])
                        cursor += count
                        if len(pending) == block_elements * element_bytes:
                            self._commit_block(ref, block_index, pending, limiter=limiter)
                            written_elements += block_elements
                            block_index += 1
                            pending.clear()
                            pending_lease.__exit__(None, None, None)
                            pending_lease = None
            if pending:
                if written_elements + len(pending) // element_bytes != total:
                    raise ValueError("Los bloques no completan las dimensiones declaradas")
                self._commit_block(ref, block_index, pending, limiter=limiter)
                written_elements += len(pending) // element_bytes
                block_index += 1
                pending.clear()
                pending_lease.__exit__(None, None, None)
                pending_lease = None
        finally:
            if pending_lease is not None:
                pending_lease.__exit__(None, None, None)
        if written_elements != total or block_index != block_count:
            raise ValueError("Los bloques no completan las dimensiones declaradas")
        return ref

    def write_block(self, ref, index, block):
        """Atomically commit one missing block, useful for explicit recovery."""
        ref = self.open_tensor(ref.job_id, ref.reference)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < ref.block_count:
            raise ValueError("Índice de bloque inválido")
        raw = self._source_bytes(block, ref.dtype)
        dtype = _DTYPES[ref.dtype]
        expected_elements = min(ref.block_elements,
                                math.prod(ref.shape) - index * ref.block_elements)
        if len(raw) != expected_elements * dtype.itemsize:
            raise ValueError("El bloque no coincide con el tamaño esperado")
        return self._commit_block(ref, index, raw)

    def _commit_block(self, ref, index, raw, *, limiter=None):
        reservation = limiter.reserve(len(raw)) if limiter else _null_context()
        with reservation:
            return self.storage.commit(ref.job_id, "tensors", self._block_name(ref.tensor_id, index), bytes(raw))

    def missing_blocks(self, ref):
        """List absent blocks; corrupt committed blocks raise IntegrityError."""
        ref = self.open_tensor(ref.job_id, ref.reference)
        missing = []
        for index in range(ref.block_count):
            try:
                raw = self.storage.read(ref.job_id, f"tensors/{self._block_name(ref.tensor_id, index)}")
            except NotFound:
                missing.append(index)
                continue
            expected = min(ref.block_elements, math.prod(ref.shape) - index * ref.block_elements)
            if len(raw) != expected * _DTYPES[ref.dtype].itemsize:
                raise IntegrityError(f"Tamaño inválido en el bloque {index}")
        return missing

    def _source_bytes(self, source, dtype):
        expected = _DTYPES[dtype]
        if isinstance(source, (bytes, bytearray, memoryview)):
            if len(source) > self.policy.max_fragment_bytes:
                raise ValueError("Cada bloque de entrada debe respetar max_fragment_bytes")
            return bytes(source)
        array = np.asarray(source)
        if array.dtype != expected:
            raise ValueError(f"El bloque debe tener dtype exacto {dtype}")
        if array.nbytes > self.policy.max_fragment_bytes:
            raise ValueError("Cada bloque de entrada debe respetar max_fragment_bytes")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        return array.tobytes(order="C")

    def open_tensor(self, job_id, reference):
        if not isinstance(reference, str) or not reference.startswith("tensor:"):
            raise ValueError("Referencia de tensor inválida")
        tensor_id = reference[7:]
        if not _TENSOR_ID.fullmatch(tensor_id):
            raise ValueError("Referencia de tensor inválida")
        try:
            import json
            manifest = json.loads(self.storage.read(str(job_id), f"tensors/{self._manifest_name(tensor_id)}"))
            shape = self._shape(manifest["shape"])
            dtype = manifest["dtype"]
            if (manifest.get("format_version") != FORMAT_VERSION or manifest.get("tensor_id") != tensor_id or
                    dtype not in _DTYPES or manifest.get("order") != "C" or
                    manifest.get("element_count") != math.prod(shape)):
                raise ValueError("Manifiesto de tensor inválido")
            elements = self._block_elements(dtype, manifest["block_elements"])
            count = math.ceil(math.prod(shape) / elements)
            if manifest.get("block_count") != count:
                raise ValueError("Cantidad de bloques inválida")
            return TensorRef(str(job_id), tensor_id, shape, dtype, "C", elements, count)
        except PizarraStorageError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise IntegrityError("Manifiesto de tensor inválido") from error

    def read_blocks(self, ref, *, limiter=None):
        ref = self.open_tensor(ref.job_id, ref.reference)
        dtype = _DTYPES[ref.dtype]
        total = math.prod(ref.shape)
        for index in range(ref.block_count):
            expected_elements = min(ref.block_elements, total - index * ref.block_elements)
            reservation = limiter.reserve(expected_elements * dtype.itemsize) if limiter else _null_context()
            with reservation:
                try:
                    raw = self.storage.read(ref.job_id, f"tensors/{self._block_name(ref.tensor_id, index)}")
                except NotFound as error:
                    raise IntegrityError(f"Falta el bloque confirmado {index}") from error
                if len(raw) != expected_elements * dtype.itemsize:
                    raise IntegrityError(f"Tamaño inválido en el bloque {index}")
                yield np.frombuffer(raw, dtype=dtype, count=expected_elements)

    def iter_chunks(self, ref):
        yield from self.read_blocks(ref)

    def read_block(self, ref, row_start, row_stop, col_start, col_stop, *, limiter=None,
                   result_reserved=False):
        """Read a 2D C-order region without materializing the full tensor."""
        ref = self.open_tensor(ref.job_id, ref.reference)
        if len(ref.shape) != 2:
            raise ValueError("El acceso matricial requiere un tensor 2D")
        rows, cols = ref.shape
        indices = (row_start, row_stop, col_start, col_stop)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in indices):
            raise ValueError("Los rangos deben ser enteros")
        if not (0 <= row_start < row_stop <= rows and 0 <= col_start < col_stop <= cols):
            raise ValueError("Rango de bloque fuera de dimensiones")
        dtype = _DTYPES[ref.dtype]
        output_bytes = (row_stop - row_start) * (col_stop - col_start) * dtype.itemsize
        if output_bytes > self.policy.max_fragment_bytes:
            raise MemoryError("El bloque 2D excede max_fragment_bytes")
        if result_reserved and limiter is None:
            raise ValueError("result_reserved requiere BufferLimiter")
        context = (limiter.reserve(output_bytes) if limiter and not result_reserved
                   else _null_context())
        with context:
            result = np.empty((row_stop - row_start, col_stop - col_start), dtype=dtype)
            for out_row, row in enumerate(range(row_start, row_stop)):
                col = col_start
                while col < col_stop:
                    flat_index = row * cols + col
                    block_index = flat_index // ref.block_elements
                    offset = flat_index % ref.block_elements
                    count = min(col_stop - col, ref.block_elements - offset)
                    try:
                        block_bytes = min(ref.block_elements * dtype.itemsize,
                                          math.prod(ref.shape) * dtype.itemsize)
                        with limiter.reserve(block_bytes) if limiter else _null_context():
                            raw = self.storage.read(ref.job_id, f"tensors/{self._block_name(ref.tensor_id, block_index)}")
                            expected = min(ref.block_elements,
                                           math.prod(ref.shape) - block_index * ref.block_elements)
                            if len(raw) != expected * dtype.itemsize:
                                raise IntegrityError(f"Tamaño inválido en el bloque {block_index}")
                            block = np.frombuffer(raw, dtype=dtype)
                            result[out_row, col - col_start:col - col_start + count] = block[offset:offset + count]
                    except NotFound as error:
                        raise IntegrityError(f"Falta el bloque confirmado {block_index}") from error
                    col += count
            return result


def _json_bytes(value):
    import json
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


@contextmanager
def _null_context():
    yield
