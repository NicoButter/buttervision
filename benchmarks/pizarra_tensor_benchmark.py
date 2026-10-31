#!/usr/bin/env python3
"""Reproducible PIZARRA tensor benchmark; writes only with explicit --execute."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import uuid

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.pizarra.config import PizarraConfig
from core.pizarra.tensor_ops import ResourceSampler, reduce_sum, scale
from core.pizarra.tensor_store import BufferPolicy, TensorStore
from core.pizarra.storage import PizarraStorage


def _values(start, count):
    return np.fromiter((((index % 101) - 50) / 50.0 for index in range(start, start + count)),
                       dtype=np.float32, count=count)


def _expected_sum(element_count):
    remainder = element_count % 101
    return sum(((index - 50) / 50.0) for index in range(remainder))


def run(args):
    if not args.execute:
        raise ValueError("La escritura física requiere --execute explícito")
    if args.device != "cpu":
        raise ValueError("Este prototipo solo implementa CPU; CUDA aún no está habilitado")
    if not math.isfinite(args.size_gib) or args.size_gib <= 0:
        raise ValueError("--size-gib debe ser positivo y finito")
    if args.block_mib <= 0 or args.buffer_mib <= 0:
        raise ValueError("Los tamaños de bloque y presupuesto deben ser positivos")

    config = PizarraConfig.from_env()
    if not config.enabled:
        raise RuntimeError("PIZARRA está deshabilitado; configure su montaje autorizado para ejecutar el benchmark")
    storage = PizarraStorage(config)
    storage.validate()
    budget = int(args.buffer_mib * 1024**2)
    partial = min(8 * 1024**2, budget // 8)
    safety = min(8 * 1024**2, budget // 8)
    policy = BufferPolicy.from_budget(budget, result_partial_bytes=partial,
                                      safety_margin_bytes=safety)
    configured_chunk = int(args.block_mib * 1024**2)
    effective_fragment = min(policy.max_fragment_bytes, policy.result_partial_bytes)
    if configured_chunk > effective_fragment:
        raise ValueError(f"El bloque solicitado supera el máximo efectivo de {effective_fragment} bytes")
    elements = int(args.size_gib * 1024**3) // np.dtype("float32").itemsize
    if elements < 1:
        raise ValueError("El tamaño seleccionado es menor que un elemento float32")
    block_elements = configured_chunk // np.dtype("float32").itemsize
    logical_bytes = elements * np.dtype("float32").itemsize
    expected_blocks = math.ceil(elements / block_elements)
    estimated_overhead = expected_blocks * 8192 + 65_536
    preflight = storage.stats()
    if logical_bytes * 2 + estimated_overhead > preflight["available_bytes"]:
        raise RuntimeError("El espacio PIZARRA utilizable no alcanza para conservar entrada, salida y metadatos")
    job_id = str(uuid.uuid4())
    storage.create_workspace(job_id)
    store = TensorStore(storage, policy)
    before = storage.stats()
    base_read = before.get("read_bytes") or 0
    base_written = before.get("written_bytes") or 0
    overall = ResourceSampler()
    overall.start()

    started = time.perf_counter()
    def source_blocks():
        for offset in range(0, elements, block_elements):
            yield _values(offset, min(block_elements, elements - offset))
    input_ref = store.write_tensor(job_id, (elements,), "float32", source_blocks(),
                                   block_elements=block_elements)
    write_input_seconds = time.perf_counter() - started

    started = time.perf_counter()
    input_sum, input_metrics = reduce_sum(store, input_ref)
    read_reduce_seconds = time.perf_counter() - started
    expected = _expected_sum(elements)
    if not math.isclose(input_sum, expected, rel_tol=1e-10, abs_tol=1e-5):
        raise RuntimeError("La reducción de entrada no coincide con el patrón sintético")

    started = time.perf_counter()
    output_ref, transform_metrics = scale(store, input_ref, 1.25)
    transform_write_seconds = time.perf_counter() - started
    started = time.perf_counter()
    output_sum, output_metrics = reduce_sum(store, output_ref)
    output_read_seconds = time.perf_counter() - started
    if not math.isclose(output_sum, expected * 1.25, rel_tol=1e-10, abs_tol=1e-5):
        raise RuntimeError("El resultado transformado no coincide con el valor de referencia")

    memory = overall.finish()
    after = storage.stats()
    transferred_read = (after.get("read_bytes") or 0) - base_read
    transferred_written = (after.get("written_bytes") or 0) - base_written
    total_seconds = write_input_seconds + read_reduce_seconds + transform_write_seconds + output_read_seconds
    report = {
        "format_version": 1, "job_id": job_id, "device": "cpu",
        "tensor": {"dtype": "float32", "shape": [elements], "elements": elements,
                   "logical_bytes": logical_bytes, "input_reference": input_ref.reference,
                   "output_reference": output_ref.reference, "input_block_count": input_ref.block_count,
                   "output_block_count": output_ref.block_count},
        "storage_preflight": {"available_bytes": preflight["available_bytes"],
                               "estimated_required_bytes_including_overhead": logical_bytes * 2 + estimated_overhead},
        "policy": {"configured_budget_bytes": budget, "max_fragment_bytes": policy.max_fragment_bytes,
                   "max_simultaneous_buffers": policy.max_simultaneous_buffers,
                   "result_partial_bytes": policy.result_partial_bytes,
                   "safety_margin_bytes": policy.safety_margin_bytes,
                   "requested_block_bytes": configured_chunk, "actual_block_elements": block_elements},
        "operations_seconds": {"write_input": write_input_seconds, "read_and_reduce_input": read_reduce_seconds,
                               "transform_and_write_output": transform_write_seconds,
                               "read_and_reduce_output": output_read_seconds, "total": total_seconds},
        "throughput_bytes_per_second": {
            "write_input": elements * 4 / write_input_seconds if write_input_seconds else None,
            "read_input": elements * 4 / read_reduce_seconds if read_reduce_seconds else None,
            "transform_read_plus_write": elements * 8 / transform_write_seconds if transform_write_seconds else None,
            "read_output": elements * 4 / output_read_seconds if output_read_seconds else None,
            "logical_total": elements * 16 / total_seconds if total_seconds else None},
        "storage_transfer": {"read_bytes": transferred_read, "written_bytes": transferred_written},
        "validation": {"input_sum": input_sum, "expected_input_sum": expected,
                       "output_sum": output_sum, "expected_output_sum": expected * 1.25,
                       "passed": True},
        "memory": {**memory, "executor": "cpu", "cuda_used_by_executor": False,
                   "executor_buffers_transform": transform_metrics,
                   "executor_buffers_input_reduction": input_metrics,
                   "executor_buffers_output_reduction": output_metrics,
                   "process_rss_is_not_a_hard_limit": True},
        "pizarra_storage_after": {"managed_bytes": after.get("managed_bytes"),
                                   "quota_remaining_bytes": after.get("quota_remaining_bytes"),
                                   "free_bytes": after.get("free_bytes"),
                                   "available_bytes": after.get("available_bytes")},
        "notes": ["No se cargó el tensor completo en memoria.",
                  "El límite aplica a buffers administrados, no al RSS total del intérprete ni de sus bibliotecas.",
                  "El área PIZARRA de este benchmark se conserva para permitir inspección y repetición."]}
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.report:
        Path(args.report).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="autoriza escrituras en el PIZARRA configurado")
    parser.add_argument("--size-gib", type=float, default=0.01, help="tamaño lógico en GiB binarios (default: 0.01)")
    parser.add_argument("--block-mib", type=float, default=8, help="tamaño solicitado del bloque (default: 8 MiB)")
    parser.add_argument("--buffer-mib", type=float, default=128, help="presupuesto administrado total (default: 128 MiB)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--report", help="ruta local opcional para guardar el informe JSON")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        parser.exit(2, f"Benchmark no completado: {error}\n")


if __name__ == "__main__":
    main()
