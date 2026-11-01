"""Opt-in, systemd-bounded progressive SDXL UNet and PIZARRA experiment."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pizarra_weight_offload_probe import cgroup_memory_path, diagnostic_memory, trim_glibc


def snapshot():
    import torch
    io = None
    try:
        io = {k: int(v) for k, v in (line.split(':', 1) for line in Path('/proc/self/io').read_text().splitlines())}
    except (OSError, ValueError):
        pass
    vram = None
    if torch.cuda.is_available():
        vram = {'allocated_bytes': torch.cuda.memory_allocated(),
                'reserved_bytes': torch.cuda.memory_reserved(),
                'free_total_bytes': torch.cuda.mem_get_info()}
    return {'memory': diagnostic_memory(), 'process_io': io, 'vram': vram}


def emit(event, **values):
    print(json.dumps({'event': event, 'snapshot': snapshot(), **values}), flush=True)


def check_scope(unit, limit_bytes):
    """Refuse workers outside their named cgroup-v2 scope with bounded RAM and no swap."""
    if not unit.startswith('pizarra-unet-') or not unit.endswith('.scope') or '/' in unit:
        raise RuntimeError('Nombre de scope inválido')
    groups = Path('/proc/self/cgroup').read_text()
    mounts = Path('/proc/self/mountinfo').read_text()
    group = next((line.split(':', 2)[2] for line in groups.splitlines() if line.startswith('0::')), None)
    memory_file = cgroup_memory_path(groups, mounts)
    if not group or group.rstrip('/').split('/')[-1] != unit or memory_file is None:
        raise RuntimeError('Worker fuera del scope systemd cgroup v2 requerido')
    directory = memory_file.parent
    maximum = (directory / 'memory.max').read_text().strip()
    swap = (directory / 'memory.swap.max').read_text().strip()
    if maximum == 'max' or int(maximum) > limit_bytes or int(maximum) <= 0 or swap != '0':
        raise RuntimeError('El scope no tiene MemoryMax solicitado y swap deshabilitado')
    return {'unit': unit, 'memory_max_bytes': int(maximum), 'swap_max_bytes': int(swap)}


def inputs(unet, block):
    """Deterministic synthetic SDXL conditioning and first down-block ResNet inputs."""
    import torch
    dtype = next(block.parameters()).dtype
    generator = torch.Generator(device='cpu').manual_seed(507)
    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=dtype)
    x = rand(1, block.in_channels, 8, 8)
    temb = rand(1, block.time_emb_proj.in_features)
    config = unet.config
    if config.addition_embed_type != 'text_time' or not isinstance(config.cross_attention_dim, int):
        raise ValueError('La configuración UNet no corresponde al condicionamiento SDXL text_time esperado')
    time_dim = config.addition_time_embed_dim
    projected = config.projection_class_embeddings_input_dim
    text_dim = projected - 6 * time_dim
    if text_dim <= 0:
        raise ValueError('Dimensiones de condicionamiento SDXL incompatibles')
    full = {'sample': rand(1, config.in_channels, 8, 8),
            'timestep': torch.tensor(1),
            'encoder_hidden_states': rand(1, 2, config.cross_attention_dim),
            'added_cond_kwargs': {'text_embeds': rand(1, text_dim),
                                  'time_ids': torch.tensor([[8, 8, 0, 0, 8, 8]], dtype=dtype)}}
    return (x, temb), full


def execute_experiment(unet, storage, *, full_forward=False, malloc_trim=False, emit_event=emit):
    """Run on a caller-owned UNet. Only CLI workers may use a production checkpoint."""
    import torch
    from core.pizarra.module_offload import ModuleOffload
    path = 'down_blocks.0.resnets.0'
    block = unet.get_submodule(path)
    adapter = ModuleOffload(unet, path, storage)
    block_inputs, full_inputs = inputs(unet, block)
    unet.eval()
    def run(label, fn):
        before = time.monotonic()
        with torch.inference_mode():
            output = fn()
        emit_event(label, seconds=time.monotonic() - before)
        return output
    def compare(label, original, restored):
        difference = (original.float() - restored.float()).abs()
        exact = bool(torch.equal(original, restored))
        emit_event(label, exact=exact, max_abs_error=float(difference.max()),
                   mean_abs_error=float(difference.mean()))
        if not exact:
            raise RuntimeError(label + ': salida distinta')
    baseline = run('block_before', lambda: block(*block_inputs))
    full_baseline = (run('unet_before', lambda: unet(**full_inputs).sample)
                     if full_forward else None)
    success = False
    try:
        session = adapter.persist(on_session=lambda job, token: emit_event('session', job_id=job, token=token))
        emit_event('persisted', metadata=adapter.metadata,
                   pizarra_write_bytes=session['write_bytes'], pizarra_verify_read_bytes=session['verify_read_bytes'],
                   write_seconds=session['write_seconds'], verify_seconds=session['verify_seconds'])
        start_stats = storage.stats()
        released = adapter.release()
        gc.collect()
        if malloc_trim:
            released['malloc_trim'] = trim_glibc()
        emit_event('released', same_module=unet.get_submodule(path) is block,
                   all_meta=all(t.is_meta for t in list(block.parameters()) + list(block.buffers())),
                   pizarra_read_bytes=storage.stats()['read_bytes'] - start_stats['read_bytes'], **released)
        if unet.get_submodule(path) is not block:
            raise RuntimeError('Cambió la identidad del submódulo')
        if not all(t.is_meta for t in list(block.parameters()) + list(block.buffers())):
            raise RuntimeError('No se liberaron todos los tensores del submódulo')
        start_stats = storage.stats()
        before = time.monotonic()
        restored_metrics = adapter.restore()
        emit_event('restored', same_module=unet.get_submodule(path) is block,
                   total_seconds=time.monotonic() - before,
                   pizarra_total_read_bytes=storage.stats()['read_bytes'] - start_stats['read_bytes'],
                   **restored_metrics)
        if unet.get_submodule(path) is not block:
            raise RuntimeError('Cambió la identidad del submódulo')
        compare('block_equivalence', baseline, run('block_after', lambda: block(*block_inputs)))
        if full_forward:
            compare('unet_equivalence', full_baseline, run('unet_after', lambda: unet(**full_inputs).sample))
        success = True
    finally:
        if adapter.session is not None:
            if adapter.state == 'released':
                try:
                    adapter.restore()
                    emit_event('exception_recovery_restored')
                except Exception as error:
                    emit_event('recovery_failed_session_preserved', error_type=type(error).__name__, reason=str(error))
            if success and adapter.state == 'resident':
                emit_event('cleaned', records_removed=adapter.cleanup())
            elif not success:
                emit_event('session_preserved', job_id=adapter.session['job_id'], token=adapter.session['token'])


def execute_automatic(unet, storage, *, malloc_trim=False, emit_event=emit, before_restore=None):
    """Two synthetic full UNet forwards; preserve the session on failure."""
    import torch
    from core.pizarra.automatic_offload import AutomaticOffload
    path = 'down_blocks.0.resnets.0'
    block = unet.get_submodule(path)
    _, full_inputs = inputs(unet, block)
    unet.eval()
    with torch.inference_mode():
        start = time.monotonic()
        expected = unet(**full_inputs).sample
        emit_event('baseline', seconds=time.monotonic() - start)
        controller = AutomaticOffload(unet, path, storage, before_restore=before_restore)
        success = False
        try:
            session = controller.install(on_session=lambda job, token: emit_event('session', job_id=job, token=token))
            emit_event('persisted', write_bytes=session['write_bytes'], write_seconds=session['write_seconds'])
            if malloc_trim:
                trim_glibc()
            for step in range(2):
                stats = storage.stats()
                start = time.monotonic()
                actual = unet(**full_inputs).sample
                elapsed = time.monotonic() - start
                after = storage.stats()
                exact = bool(torch.equal(expected, actual))
                all_meta = all(t.is_meta for t in list(block.parameters()) + list(block.buffers()))
                emit_event('automatic_forward', step=step, seconds=elapsed, exact=exact,
                           all_meta=all_meta, read_bytes=after['read_bytes'] - stats['read_bytes'],
                           write_bytes=after['written_bytes'] - stats['written_bytes'])
                if not exact or not all_meta or after['written_bytes'] != stats['written_bytes']:
                    raise RuntimeError('Equivalencia, estado meta o persistencia inválidos')
                del actual
            success = True
        finally:
            if success:
                emit_event('cleaned', records_removed=controller.close())
            elif controller.adapter.session is not None:
                emit_event('session_preserved', job_id=controller.adapter.session['job_id'],
                           token=controller.adapter.session['token'])


def check_restore_budget(adapter):
    """Conservative headroom for weights, forward temporaries and Fedora."""
    memory_file = cgroup_memory_path(Path('/proc/self/cgroup').read_text(), Path('/proc/self/mountinfo').read_text())
    if memory_file is None:
        raise RuntimeError('No se puede verificar presupuesto cgroup')
    maximum = (memory_file.parent / 'memory.max').read_text().strip()
    if maximum == 'max':
        raise RuntimeError('Límite cgroup no disponible')
    weights = sum(item['bytes'] for item in adapter.session['index'].values())
    available = int(maximum) - int(memory_file.read_text())
    required = weights * 2 + 512 * 1024**2
    emit('restore_budget', available_bytes=available, required_bytes=required)
    if available < required:
        raise RuntimeError('Presupuesto insuficiente; sesión conservada')

def worker(args):
    import torch
    from core.sdxl_pipeline import local_sdxl_config
    from core.sdxl_progressive_loader import load_progressive, validate_source
    from core.pizarra.config import PizarraConfig
    from core.pizarra.storage import PizarraStorage
    scope = check_scope(args.unit, args.memory_max_gib * 1024**3)
    emit('scope_verified', scope=scope)
    path = validate_source(args.model)
    storage = PizarraStorage(PizarraConfig.from_env())
    if not storage.validate().get('available'):
        raise RuntimeError('Volumen PIZARRA no disponible')
    emit('before_load')
    pipe = load_progressive(path, local_sdxl_config(path), dtype=torch.float16,
                            report=lambda record: emit('progressive_load', progress=record))
    emit('loaded', load_metrics=pipe.progressive_load_metrics)
    if args.automatic:
        execute_automatic(pipe.unet, storage, malloc_trim=args.malloc_trim, before_restore=check_restore_budget)
    else:
        execute_experiment(pipe.unet, storage, full_forward=args.full_forward, malloc_trim=args.malloc_trim)
    emit('success', full_forward=args.full_forward)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--model')
    p.add_argument('--memory-max-gib', type=int, default=10)
    p.add_argument('--timeout-seconds', type=int, default=3600)
    p.add_argument('--full-forward', action='store_true')
    p.add_argument('--automatic', action='store_true')
    p.add_argument('--malloc-trim', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--unit', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not args.execute:
        p.error('requiere --execute')
    if not args.model or not 1 <= args.memory_max_gib <= 1024 or not 60 <= args.timeout_seconds <= 7200:
        p.error('modelo, límite de memoria y timeout válidos requeridos')
    if args.automatic and args.memory_max_gib > 10:
        p.error('El probe automático no permite más de 10 GiB')
    if args.worker:
        try:
            worker(args)
            return 0
        except Exception as error:
            print(json.dumps({'event': 'failed', 'error_type': type(error).__name__, 'reason': str(error)}), file=sys.stderr, flush=True)
            return 1
    systemd, timeout = shutil.which('systemd-run'), shutil.which('timeout')
    if not systemd or not timeout:
        raise SystemExit('systemd-run y timeout son obligatorios')
    from core.inference_lock import acquire_inference_lock, release_inference_lock
    unit = 'pizarra-unet-' + uuid.uuid4().hex + '.scope'
    command = [systemd, '--user', '--scope', '--collect', '--quiet', '--unit', unit,
               f'--property=MemoryMax={args.memory_max_gib}G', '--property=MemorySwapMax=0',
               timeout, '--signal=TERM', '--kill-after=30s', f'{args.timeout_seconds}s',
               sys.executable, str(Path(__file__).resolve()), '--execute', '--worker', '--unit', unit,
               '--model', args.model, '--memory-max-gib', str(args.memory_max_gib),
               '--timeout-seconds', str(args.timeout_seconds)]
    if args.full_forward:
        command.append('--full-forward')
    if args.automatic:
        command.append('--automatic')
    if args.malloc_trim:
        command.append('--malloc-trim')
    lock = acquire_inference_lock()
    if lock is None:
        raise SystemExit('Ya hay una inferencia o carga activa')
    try:
        result = subprocess.run(command, check=False)
        if result.returncode:
            print(json.dumps({'event': 'supervisor_failed', 'return_code': result.returncode,
                              'reason': 'Revisar eventos, journal systemd y posible límite de memoria o timeout'}), file=sys.stderr)
        return result.returncode
    finally:
        release_inference_lock(lock)


if __name__ == '__main__':
    raise SystemExit(main())
