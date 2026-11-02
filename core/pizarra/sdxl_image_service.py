"""Reusable, bounded CPU SDXL experiment used by the app and probe."""
import gc
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from scripts.pizarra_unet_probe import emit
from scripts.pizarra_weight_offload_probe import cgroup_memory_path, diagnostic_memory, trim_glibc

PROMPT = 'A small red apple on a wooden table, soft daylight'
SEED = 507
BLOCK = 'down_blocks.0.resnets.0'

def memory_snapshot():
    memory = diagnostic_memory()
    stats = {key: None for key in ('anon', 'file', 'inactive_file')}
    try:
        path = cgroup_memory_path(Path('/proc/self/cgroup').read_text(),
                                  Path('/proc/self/mountinfo').read_text())
        if path is not None:
            for line in (path.parent / 'memory.stat').read_text().splitlines():
                key, value = line.split()
                if key in stats:
                    stats[key] = int(value)
    except (OSError, ValueError):
        pass
    memory['cgroup_memory_stat_bytes'] = stats
    return memory


def prepare_embeddings(pipe, emit_event=emit, prompt_text=PROMPT):
    """Encode once on CPU and drop the two SDXL encoders before denoising."""
    import torch
    emit_event('memory_before_prompt_encoding', memory=memory_snapshot())
    start = time.monotonic()
    emit_event('prompt_encoding_start')
    with torch.inference_mode():
        prompt, negative, pooled, negative_pooled = pipe.encode_prompt(
            prompt=prompt_text, device=torch.device('cpu'), num_images_per_prompt=1,
            do_classifier_free_guidance=False)
    emit_event('prompt_encoding_end', seconds=time.monotonic() - start)
    emit_event('memory_after_prompt_encoding', memory=memory_snapshot())
    if prompt is None or pooled is None:
        raise RuntimeError('Faltan embeddings SDXL del prompt')
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    gc.collect()
    trimmed = trim_glibc()
    emit_event('memory_after_encoder_release', memory=memory_snapshot(), malloc_trim=trimmed)
    return dict(prompt_embeds=prompt, negative_prompt_embeds=negative,
                pooled_prompt_embeds=pooled, negative_pooled_prompt_embeds=negative_pooled)


def generate(pipe, storage, output, size, *, emit_event=emit, before_restore=None, prompt=PROMPT, seed=SEED):
    """Run one image with a caller-owned pipeline and one persistent block session."""
    import torch
    from core.pizarra.automatic_offload import AutomaticOffload

    if size not in (64, 128):
        raise ValueError('Solo se admiten 64 o 128 píxeles')
    output = Path(output)
    if output.suffix.lower() != '.png' or not output.parent.is_dir():
        raise ValueError('La salida debe ser un PNG en un directorio existente')
    # Reserve the name atomically before creating a persistent session.
    reservation = output.open('xb')
    controller = None
    def guarded_restore(adapter):
        emit_event('memory_before_restore', memory=memory_snapshot())
        if before_restore:
            before_restore(adapter)
    original_forward = pipe.unet.forward
    original_decode = pipe.vae.decode
    forwards = 0
    restores = 0
    denoise_start = None
    decode_seen = False
    success = False
    def forward(*args, **kwargs):
        nonlocal forwards, denoise_start
        if denoise_start is None:
            denoise_start = time.monotonic()
            emit_event('denoising_start')
        forwards += 1
        return original_forward(*args, **kwargs)
    def decode(*args, **kwargs):
        nonlocal decode_seen
        decode_seen = True
        emit_event('denoising_end', seconds=time.monotonic() - denoise_start if denoise_start else None,
                   unet_forwards=forwards)
        start = time.monotonic()
        try:
            result = original_decode(*args, **kwargs)
            emit_event('vae_result', success=True, seconds=time.monotonic() - start)
            return result
        except Exception:
            emit_event('vae_result', success=False, seconds=time.monotonic() - start)
            raise
    try:
        controller = AutomaticOffload(pipe.unet, BLOCK, storage, before_restore=guarded_restore)
        pipe.unet.eval()
        pipe.unet.forward = forward
        pipe.vae.decode = decode
        with torch.inference_mode():
            session = controller.install(on_session=lambda job, token: emit_event('session', job_id=job, token=token))
            emit_event('persisted', write_bytes=session['write_bytes'], verify_read_bytes=session['verify_read_bytes'],
                       write_seconds=session['write_seconds'], verify_seconds=session['verify_seconds'])
            original_restore = controller.adapter.restore
            def counted_restore():
                nonlocal restores
                result = original_restore()
                restores += 1
                return result
            controller.adapter.restore = counted_restore
            embeddings = prepare_embeddings(pipe, emit_event, prompt)
            before = storage.stats()
            start = time.monotonic()
            result = pipe(**embeddings, height=size, width=size, num_inference_steps=1,
                          num_images_per_prompt=1, guidance_scale=1.0,
                          generator=torch.Generator(device='cpu').manual_seed(seed), output_type='pil')
            after = storage.stats()
            block = pipe.unet.get_submodule(BLOCK)
            all_meta = all(t.is_meta for t in list(block.parameters()) + list(block.buffers()))
            emit_event('generation_end', seconds=time.monotonic() - start, unet_forwards=forwards,
                       restores=restores, pizarra_read_bytes=after['read_bytes'] - before['read_bytes'],
                       additional_write_bytes=after['written_bytes'] - before['written_bytes'], all_meta=all_meta)
            if not decode_seen or forwards != 1 or restores != 1 or not all_meta or after['written_bytes'] != before['written_bytes']:
                raise RuntimeError('Generación o estado PIZARRA inesperado')
            if len(result.images) != 1 or result.images[0].size != (size, size):
                raise RuntimeError('Resultado de imagen inesperado')
            result.images[0].save(reservation, format='PNG')
            reservation.close()
            emit_event('png_saved', path=str(output.resolve()), width=size, height=size)
            guarded_restore(controller.adapter)
            removed = controller.close()
            emit_event('cleaned', records_removed=removed)
            success = True
    finally:
        reservation.close()
        if not success and controller is not None and controller.adapter.session is not None:
            if controller.adapter.state == 'released':
                try:
                    guarded_restore(controller.adapter)
                    controller.adapter.restore()
                    emit_event('failure_recovered_session_preserved')
                except Exception as error:
                    emit_event('recovery_failed_session_preserved', reason=str(error))
            emit_event('session_preserved', job_id=controller.adapter.session['job_id'],
                       token=controller.adapter.session['token'], state=controller.adapter.state)
        # On failure leave controller and its session untouched for recovery.
        if controller is None or success or controller.adapter.session is None:
            pipe.unet.forward = original_forward
        pipe.vae.decode = original_decode
        if not success:
            output.unlink(missing_ok=True)



def run_image(model, prompt, size, seed, timeout_seconds=3600):
    """Run the validated worker in its bounded scope; caller owns inference lock."""
    from PIL import Image
    from core.sdxl_progressive_loader import validate_source
    validate_source(model)
    systemd, timeout = shutil.which('systemd-run'), shutil.which('timeout')
    if not systemd or not timeout:
        raise RuntimeError('PIZARRA experimental requiere systemd-run y timeout')
    unit = 'pizarra-unet-' + uuid.uuid4().hex + '.scope'
    script = Path(__file__).resolve().parents[2] / 'scripts/pizarra_sdxl_image_probe.py'
    with tempfile.TemporaryDirectory(prefix='buttervision-pizarra-') as directory:
        output = Path(directory) / 'image.png'
        command = [systemd, '--user', '--scope', '--collect', '--quiet', '--unit', unit,
                   '--property=MemoryMax=10G', '--property=MemorySwapMax=0', timeout,
                   '--signal=TERM', '--kill-after=30s', f'{timeout_seconds}s', sys.executable,
                   str(script), '--execute', '--worker', '--unit', unit, '--model', str(model),
                   '--output', str(output), '--size', str(size), '--timeout-seconds', str(timeout_seconds),
                   '--request-stdin']
        result = subprocess.run(command, input=json.dumps({'prompt': prompt, 'seed': seed}),
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode:
            for line in result.stdout.splitlines():
                try:
                    event = json.loads(line)
                    if event.get('event') in ('session', 'session_preserved'):
                        logging.getLogger(__name__).warning("Sesión PIZARRA preservada para recuperación: job_id=%s token=%s estado=%s", event.get('job_id'), event.get('token'), event.get('state'))
                except (ValueError, AttributeError):
                    pass
            reason = ''
            for line in result.stderr.splitlines():
                try:
                    event = json.loads(line)
                    if event.get('event') == 'failed':
                        reason = str(event.get('reason', ''))
                except (ValueError, AttributeError):
                    pass
            raise RuntimeError('PIZARRA experimental falló: ' + (reason or 'el proceso aislado terminó sin generar la imagen; revisa systemd y el almacenamiento PIZARRA.'))
        with Image.open(output) as image:
            return image.copy()
