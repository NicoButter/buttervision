"""Opt-in SDXL checkpoint ResNet block execution probe; run in a bounded scope."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pizarra_weight_offload_probe import diagnostic_memory, trim_glibc


def snapshot():
    import torch
    try:
        io = {}
        for line in Path('/proc/self/io').read_text().splitlines():
            key, value = line.split(':')
            io[key] = int(value)
    except (OSError, ValueError):
        io = None
    cuda = None
    if torch.cuda.is_available():
        cuda = {'allocated_bytes': torch.cuda.memory_allocated(),
                'reserved_bytes': torch.cuda.memory_reserved(),
                'device_free_total_bytes': torch.cuda.mem_get_info()}
    return {'memory': diagnostic_memory(), 'process_io': io, 'cuda': cuda}


def emit(event, **fields):
    print(json.dumps({'event': event, 'snapshot': snapshot(), **fields}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--model', required=True)
    p.add_argument('--malloc-trim', action='store_true')
    args = p.parse_args()
    if not args.execute:
        p.error('requiere --execute y un scope aislado con límite de memoria')
    import torch
    from diffusers.models.resnet import ResnetBlock2D
    from core.model_catalog import approved_path, checkpoint_family, SDXL
    from core.pizarra.config import PizarraConfig
    from core.pizarra.storage import PizarraStorage
    from core.pizarra.weight_offload import load_unet_block
    from core.pizarra.module_offload import ModuleOffload
    path = approved_path(args.model, 'checkpoint')
    if path is None or checkpoint_family(path) != SDXL or path.suffix.lower() != '.safetensors':
        p.error('se requiere un checkpoint SDXL local registrado')
    storage = PizarraStorage(PizarraConfig.from_env())
    if not storage.validate().get('available'):
        p.error('volumen PIZARRA no disponible')
    emit('before_load')
    # The first SDXL down-block ResNet is LDM input_blocks.1.0.
    raw = load_unet_block(path, 'model.diffusion_model.input_blocks.1.')
    mapping = {'in_layers.0.': 'norm1.', 'in_layers.2.': 'conv1.',
               'emb_layers.1.': 'time_emb_proj.', 'out_layers.0.': 'norm2.',
               'out_layers.3.': 'conv2.', 'skip_connection.': 'conv_shortcut.'}
    state = {}
    prefix = 'model.diffusion_model.input_blocks.1.0.'
    for name, tensor in raw.items():
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        for old, new in mapping.items():
            if suffix.startswith(old):
                state[new + suffix[len(old):]] = tensor
                break
        else:
            raise ValueError('Tensor ResNet desconocido: ' + name)
    del raw
    conv = state['conv1.weight']
    channels_out, channels_in = conv.shape[:2]
    temb_channels = state['time_emb_proj.weight'].shape[1]
    block = ResnetBlock2D(in_channels=channels_in, out_channels=channels_out,
                          temb_channels=temb_channels, groups=32).to(dtype=conv.dtype).eval()
    block.load_state_dict(state, strict=True)
    del state, conv
    root = torch.nn.Sequential(block)
    torch.manual_seed(7)
    x = torch.randn(1, channels_in, 8, 8, dtype=next(block.parameters()).dtype)
    temb = torch.randn(1, temb_channels, dtype=x.dtype)
    with torch.inference_mode():
        start = time.monotonic(); expected = block(x, temb); elapsed = time.monotonic() - start
    emit('baseline', seconds=elapsed)
    adapter = ModuleOffload(root, '0', storage)
    session = adapter.persist(on_session=lambda job, token: emit('session', job_id=job, token=token))
    emit('persisted', metadata=adapter.metadata, write_bytes=session['write_bytes'],
         write_seconds=session['write_seconds'], verify_read_bytes=session['verify_read_bytes'],
         verify_seconds=session['verify_seconds'])
    release = adapter.release()
    if args.malloc_trim:
        release['malloc_trim'] = trim_glibc()
    emit('released', **release)
    metrics = adapter.restore()
    emit('restored', **metrics)
    start = time.monotonic()
    actual = adapter.run(x, temb)
    elapsed = time.monotonic() - start
    difference = (actual.float() - expected.float()).abs()
    exact = torch.equal(actual, expected)
    emit('executed', seconds=elapsed, exact=exact, max_abs_error=difference.max().item())
    if not exact:
        raise RuntimeError('La salida difiere; sesión conservada')
    adapter.release()
    emit('released_again')
    adapter.restore()
    emit('cleaned', records_removed=adapter.cleanup())


if __name__ == '__main__':
    main()
