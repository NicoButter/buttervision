"""Opt-in CPU-only SDXL image probe and isolated worker."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pizarra_unet_probe import check_scope, check_restore_budget, emit

PROMPT = 'A small red apple on a wooden table, soft daylight'
SEED = 507


from core.pizarra.sdxl_image_service import generate, prepare_embeddings

def worker(args):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from core.sdxl_pipeline import local_sdxl_config
    from core.sdxl_progressive_loader import load_progressive, validate_source
    from core.pizarra.config import PizarraConfig
    from core.pizarra.storage import PizarraStorage
    emit('scope_verified', scope=check_scope(args.unit, 10 * 1024**3))
    path = validate_source(args.model)
    storage = PizarraStorage(PizarraConfig.from_env())
    if not storage.validate().get('available'):
        raise RuntimeError('Volumen PIZARRA no disponible')
    start = time.monotonic()
    emit('sdxl_load_start')
    pipe = load_progressive(path, local_sdxl_config(path), dtype=torch.float16,
                            report=lambda record: emit('progressive_load', progress=record))
    emit('sdxl_load_end', seconds=time.monotonic() - start, load_metrics=pipe.progressive_load_metrics)
    request = json.load(sys.stdin) if args.request_stdin else {'prompt': PROMPT, 'seed': SEED}
    generate(pipe, storage, args.output, args.size, before_restore=check_restore_budget,
             prompt=request['prompt'], seed=request['seed'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--size', type=int, choices=(64, 128), default=64)
    p.add_argument('--timeout-seconds', type=int, default=3600)
    p.add_argument('--request-stdin', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--unit', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not args.execute or not 60 <= args.timeout_seconds <= 7200:
        p.error('requiere --execute y timeout entre 60 y 7200 segundos')
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
               '--property=MemoryMax=10G', '--property=MemorySwapMax=0', timeout,
               '--signal=TERM', '--kill-after=30s', f'{args.timeout_seconds}s', sys.executable,
               str(Path(__file__).resolve()), '--execute', '--worker', '--unit', unit,
               '--model', args.model, '--output', str(Path(args.output).absolute()), '--size', str(args.size),
               '--timeout-seconds', str(args.timeout_seconds)]
    lock = acquire_inference_lock()
    if lock is None:
        raise SystemExit('Ya hay una inferencia o carga activa')
    try:
        return subprocess.run(command, check=False).returncode
    finally:
        release_inference_lock(lock)


if __name__ == '__main__':
    raise SystemExit(main())
