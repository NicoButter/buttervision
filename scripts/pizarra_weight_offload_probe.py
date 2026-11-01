"""Opt-in CPU-only physical UNet block round-trip; never performs inference."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='authorize experimental PIZARRA writes')
    parser.add_argument('--model', help='registered local Juggernaut XL safetensors')
    parser.add_argument('--recover-job', help='resume an interrupted session by job UUID')
    parser.add_argument('--token', help='session token printed at creation')
    parser.add_argument('--cleanup-only', action='store_true', help='clean an incomplete session without restoring')
    parser.add_argument('--block', type=int, default=4, help='LDM UNet input block number')
    args = parser.parse_args()
    if not args.execute:
        parser.error('use --execute to authorize the experimental write')
    from core.model_catalog import approved_path, checkpoint_family, SDXL
    from core.pizarra.config import PizarraConfig
    from core.pizarra.storage import PizarraStorage
    from core.pizarra.weight_offload import WeightOffload, load_unet_block, memory_snapshot
    if args.block < 0:
        parser.error('block must be nonnegative')
    storage = PizarraStorage(PizarraConfig.from_env())
    if not storage.validate().get('available'):
        parser.error('authorized PIZARRA volume unavailable')
    adapter = WeightOffload(storage)
    if args.recover_job:
        if not args.token:
            parser.error('--recover-job requires --token')
        if not args.cleanup_only:
            session = {'job_id': args.recover_job, 'token': args.token,
                       'index': adapter._read_index(args.recover_job)}
            restored, metrics = adapter.restore(session)
            print(json.dumps({'event': 'restored', 'tensors': len(restored),
                              'integrity': 'sha256 exact bytes', **metrics}), flush=True)
        removed = storage.cleanup_offload(args.recover_job, args.token)
        print(json.dumps({'event': 'cleaned', 'records_removed': removed}), flush=True)
        return 0
    if args.token or args.cleanup_only or not args.model:
        parser.error('provide --model or --recover-job with --token')
    path = approved_path(args.model, 'checkpoint')
    if path is None or path.suffix.lower() != '.safetensors' or checkpoint_family(path) != SDXL:
        parser.error('model must be a registered local SDXL safetensors checkpoint')
    prefix = f'model.diffusion_model.input_blocks.{args.block}.'
    weights = load_unet_block(path, prefix)
    print(json.dumps({'event': 'loaded', 'prefix': prefix, 'tensors': len(weights),
                      'bytes': sum(t.numel() * t.element_size() for t in weights.values()),
                      'memory': memory_snapshot()}), flush=True)
    def report_session(job, token):
        print(json.dumps({"event": "session", "job_id": job, "token": token}), flush=True)
    session = adapter.offload(weights, on_session=report_session)
    print(json.dumps({'event': 'released', **session}), flush=True)
    restored, metrics = adapter.restore(session)
    print(json.dumps({'event': 'restored', 'tensors': len(restored), 'integrity': 'sha256 exact bytes',
                      **metrics}), flush=True)
    removed = storage.cleanup_offload(session['job_id'], session['token'])
    print(json.dumps({'event': 'cleaned', 'records_removed': removed}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
