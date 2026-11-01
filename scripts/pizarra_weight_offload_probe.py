"""Opt-in CPU-only physical UNet block round-trip; never performs inference."""
import argparse
import ctypes
import gc
import json
import os
import platform
import re
import resource
import weakref
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _mount_unescape(value):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def cgroup_memory_path(cgroup_text, mountinfo_text):
    """Resolve this process's cgroup v2 or memory-v1 file via mountinfo."""
    groups = {}
    for line in cgroup_text.splitlines():
        parts = line.split(':', 2)
        if len(parts) == 3:
            for controller in (parts[1].split(',') if parts[1] else ['v2']):
                groups[controller] = parts[2]
    for line in mountinfo_text.splitlines():
        try:
            before, after = line.split(' - ', 1)
            fields, mount = before.split(), after.split()
            fs = mount[0]
            if fs == 'cgroup2' and 'v2' in groups and 'memory' not in groups:
                group, filename = groups['v2'], 'memory.current'
            elif fs == 'cgroup' and 'memory' in groups and 'memory' in mount[2].split(','):
                group, filename = groups['memory'], 'memory.usage_in_bytes'
            else:
                continue
            root = _mount_unescape(fields[3])
            point = _mount_unescape(fields[4])
            if not all(x.startswith('/') for x in (root, point, group)):
                continue
            relative = os.path.relpath(group, root)
            if relative == '..' or relative.startswith('../'):
                continue
            return Path(point) / relative / filename
        except (ValueError, IndexError):
            continue
    return None


def diagnostic_memory():
    """Process RSS/USS/anonymous and the process's actual cgroup charge."""
    import psutil
    process = psutil.Process()
    rss = process.memory_info().rss
    try:
        uss = process.memory_full_info().uss
    except (psutil.Error, OSError, AttributeError):
        uss = None
    anonymous = None
    try:
        with open('/proc/self/smaps_rollup', encoding='ascii') as source:
            for line in source:
                if line.startswith('Anonymous:'):
                    anonymous = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError, IndexError):
        pass
    cgroup = None
    try:
        with open('/proc/self/cgroup', encoding='utf-8') as source:
            groups = source.read()
        with open('/proc/self/mountinfo', encoding='utf-8') as source:
            mounts = source.read()
        path = cgroup_memory_path(groups, mounts)
        if path is not None:
            cgroup = int(path.read_text(encoding='ascii').strip())
    except (OSError, ValueError):
        pass
    return {'rss_bytes': rss, 'uss_bytes': uss, 'anonymous_bytes': anonymous,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'cgroup_current_bytes': cgroup}


def trim_glibc():
    """Only called when the user explicitly requests this optional experiment."""
    if platform.libc_ver()[0] != 'glibc':
        return {'available': False, 'result': None}
    try:
        function = ctypes.CDLL(None).malloc_trim
        function.argtypes = [ctypes.c_size_t]
        function.restype = ctypes.c_int
        return {'available': True, 'result': int(function(0))}
    except (OSError, AttributeError):
        return {'available': False, 'result': None}


def weakref_counts(references):
    alive = sum(ref() is not None for ref in references)
    return {'original_tensors': len(references), 'alive': alive,
            'collected': len(references) - alive}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='authorize experimental PIZARRA writes')
    parser.add_argument('--model', help='registered local Juggernaut XL safetensors')
    parser.add_argument('--recover-job', help='resume an interrupted session by job UUID')
    parser.add_argument('--token', help='session token printed at creation')
    parser.add_argument('--cleanup-only', action='store_true', help='clean an incomplete session without restoring')
    parser.add_argument('--malloc-trim', action='store_true', help='explicitly test glibc malloc_trim after offload')
    parser.add_argument('--block', type=int, default=4, help='LDM UNet input block number')
    args = parser.parse_args()
    if not args.execute:
        parser.error('use --execute to authorize the experimental write')
    from core.model_catalog import approved_path, checkpoint_family, SDXL
    from core.pizarra.config import PizarraConfig
    from core.pizarra.storage import PizarraStorage
    from core.pizarra.weight_offload import WeightOffload, load_unet_block
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
                              'integrity': 'sha256 exact bytes', 'read_seconds': metrics['read_seconds'],
                              'read_bytes': metrics['read_bytes'], 'memory': diagnostic_memory()}), flush=True)
        removed = storage.cleanup_offload(args.recover_job, args.token)
        print(json.dumps({'event': 'cleaned', 'records_removed': removed}), flush=True)
        return 0
    if args.token or args.cleanup_only or not args.model:
        parser.error('provide --model or --recover-job with --token')
    path = approved_path(args.model, 'checkpoint')
    if path is None or path.suffix.lower() != '.safetensors' or checkpoint_family(path) != SDXL:
        parser.error('model must be a registered local SDXL safetensors checkpoint')
    prefix = f'model.diffusion_model.input_blocks.{args.block}.'
    print(json.dumps({'event': 'before_load', 'memory': diagnostic_memory()}), flush=True)
    weights = load_unet_block(path, prefix)
    references = [weakref.ref(tensor) for tensor in weights.values()]
    print(json.dumps({'event': 'loaded', 'prefix': prefix, 'tensors': len(weights),
                      'bytes': sum(t.numel() * t.element_size() for t in weights.values()),
                      'memory': diagnostic_memory()}), flush=True)
    def report_session(job, token):
        print(json.dumps({"event": "session", "job_id": job, "token": token}), flush=True)
    session = adapter.offload(weights, on_session=report_session)
    print(json.dumps({'event': 'released', 'job_id': session['job_id'],
                      'logical_released_bytes': session['logical_released_bytes'],
                      'write_bytes': session['write_bytes'], 'write_seconds': session['write_seconds'],
                      'verify_read_bytes': session['verify_read_bytes'],
                      'verify_seconds': session['verify_seconds'],
                      'weakrefs': weakref_counts(references), 'memory': diagnostic_memory()}), flush=True)
    collected = gc.collect()
    print(json.dumps({'event': 'after_gc', 'gc_collected_objects': collected,
                      'weakrefs': weakref_counts(references), 'memory': diagnostic_memory()}), flush=True)
    if args.malloc_trim:
        result = trim_glibc()
        print(json.dumps({'event': 'after_malloc_trim', 'trim': result,
                          'memory': diagnostic_memory()}), flush=True)
    restored, metrics = adapter.restore(session)
    print(json.dumps({'event': 'restored', 'tensors': len(restored), 'integrity': 'sha256 exact bytes',
                      'read_seconds': metrics['read_seconds'], 'read_bytes': metrics['read_bytes'],
                      'memory': diagnostic_memory()}), flush=True)
    removed = storage.cleanup_offload(session['job_id'], session['token'])
    print(json.dumps({'event': 'cleaned', 'records_removed': removed}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
