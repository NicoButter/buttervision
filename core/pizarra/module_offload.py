"""Explicit experimental CPU module offload; caller must provide exclusive access.

No hooks are installed. The caller must serialize all module access, use inference
mode and ensure that no external references retain or mutate its tensors.
"""
import time
import torch

from .weight_offload import WeightOffload


class ModuleOffload:
    def __init__(self, root, path, storage, *, policy=None):
        self.root, self.path = root, path
        self.module = root.get_submodule(path)
        self.backend = WeightOffload(storage, policy=policy)
        self.session = None
        self.state = 'resident'
        # Accelerate attaches _hf_hook to modules, including ancestors. Ordinary
        # PyTorch hooks may also rely on parameter identity or mutate execution.
        self._check_hooks()
        entries = []
        for name, module in self.module.named_modules(remove_duplicate=False):
            for kind in ('_parameters', '_buffers'):
                for key, tensor in getattr(module, kind).items():
                    if tensor is None:
                        continue
                    full = '.'.join(x for x in (name, key) if x)
                    entries.append((full, module, kind, key, tensor))
        if not entries:
            raise ValueError('El módulo no contiene tensores')
        # Shared storage anywhere in the root would invalidate release and/or
        # break weight tying. Reject it rather than silently changing semantics.
        all_tensors = list(root.named_parameters(remove_duplicate=False)) + list(root.named_buffers(remove_duplicate=False))
        pointers = [(t.device.type, t.untyped_storage().data_ptr()) for _, t in all_tensors if t.device.type != 'meta']
        for _, _, _, _, t in entries:
            if (t.device.type != 'cpu' or t.dtype not in (torch.float16, torch.float32)
                or not t.is_contiguous() or t.numel() == 0 or t.ndim == 0
                or pointers.count((t.device.type, t.untyped_storage().data_ptr())) != 1):
                raise ValueError('Tensor o almacenamiento compartido incompatible')
        self.entries = [(n, m, k, key, bool(t.requires_grad)) for n, m, k, key, t in entries]
        self.metadata = {n: {'shape': list(t.shape), 'dtype': str(t.dtype),
                             'kind': k, 'requires_grad': bool(t.requires_grad)}
                         for n, _, k, _, t in entries}

    def _check_hooks(self):
        if self.root.get_submodule(self.path) is not self.module:
            raise RuntimeError('El módulo seleccionado cambió')
        for _, module in self.root.named_modules(remove_duplicate=False):
            if (getattr(module, '_hf_hook', None) is not None or
                any(getattr(module, attr, {}) for attr in (
                    '_forward_hooks', '_forward_pre_hooks', '_backward_hooks',
                    '_backward_pre_hooks', '_state_dict_hooks', '_load_state_dict_pre_hooks'))):
                raise ValueError('Hooks existentes incompatibles con offload explícito')

    def persist(self, *, on_session=None):
        if self.session is not None or self.state != 'resident':
            raise RuntimeError('Estado incompatible con persistencia')
        self._check_hooks()
        weights = {n: getattr(m, k)[key].detach() for n, m, k, key, _ in self.entries}
        self.session = self.backend.offload(weights, on_session=on_session)
        return self.session

    def release(self):
        if self.session is None or self.state != 'resident':
            raise RuntimeError('Estado incompatible con liberación')
        self._check_hooks()
        # Re-verify the only durable copy before replacing any live tensor.
        self.backend._verify(self.session['job_id'], self.session['index'])
        replacements = []
        for name, module, kind, key, requires_grad in self.entries:
            tensor = getattr(module, kind)[key]
            item = self.session['index'][name]
            import hashlib
            if (tensor.device.type != 'cpu' or list(tensor.shape) != item['shape'] or
                str(tensor.detach().numpy().dtype) != item['dtype'] or
                hashlib.sha256(memoryview(tensor.detach().numpy()).cast('B')).hexdigest() != item['sha256']):
                raise RuntimeError('El tensor residente cambió desde su persistencia')
            meta = torch.empty_like(tensor, device='meta')
            replacements.append((module, kind, key, torch.nn.Parameter(meta, requires_grad=requires_grad)
                                 if kind == '_parameters' else meta))
        start = time.monotonic()
        for module, kind, key, replacement in replacements:
            getattr(module, kind)[key] = replacement
        self.state = 'released'
        return {'release_seconds': time.monotonic() - start}

    def restore(self):
        if self.state != 'released':
            raise RuntimeError('Estado incompatible con restauración')
        self._check_hooks()
        tensors, metrics = self.backend.restore(self.session)
        replacements = [(module, kind, key, torch.nn.Parameter(tensors[name], requires_grad=requires_grad)
                         if kind == '_parameters' else tensors[name])
                        for name, module, kind, key, requires_grad in self.entries]
        for module, kind, key, replacement in replacements:
            getattr(module, kind)[key] = replacement
        self.state = 'resident'
        return metrics

    def run(self, *args, **kwargs):
        if self.state == 'released':
            self.restore()
        if self.state != 'resident':
            raise RuntimeError('Módulo no disponible')
        with torch.inference_mode():
            return self.module(*args, **kwargs)

    def cleanup(self):
        if self.state != 'resident' or self.session is None:
            raise RuntimeError('Restaurar antes de limpiar')
        removed = self.backend.storage.cleanup_offload(self.session['job_id'], self.session['token'])
        self.session = None
        return removed
