"""Opt-in single-block CPU inference controller. Caller owns all root access.

Forward wrappers are instance-local and reversible; no PyTorch hooks are used.
External tensor references, direct descendant calls and mutation are forbidden.
"""
import threading
import torch

from .module_offload import ModuleOffload


class AutomaticOffload:
    def __init__(self, root, path, storage, *, before_restore=None):
        if not path:
            raise ValueError('Se requiere un submódulo distinto del root')
        self.adapter = ModuleOffload(root, path, storage)
        self.root, self.block = root, self.adapter.module
        self.before_restore = before_restore
        self.lock = threading.Lock()
        self.owner = None
        self.installed = False
        self._original = {}

    def _validate(self):
        self.adapter._check_hooks()
        if any(m.training for m in self.root.modules()):
            raise RuntimeError('Se requiere modo evaluación')
        if not torch.is_inference_mode_enabled():
            raise RuntimeError('Se requiere inference_mode')
        if self.installed:
            for module, (_, wrapper) in self._original.items():
                if module.__dict__.get('forward') is not wrapper:
                    raise RuntimeError('El forward fue modificado')

    def install(self, *, on_session=None):
        if self.installed or self.adapter.session is not None:
            raise RuntimeError('Controlador ya iniciado')
        self._validate()
        # Persist and release before installing wrappers. A failed release leaves
        # the session available for explicit recovery/cleanup.
        self.adapter.persist(on_session=on_session)
        self.adapter.release()
        root_forward, block_forward = self.root.forward, self.block.forward

        def root_wrapper(*args, **kwargs):
            if not self.lock.acquire(blocking=False):
                raise RuntimeError('Acceso concurrente o recursivo rechazado')
            try:
                self._validate()
                self.owner = threading.get_ident()
                return root_forward(*args, **kwargs)
            finally:
                self.owner = None
                self.lock.release()

        def block_wrapper(*args, **kwargs):
            if self.owner != threading.get_ident():
                raise RuntimeError('Acceso directo al bloque rechazado')
            self._validate()
            if self.adapter.state != 'released':
                raise RuntimeError('Estado del bloque incompatible')
            if self.before_restore:
                self.before_restore(self.adapter)
            self.adapter.restore()
            try:
                return block_forward(*args, **kwargs)
            finally:
                # release verifies both persistent and resident copies; on a
                # mismatch leave resident weights and the session recoverable.
                self.adapter.release()

        for module, wrapper in ((self.root, root_wrapper), (self.block, block_wrapper)):
            self._original[module] = (module.__dict__.get('forward'), wrapper)
            module.forward = wrapper
        self.installed = True
        return self.adapter.session

    def close(self):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('Controlador en ejecución')
        try:
            if self.installed:
                for module, (_, wrapper) in self._original.items():
                    if module.__dict__.get('forward') is not wrapper:
                        raise RuntimeError('El forward fue modificado')
            if self.adapter.state == 'released':
                self.adapter.restore()
            if self.installed:
                for module, (original, _) in self._original.items():
                    if original is None:
                        del module.forward
                    else:
                        module.forward = original
                self.installed = False
            if self.adapter.session is not None:
                return self.adapter.cleanup()
        finally:
            self.lock.release()
