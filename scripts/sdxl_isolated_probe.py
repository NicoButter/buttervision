"""Run one opt-in SDXL probe inside a bounded user systemd scope.

The default invocation is informational and never loads a model. The worker is
intended only for manual hardware validation, not API requests or automated tests.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MEMORY_MAX_GIB = 10
HOST_HEADROOM_GIB = 4


def build_scope_command(model, memory_max_gib=DEFAULT_MEMORY_MAX_GIB, timeout_seconds=3600,
                        python_executable=sys.executable, systemd_run="systemd-run", timeout_executable="timeout"):
    """Build, but do not run, a bounded local systemd user scope command."""
    from core.model_catalog import SDXL, approved_path, checkpoint_family

    approved = approved_path(model, "checkpoint")
    if approved is None or checkpoint_family(approved) != SDXL:
        raise ValueError("El checkpoint debe ser SDXL y pertenecer al catálogo local registrado.")
    maximum = int(memory_max_gib)
    total_gib = psutil.virtual_memory().total // (1024**3)
    if maximum < 6 or maximum > total_gib - HOST_HEADROOM_GIB:
        raise ValueError("MemoryMax debe ser de al menos 6 GiB y dejar 4 GiB para el sistema.")
    if int(timeout_seconds) < 60 or int(timeout_seconds) > 7200:
        raise ValueError("El timeout debe estar entre 60 y 7200 segundos.")
    return [
        systemd_run, "--user", "--scope", "--wait", "--collect", "--quiet",
        f"--property=MemoryHigh={maximum - 1}G",
        f"--property=MemoryMax={maximum}G",
        "--property=MemorySwapMax=0",
        timeout_executable, "--signal=TERM", "--kill-after=30s", f"{int(timeout_seconds)}s",
        python_executable, str(Path(__file__).resolve()), "--worker", "--model", str(approved),
        "--timeout-seconds", str(int(timeout_seconds)),
    ]


def _worker(model, timeout_seconds):
    from core.model_catalog import SDXL, approved_path, checkpoint_family
    from core.sdxl_pipeline import SDXLPipeline, local_sdxl_config
    import config
    import torch

    approved = approved_path(model, "checkpoint")
    if approved is None or checkpoint_family(approved) != SDXL:
        raise SystemExit("El worker rechazó el checkpoint no registrado/incompatible.")
    if approved.is_file():
        local_sdxl_config(approved)  # validates a complete local config; never downloads
    if not torch.cuda.is_available():
        raise SystemExit("No se detecta CUDA; este sondeo valida SDXL en la GPU objetivo, no CPU.")
    if not config.model_config.enable_cpu_offload:
        raise SystemExit("La prueba aislada requiere el offload de CPU configurado.")

    backend = SDXLPipeline(str(approved))
    start = time.monotonic()
    try:
        images = backend.generate_image(
            "A simple landscape, controlled local SDXL validation",
            num_inference_steps=10, guidance_scale=5.0, width=512, height=512,
            seed=230231, num_images=1,
        )
        if time.monotonic() - start > timeout_seconds:
            raise TimeoutError("La inferencia excedió el tiempo manual configurado.")
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        output = config.OUTPUTS_DIR / f"sdxl_probe_{int(time.time())}.png"
        images[0].save(output)
        print(json.dumps({"status": "completed", "model": approved.name,
                          "seconds": round(time.monotonic() - start, 2), "output": str(output)}))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "seconds": round(time.monotonic() - start, 2)}), file=sys.stderr)
        return 1
    finally:
        backend.cleanup()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Opt-in: start one bounded hardware probe")
    parser.add_argument("--model", help="Path of a locally registered SDXL checkpoint")
    parser.add_argument("--memory-max-gib", type=int, default=DEFAULT_MEMORY_MAX_GIB)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if not args.model:
            parser.error("--model requerido")
        return _worker(args.model, args.timeout_seconds)
    if not args.execute:
        print("Sin inferencia. Para iniciar la única prueba controlada: añade --execute --model RUTA_LOCAL_REGISTRADA.")
        return 0
    if not args.model:
        parser.error("--model requerido junto a --execute")
    if not config_local_offload_enabled():
        raise SystemExit("Activa enable_cpu_offload en la configuración local antes de la prueba.")
    executable = shutil_which("systemd-run")
    if not executable:
        raise SystemExit("systemd-run no está disponible; no se iniciará una prueba sin límite de memoria.")
    timeout_executable = shutil_which("timeout")
    if not timeout_executable:
        raise SystemExit("timeout de GNU coreutils no está disponible; no se iniciará una prueba sin límite temporal.")
    command = build_scope_command(args.model, args.memory_max_gib, args.timeout_seconds,
                                  python_executable=sys.executable, systemd_run=executable,
                                  timeout_executable=timeout_executable)
    print("Iniciando prueba SDXL 512×512, 10 pasos, lote 1 en scope systemd limitado.", flush=True)
    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT, timeout=args.timeout_seconds + 60, check=False)
    except subprocess.TimeoutExpired:
        raise SystemExit("Timeout del scope; inspecciona systemd y el estado de la GPU.")
    return result.returncode


def config_local_offload_enabled():
    import config
    return bool(config.model_config.enable_cpu_offload)


def shutil_which(name):
    from shutil import which
    return which(name)


if __name__ == "__main__":
    raise SystemExit(main())
