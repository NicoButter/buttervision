"""Run one opt-in SDXL probe inside a bounded user systemd scope.

The default invocation is informational and never loads a model. The worker is
intended only for manual hardware validation, not API requests or automated tests.

One Fedora acceptance run (GTX 1650 4 GB / 16 GB RAM), from the repository:
    venv/bin/python scripts/sdxl_isolated_probe.py --execute --load-only \
        --model "$HOME/stable-diffusion-webui-forge/models/Stable-diffusion/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors" \
        --strategy LOW_RAM --memory-max-gib 10 --timeout-seconds 3600

Uses the existing local Forge configuration, never downloads. Requires configured
CPU offload as existing probes do. Exit 0 requires all four weight components,
both tokenizers and scheduler, exact shapes and zero meta parameters/buffers.
JSON records contain per-component/read/conversion/total times, loaded counts,
observed RSS/cgroup/CUDA allocator peaks (null if unavailable), and release.
Read time covers mapping/get_tensor; demand-page I/O during copy is included in
conversion time. Peaks are observed per tensor, not a claim of continuous hardware
sampling. The existing supervisor additionally samples process-tree RSS.
A successful tiny fixture or header plan is not physical Juggernaut acceptance.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from types import SimpleNamespace
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MEMORY_MAX_GIB = 10
HOST_HEADROOM_GIB = 4
from core.inference_lock import acquire_inference_lock, release_inference_lock


def build_scope_command(model, memory_max_gib=DEFAULT_MEMORY_MAX_GIB, timeout_seconds=3600,
                        python_executable=sys.executable, systemd_run="systemd-run", timeout_executable="timeout",
                        strategy="AUTO", disk_offload_dir=None, pizarra_adapter=False,
                        pizarra_strategy="AUTO", load_only=False):
    """Build, but do not run, a bounded local systemd user scope command."""
    from core.model_catalog import SDXL, approved_path, checkpoint_family

    strategy = str(strategy).upper()
    if strategy not in ("AUTO", "LOW_RAM", "LOW_VRAM", "DISK_OFFLOAD"):
        raise ValueError("El sondeo acepta AUTO, LOW_RAM, LOW_VRAM o DISK_OFFLOAD.")
    if pizarra_adapter and strategy == "DISK_OFFLOAD":
        raise ValueError("El sondeo PIZARRA no puede usar el offload a disco no administrado por PIZARRA.")
    approved = approved_path(model, "checkpoint")
    if approved is None or checkpoint_family(approved) != SDXL:
        raise ValueError("El checkpoint debe ser SDXL y pertenecer al catálogo local registrado.")
    maximum = int(memory_max_gib)
    total_gib = psutil.virtual_memory().total // (1024**3)
    if maximum < 6 or maximum > total_gib - HOST_HEADROOM_GIB:
        raise ValueError("MemoryMax debe ser de al menos 6 GiB y dejar 4 GiB para el sistema.")
    if int(timeout_seconds) < 60 or int(timeout_seconds) > 7200:
        raise ValueError("El timeout debe estar entre 60 y 7200 segundos.")
    command = [
        systemd_run, "--user", "--scope", "--wait", "--collect", "--quiet",
        f"--property=MemoryHigh={maximum - 1}G",
        f"--property=MemoryMax={maximum}G",
        "--property=MemorySwapMax=0",
        timeout_executable, "--signal=TERM", "--kill-after=30s", f"{int(timeout_seconds)}s",
        python_executable, str(Path(__file__).resolve()), "--worker", "--model", str(approved),
        "--timeout-seconds", str(int(timeout_seconds)), "--strategy", strategy,
    ]
    if load_only:
        if pizarra_adapter or strategy == "DISK_OFFLOAD":
            raise ValueError("--load-only no se combina con PIZARRA ni offload HDD")
        command.append("--load-only")
    if disk_offload_dir is not None:
        command.insert(7, f"--setenv=BUTTERVISION_SDXL_DISK_OFFLOAD_DIR={disk_offload_dir}")
    if pizarra_adapter:
        command.append("--pizarra-adapter")
        command.extend(("--pizarra-strategy", str(pizarra_strategy).upper()))
    return command


def _worker(model, timeout_seconds, strategy="AUTO", pizarra_adapter=False, pizarra_strategy="AUTO", load_only=False):
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
    if pizarra_adapter and strategy == "DISK_OFFLOAD":
        raise SystemExit("El adaptador PIZARRA bloquea el offload a disco no administrado por su almacenamiento.")
    start = time.monotonic()
    if load_only:
        from core.sdxl_pipeline import check_budget
        from core.sdxl_progressive_loader import load_progressive, assert_initialized
        import gc
        pipe = None
        try:
            plan = check_budget(approved, requested=strategy)
            if not plan.get("progressive_load"):
                raise ValueError("El checkpoint no admite carga progresiva")
            pipe = load_progressive(approved, local_sdxl_config(approved), dtype=torch.float16,
                                    report=lambda record: print(json.dumps(record), flush=True))
            assert_initialized({name: getattr(pipe, name) for name in
                                ("unet", "vae", "text_encoder", "text_encoder_2")})
            if any(getattr(pipe, name, None) is None for name in
                   ("tokenizer", "tokenizer_2", "scheduler")):
                raise ValueError("Pipeline auxiliar incompleto")
            print(json.dumps({"status": "loaded", **pipe.progressive_load_metrics}), flush=True)
            return 0
        except Exception as error:
            print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                              "reason": str(error)}), file=sys.stderr, flush=True)
            return 1
        finally:
            pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            print(json.dumps({"event": "sdxl_progressive_load", "phase": "released",
                              "rss_bytes": psutil.Process().memory_info().rss,
                              "seconds": time.monotonic() - start}), flush=True)
    if pizarra_adapter:
        from core.pizarra.config import PizarraConfig
        from core.pizarra.storage import PizarraStorage
        from core.pizarra.inference_adapter import (
            InferenceExecutionError, InferencePlanError, PizarraInferenceAdapter,
        )

        pizarra_config = PizarraConfig.from_env()
        storage = PizarraStorage(pizarra_config) if pizarra_config.enabled else None
        backend = PizarraInferenceAdapter(
            str(approved), storage, requested_strategy=pizarra_strategy,
            metrics_enabled=pizarra_config.metrics_enabled,
        )
        run_id = str(uuid.uuid4())
        try:
            planned = backend.plan()

            def report_phase(phase, model_path, plan, current_run_id):
                print(json.dumps({"event": "pizarra_inference_probe", "run_id": current_run_id,
                                  "phase": phase, "model": model_path.name, "family": "sdxl",
                                  "strategy": plan["strategy"], "profile": plan["selected_profile"],
                                  "initial_resources": {
                                      "ram_available_bytes": plan["ram_available_bytes"],
                                      "process_rss_bytes": plan.get("ram_process_rss_bytes"),
                                      "vram_available_bytes": plan.get("vram_available_bytes"),
                                  }}, sort_keys=True), flush=True)

            report_phase("preparing", planned[0], planned[1], run_id)
            images, report = backend.execute(
                "A simple landscape, controlled local SDXL validation",
                num_inference_steps=10, guidance_scale=5.0, width=512, height=512,
                seed=230231, num_images=1, run_id=run_id, planned=planned,
                on_phase=report_phase,
            )
            config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            output = config.OUTPUTS_DIR / f"sdxl_pizarra_probe_{int(time.time())}.png"
            images[0].save(output)
            report["output_file"] = output.name
            report["timings_seconds"]["total_including_output_save"] = time.monotonic() - start
            print(json.dumps(report, sort_keys=True))
            return 0
        except InferenceExecutionError as error:
            print(json.dumps(error.record, sort_keys=True), file=sys.stderr)
            return 1
        except InferencePlanError as error:
            print(json.dumps({"status": "rejected", "run_id": run_id,
                              "model": approved.name,
                              "family": "sdxl", "error_type": type(error).__name__,
                              "requested_strategy": pizarra_strategy.lower(),
                              "reason": str(error), "plan": error.plan,
                              "elapsed_seconds": time.monotonic() - start}, sort_keys=True), file=sys.stderr)
            return 1
        except Exception as error:
            print(json.dumps({"status": "failed", "model": approved.name,
                              "family": "sdxl", "error_type": type(error).__name__,
                              "elapsed_seconds": time.monotonic() - start}, sort_keys=True), file=sys.stderr)
            return 1

    if strategy != "DISK_OFFLOAD" and not config.model_config.enable_cpu_offload:
        raise SystemExit("La prueba aislada requiere el offload de CPU configurado.")

    backend = SDXLPipeline(str(approved), requested_strategy=strategy,
                           allow_experimental_disk=strategy == "DISK_OFFLOAD")
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
    parser.add_argument("--strategy", default="AUTO", choices=("AUTO", "LOW_RAM", "LOW_VRAM", "DISK_OFFLOAD"),
                        type=str.upper, help="Perfil interno para la prueba; DISK_OFFLOAD es experimental")
    parser.add_argument("--pizarra-adapter", action="store_true",
                        help="Opt-in: ejecutar mediante el adaptador experimental PIZARRA")
    parser.add_argument("--pizarra-strategy", default="AUTO",
                        choices=("AUTO", "STANDARD", "CPU_OFFLOAD", "DISK_OFFLOAD_EXPERIMENTAL"),
                        type=str.upper, help="Estrategia del adaptador PIZARRA; el offload HDD está bloqueado")
    parser.add_argument("--load-only", action="store_true", help="Opt-in: cargar SDXL completo sin inferencia ni imágenes")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.load_only and (args.pizarra_adapter or args.strategy == "DISK_OFFLOAD"):
        parser.error("--load-only requiere el cargador SDXL sin PIZARRA ni HDD")
    if args.pizarra_adapter and args.strategy == "DISK_OFFLOAD":
        parser.error("--pizarra-adapter no se combina con --strategy DISK_OFFLOAD; el HDD no tiene una concesión segura.")
    if args.worker:
        if not args.model:
            parser.error("--model requerido")
        return _worker(args.model, args.timeout_seconds, args.strategy,
                       args.pizarra_adapter, args.pizarra_strategy, args.load_only)
    if not args.execute:
        print("Sin inferencia. Para iniciar la única prueba controlada: añade --execute --model RUTA_LOCAL_REGISTRADA.")
        return 0
    if not args.model:
        parser.error("--model requerido junto a --execute")
    if not args.pizarra_adapter and args.strategy != "DISK_OFFLOAD" and not config_local_offload_enabled():
        raise SystemExit("Activa enable_cpu_offload en la configuración local antes de la prueba.")
    executable = shutil_which("systemd-run")
    if not executable:
        raise SystemExit("systemd-run no está disponible; no se iniciará una prueba sin límite de memoria.")
    timeout_executable = shutil_which("timeout")
    if not timeout_executable:
        raise SystemExit("timeout de GNU coreutils no está disponible; no se iniciará una prueba sin límite temporal.")
    command = build_scope_command(args.model, args.memory_max_gib, args.timeout_seconds,
                                  python_executable=sys.executable, systemd_run=executable,
                                  timeout_executable=timeout_executable, strategy=args.strategy,
                                  pizarra_adapter=args.pizarra_adapter,
                                  pizarra_strategy=args.pizarra_strategy, **({"load_only": True} if args.load_only else {}))
    print("Iniciando carga SDXL sin inferencia en scope limitado." if args.load_only else
          "Iniciando prueba SDXL 512×512, 10 pasos, lote 1 en scope systemd limitado.", flush=True)
    inference_fd = acquire_inference_lock()
    if inference_fd is None:
        raise SystemExit("Ya hay una carga o inferencia activa en Buttervision; la prueba aislada no se iniciará.")
    disk_run_dir = None
    try:
        if args.strategy == "DISK_OFFLOAD":
            from core.sdxl_memory import _disk_target

            disk = _disk_target()
            if not disk["valid"]:
                raise SystemExit(disk["reason"])
            Path(disk["path"]).mkdir(parents=True, exist_ok=True, mode=0o700)
            disk_run_dir = tempfile.mkdtemp(prefix="buttervision-sdxl-probe-", dir=disk["path"])
            os.chmod(disk_run_dir, 0o700)
            command = build_scope_command(
                args.model, args.memory_max_gib, args.timeout_seconds,
                python_executable=sys.executable, systemd_run=executable,
                timeout_executable=timeout_executable, strategy=args.strategy,
                disk_offload_dir=disk_run_dir, pizarra_adapter=args.pizarra_adapter,
                pizarra_strategy=args.pizarra_strategy,
            )
        result = run_supervised(command, cwd=PROJECT_ROOT, timeout_seconds=args.timeout_seconds + 60)
    finally:
        if disk_run_dir is not None:
            shutil.rmtree(disk_run_dir, ignore_errors=True)
        release_inference_lock(inference_fd)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    print(json.dumps({"event": "supervisor_process_monitor", **result.process_monitor}, sort_keys=True))
    if result.returncode:
        progress = _last_pizarra_progress(result.stdout)
        if result.returncode == 124:
            category = "timeout"
        elif result.returncode in (137, -9, -15):
            category = "worker_terminated_signal_or_cgroup_oom"
        elif "oom" in result.stderr.lower() or "memory" in result.stderr.lower():
            category = "possible_memory_limit"
        else:
            category = "worker_or_systemd_error"
        print(json.dumps({"status": "failed", "isolated": True,
                          "return_code": result.returncode, "failure_category": category,
                          "last_worker_progress": progress,
                          "reason": "Inspecciona stderr, el journal del scope y la última fase registrada.",
                          "process_monitor": result.process_monitor}, sort_keys=True), file=sys.stderr)
    return result.returncode


def _last_pizarra_progress(output):
    result = None
    for line in output.splitlines():
        try:
            candidate = json.loads(line)
        except (TypeError, ValueError):
            continue
        if candidate.get("event") == "pizarra_inference_probe":
            result = candidate
    return result


def run_supervised(command, *, cwd, timeout_seconds, sample_interval=0.25):
    """Capture worker output and periodically sample the wrapper process tree."""
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file, \
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file:
        process = subprocess.Popen(command, cwd=cwd, stdout=stdout_file, stderr=stderr_file,
                                   text=True, close_fds=True)
        started = time.monotonic()
        initial, peak, final = _process_tree_rss(process.pid), 0, 0
        peak = initial
        timed_out = False
        while process.poll() is None:
            current = _process_tree_rss(process.pid)
            peak = max(peak, current)
            try:
                process.wait(timeout=sample_interval)
            except subprocess.TimeoutExpired:
                if time.monotonic() - started >= timeout_seconds:
                    timed_out = True
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
        final = _process_tree_rss(process.pid)
        peak = max(peak, final)
        stdout_file.seek(0)
        stderr_file.seek(0)
        code = 124 if timed_out else process.returncode
        return SimpleNamespace(
            returncode=code, stdout=stdout_file.read(), stderr=stderr_file.read(),
            process_monitor={"rss_initial_process_tree_bytes": initial,
                             "rss_final_process_tree_bytes": final,
                             "rss_peak_sampled_process_tree_bytes": peak,
                             "sample_interval_seconds": sample_interval,
                             "duration_seconds": time.monotonic() - started,
                             "timed_out": timed_out},
        )


def _process_tree_rss(pid):
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
            continue
    return total


def config_local_offload_enabled():
    import config
    return bool(config.model_config.enable_cpu_offload)


def shutil_which(name):
    from shutil import which
    return which(name)


if __name__ == "__main__":
    raise SystemExit(main())
