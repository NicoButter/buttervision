#!/usr/bin/env python3
"""Operator-driven comparison of existing txt2img with and without PIZARRA.

Run the two modes in separate server sessions, then use `compare`. Nothing is
executed at import time or without the explicit --execute flag.
"""
import argparse
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import psutil

PROMPT = "A watercolor landscape with mountains and a lake"


def base_url(value):
    parts = urlsplit(value)
    if parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost", "::1") or parts.username or parts.password or parts.path not in ("", "/") or parts.query or parts.fragment:
        raise ValueError("El benchmark sólo acepta una URL HTTP de loopback, sin ruta ni credenciales.")
    return value.rstrip("/")


def request(base, path, payload=None, binary=False):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    with urlopen(Request(base + path, data=data, headers=headers), timeout=30) as response:
        content = response.read()
    return content if binary else json.loads(content)


class ProcessMonitor:
    def __init__(self, pid):
        self.process = psutil.Process(pid)
        self.pid = pid
        self.stop = threading.Event()
        self.thread = None
        self.rss_peak = None
        self.vram_peak = None
        self.io_start = None
        self.io_end = None

    def io(self):
        try:
            data = self.process.io_counters()
            return {"read_bytes": data.read_bytes, "write_bytes": data.write_bytes}
        except (psutil.Error, OSError, AttributeError):
            return None

    def sample(self):
        try:
            rss = self.process.memory_info().rss
            self.rss_peak = max(self.rss_peak or 0, rss)
        except (psutil.Error, OSError):
            pass
        try:
            output = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                text=True, capture_output=True, timeout=3, check=True,
            ).stdout
            matches = []
            for row in output.splitlines():
                fields = [part.strip() for part in row.split(",")]
                if len(fields) >= 2 and fields[0].isdigit() and int(fields[0]) == self.pid and fields[1].isdigit():
                    matches.append(int(fields[1]) * 1024**2)
            if matches:
                used = sum(matches)
                self.vram_peak = max(self.vram_peak or 0, used)
        except (OSError, subprocess.SubprocessError):
            pass

    def _poll(self):
        while not self.stop.wait(1.0):
            self.sample()

    def start(self):
        self.io_start = self.io()
        self.sample()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=4)
        self.sample()
        self.io_end = self.io()
        io_delta = None
        if self.io_start is not None and self.io_end is not None:
            io_delta = {key: max(0, self.io_end[key] - self.io_start[key]) for key in self.io_start}
        return {"rss_process_peak_sampled_bytes": self.rss_peak,
                "vram_process_peak_nvidia_smi_sampled_bytes": self.vram_peak,
                "process_io_bytes": io_delta}


def run(args):
    if not args.execute:
        raise ValueError("La generación real requiere la opción explícita --execute.")
    base = base_url(args.url)
    status = request(base, "/api/pizarra/status")
    if bool(status.get("enabled")) != (args.mode == "pizarra"):
        raise ValueError("El modo del servidor no coincide con el modo solicitado.")
    if args.mode == "pizarra" and not status.get("available"):
        raise ValueError("El volumen PIZARRA no está disponible o no está autorizado.")
    models = request(base, "/api/models")
    active = next((model for model in models if model.get("active")), None)
    if not active:
        raise ValueError("No hay un modelo activo disponible.")
    parameters = {"prompt": PROMPT, "negative_prompt": "", "width": args.width,
                  "height": args.height, "steps": args.steps, "cfg_scale": args.cfg,
                  "batch_size": 1, "seed": args.seed}
    monitor = ProcessMonitor(args.server_pid)
    monitor.start()
    start = time.monotonic()
    created = None
    try:
        created = request(base, "/api/generate/txt2img", parameters)
        job_id = created["job_id"]
        deadline = time.monotonic() + args.timeout
        while True:
            job = request(base, "/api/jobs/" + job_id)
            if job["status"] in ("completed", "failed", "interrupted"):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Tiempo de espera agotado; el trabajo puede seguir ejecutándose en el servidor.")
            time.sleep(1)
    finally:
        observations = monitor.finish()
    duration = time.monotonic() - start
    engine = request(base, "/api/engine")
    current_models = request(base, "/api/models")
    current_active = next((model for model in current_models if model.get("active")), None)
    if not current_active or current_active.get("value") != active["value"]:
        raise ValueError("El modelo activo cambió durante la medición; el resultado no es comparable.")
    active_loras = sorted(
        [{"identifier_sha256": hashlib.sha256(item["value"].encode()).hexdigest(), "weight": item["weight"]}
         for item in engine.get("active_loras", [])], key=lambda item: item["identifier_sha256"])
    result_hashes = []
    for value in job.get("result_images", []):
        path = PurePosixPath(value)
        if not value.startswith("/outputs/") or ".." in path.parts or "?" in value or "#" in value:
            raise ValueError("La respuesta contiene una ruta de resultado no válida.")
        result_hashes.append(hashlib.sha256(request(base, value, binary=True)).hexdigest())
    pizarra_metrics = None
    if args.mode == "pizarra":
        try:
            pizarra_metrics = request(base, "/api/pizarra/jobs/" + job_id + "/metrics")
        except HTTPError:
            pass
    return {"format_version": 1, "mode": args.mode, "job_id": job_id,
            "model_name": active["label"],
            "model_identifier_sha256": hashlib.sha256(active["value"].encode()).hexdigest(),
            "family": active.get("family"), "scheduler": engine.get("scheduler"), "active_loras": active_loras,
            "parameters": {key: value for key, value in parameters.items() if key not in ("prompt", "negative_prompt")},
            "prompt_id": "pizarra-benchmark-fixed-v1", "status": job["status"],
            "failure": job.get("error"), "result_sha256": result_hashes,
            "wall_seconds": duration, "observations": observations, "pizarra_metrics": pizarra_metrics}


def compare(first, second):
    runs = {first.get("mode"): first, second.get("mode"): second}
    if set(runs) != {"legacy", "pizarra"}:
        raise ValueError("Se necesita un resultado legacy y otro pizarra.")
    a, b = runs["legacy"], runs["pizarra"]
    fields = ("model_identifier_sha256", "family", "scheduler", "parameters", "prompt_id", "active_loras")
    mismatch = [key for key in fields if a.get(key) != b.get(key) or a.get(key) is None]
    if mismatch:
        raise ValueError("La comparación no está controlada; difieren o faltan: " + ", ".join(mismatch))
    return {"comparable_parameters": True, "legacy_status": a["status"], "pizarra_status": b["status"],
            "legacy_wall_seconds": a["wall_seconds"], "pizarra_wall_seconds": b["wall_seconds"],
            "legacy_observations": a["observations"], "pizarra_observations": b["observations"],
            "pizarra_metrics": b.get("pizarra_metrics"),
            "same_result_file_sha256": a["result_sha256"] == b["result_sha256"] if a["result_sha256"] and b["result_sha256"] else None,
            "note": "Los hashes son del archivo completo, incluida metadata; los picos son muestreados y la E/S es del proceso. Cachés y otros procesos pueden influir."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run", help="Ejecuta una medición; requiere --execute")
    one.add_argument("--execute", action="store_true")
    one.add_argument("--mode", choices=("legacy", "pizarra"), required=True)
    one.add_argument("--url", default="http://127.0.0.1:7861")
    one.add_argument("--server-pid", type=int, required=True)
    one.add_argument("--output", type=Path, required=True)
    one.add_argument("--seed", type=int, default=42)
    one.add_argument("--width", type=int, default=512)
    one.add_argument("--height", type=int, default=512)
    one.add_argument("--steps", type=int, default=20)
    one.add_argument("--cfg", type=float, default=5.0)
    one.add_argument("--timeout", type=float, default=7200)
    comparison = sub.add_parser("compare", help="Compara dos archivos previamente guardados")
    comparison.add_argument("legacy", type=Path)
    comparison.add_argument("pizarra", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "run":
            if not args.execute:
                raise ValueError("La generación real requiere --execute.")
            # Reserve the destination first so an existing report is never overwritten.
            with args.output.open("x", encoding="utf-8") as file:
                try:
                    result = run(args)
                    json.dump(result, file, indent=2, ensure_ascii=False)
                    file.write("\n")
                except Exception:
                    # Keep the empty file as evidence of an incomplete benchmark.
                    raise
            print(f"Medición guardada en {args.output}")
        else:
            result = compare(json.loads(args.legacy.read_text()), json.loads(args.pizarra.read_text()))
            print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, OSError, TimeoutError, HTTPError, psutil.Error, KeyError) as error:
        parser.exit(2, f"Error de benchmark: {error}\n")


if __name__ == "__main__":
    main()
