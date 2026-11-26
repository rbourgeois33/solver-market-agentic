"""Bench worker: runs on the GPU node, keeps one driver server per library alive (matrix loaded
once on the GPU) and serves the request queue one request at a time (no concurrent timings).

Queue directory layout (shared filesystem between the login node and the GPU node):
  requests/<id>.req   written atomically by `sma-tools run` (tmp + rename)
  done/<id>.req       processed requests
  results/<id>.json   written by the driver (or by the dispatcher on driver failure)
  drivers/<lib>.json  load statistics of each driver start (read/upload time)
  heartbeat.json      refreshed every loop: host, job id, pid, time, state
  STOP                create this file to make the worker exit
The worker exits after `idle_timeout` seconds without requests (safety net against orphan jobs).

Usage: python -m sma.worker.dispatcher --system DIR --queue DIR --bin DIR [--idle-timeout S]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

LIBS = ("hypre", "muelu", "petsc")


def read_kv(path: Path) -> dict:
    kv = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            k, _, v = line.partition(" ")
            kv[k] = v
    return kv


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


class Driver:
    def __init__(self, lib: str, exe: Path, system: Path, queue: Path, launcher: list[str]):
        self.lib, self.exe, self.system, self.queue = lib, exe, system, queue
        self.launcher = launcher
        self.proc: subprocess.Popen | None = None
        self.stderr_tail: deque[str] = deque(maxlen=60)
        self.lines: deque[str] = deque()
        self.cv = threading.Condition()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _pump(self, stream, sink):
        for line in iter(stream.readline, ""):
            sink(line.rstrip("\n"))

    def _on_stdout(self, line: str):
        with self.cv:
            self.lines.append(line)
            self.cv.notify_all()

    def _wait_line(self, prefix: str, timeout: float) -> str | None:
        end = time.time() + timeout
        with self.cv:
            while True:
                while self.lines:
                    line = self.lines.popleft()
                    if line.startswith(prefix):
                        return line[len(prefix):].strip()
                    self.stderr_tail.append("[stdout] " + line)
                if not self.alive() or time.time() > end:
                    return None
                self.cv.wait(timeout=1.0)

    def start(self, load_timeout: float = 1800):
        log = self.queue / "drivers" / f"{self.lib}.stderr.log"
        self.proc = subprocess.Popen([*self.launcher, str(self.exe), "--system", str(self.system)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.stderr_tail.clear()
        self.lines.clear()

        def err_sink(line, f=open(log, "a")):
            f.write(line + "\n"); f.flush()
            self.stderr_tail.append(line)

        threading.Thread(target=self._pump, args=(self.proc.stdout, self._on_stdout), daemon=True).start()
        threading.Thread(target=self._pump, args=(self.proc.stderr, err_sink), daemon=True).start()
        t0 = time.time()
        ready = self._wait_line("READY", load_timeout)
        if ready is None:
            self.stop()
            raise RuntimeError(f"{self.lib} driver failed to start:\n" + "\n".join(self.stderr_tail))
        info = json.loads(ready)
        info.update({"lib": self.lib, "wall_start_s": time.time() - t0, "started_at": time.time()})
        write_json(self.queue / "drivers" / f"{self.lib}.json", info)

    def request(self, req: Path, timeout: float) -> str | None:
        self.proc.stdin.write(f"{req}\n")
        self.proc.stdin.flush()
        return self._wait_line("DONE", timeout)

    def stop(self):
        if self.proc is None:
            return
        if self.alive():
            try:
                self.proc.stdin.write("QUIT\n"); self.proc.stdin.flush()
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
        self.proc = None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--system", required=True)
    p.add_argument("--queue", required=True)
    p.add_argument("--bin", required=True, help="directory with sma_hypre, sma_muelu, sma_petsc")
    p.add_argument("--idle-timeout", type=float, default=1800)
    p.add_argument("--launcher", default="",
                   help="prefix for driver processes, e.g. 'srun --overlap -n1' inside a Slurm job "
                        "(each driver is its own 1-rank MPI job step)")
    a = p.parse_args(argv)
    queue = Path(a.queue)
    for d in ("requests", "done", "results", "drivers"):
        (queue / d).mkdir(parents=True, exist_ok=True)
    drivers = {lib: Driver(lib, Path(a.bin) / f"sma_{lib}", Path(a.system), queue, a.launcher.split()) for lib in LIBS}
    last_activity = time.time()
    state = "idle"

    def heartbeat():
        write_json(queue / "heartbeat.json", {
            "host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"), "pid": os.getpid(),
            "time": time.time(), "state": state, "idle_timeout": a.idle_timeout,
            "drivers_alive": [l for l, d in drivers.items() if d.alive()]})

    print(f"worker up on {socket.gethostname()}, queue {queue}", flush=True)
    try:
        while True:
            heartbeat()
            if (queue / "STOP").exists():
                print("STOP file found", flush=True)
                break
            reqs = sorted((queue / "requests").glob("*.req"), key=lambda q: q.stat().st_mtime)
            if not reqs:
                if time.time() - last_activity > a.idle_timeout:
                    print(f"idle for {a.idle_timeout} s: exiting", flush=True)
                    break
                time.sleep(1.0)
                continue
            req = reqs[0]
            kv = read_kv(req)
            lib, out = kv.get("lib"), Path(kv.get("out", queue / "results" / (req.stem + ".json")))
            state = f"running {req.stem} ({lib})"
            heartbeat()
            t0 = time.time()
            try:
                if lib not in drivers:
                    raise RuntimeError(f"unknown lib {lib!r}")
                d = drivers[lib]
                if not d.alive():
                    d.start()
                done = d.request(req, float(kv.get("timeout", 900)))
                if done is None:
                    tail = "\n".join(d.stderr_tail)
                    d.stop()  # timed out or crashed: restart on next request
                    raise RuntimeError(f"{lib} driver crashed or timed out after {time.time() - t0:.0f} s; "
                                       f"stderr tail:\n{tail}")
            except Exception as e:
                write_json(out, {"id": kv.get("id", req.stem), "lib": lib, "converged": False,
                                 "error": str(e)})
            os.replace(req, queue / "done" / req.name)
            last_activity = time.time()
            state = "idle"
    finally:
        for d in drivers.values():
            d.stop()
        state = "exited"
        heartbeat()


if __name__ == "__main__":
    sys.exit(main())
