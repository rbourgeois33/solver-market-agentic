"""sma-tools build: libraries (spack, cache reuse) + the three drivers + parameter catalogues.

Cluster (mode slurm, machines/<machine>/):
  1. concretize.slurm  on a node with internet: render the env, concretize, fetch missing sources
  2. build_libs_gpu.slurm  on a GPU node (after 1 succeeded): spack install, drivers, catalogues
  The command waits for both jobs and reports the outcome; job logs are in $SMA_WORK/logs.
Local (mode local, a machine with a GPU and internet): the same steps in this process.

spack reuses what is installed: a build with nothing to change is quick. --rebuild (uninstall the
GPU stack first, ~1 h) needs SMA_ALLOW_REBUILD=1 in the environment: the agent cannot trigger it.
Every build is appended to $SMA_WORK/env/build_log.md.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from . import REPO_DIR, sma_work

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}


def _log(entry: str) -> None:
    p = sma_work() / "env" / "build_log.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M')} sma-tools build\n{entry}\n")


def _sbatch(script: Path, *args: str, dependency: str | None = None) -> str:
    cmd = ["sbatch", "--parsable"] + ([f"--dependency=afterok:{dependency}"] if dependency else []) + \
          [str(script), *args]
    out = subprocess.run(cmd, cwd=REPO_DIR, check=True, capture_output=True, text=True).stdout
    return out.strip().split(";")[0]


def _state(job: str) -> str:
    out = subprocess.run(["sacct", "-j", job, "-X", "-n", "--format=State"], capture_output=True, text=True).stdout
    s = out.split()
    return s[0].rstrip("+") if s else "PENDING"


def _wait(jobs: list[str], poll: float = 30.0) -> dict[str, str]:
    states = {}
    while True:
        states = {j: _state(j) for j in jobs}
        # a dependent job whose dependency failed stays PENDING (DependencyNeverSatisfied): stop then
        if any(states[j] in TERMINAL and states[j] != "COMPLETED" for j in jobs) or \
                all(s in TERMINAL for s in states.values()):
            break
        time.sleep(poll)
    for j, s in states.items():
        if s not in TERMINAL:
            subprocess.run(["scancel", j])
            states[j] = "CANCELLED (dependency failed)"
    return states


def _tail(path: Path, n: int = 25) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return f"(no log {path})"


def build_slurm(machine: str, rebuild: bool) -> int:
    mdir = REPO_DIR / "machines" / machine
    logs = sma_work() / "logs"
    j1 = _sbatch(mdir / "concretize.slurm")
    j2 = _sbatch(mdir / "build_libs_gpu.slurm", *(["--rebuild"] if rebuild else []), dependency=j1)
    print(f"submitted: concretize+fetch job {j1}, GPU build job {j2} (waiting; logs in {logs})", flush=True)
    states = _wait([j1, j2])
    ok = all(s == "COMPLETED" for s in states.values())
    l1 = next(logs.glob(f"*-{j1}.out"), logs / f"?-{j1}.out")
    l2 = next(logs.glob(f"*-{j2}.out"), logs / f"?-{j2}.out")
    gpu_tail = _tail(l2)  # the GPU job exits non-zero if install, drivers or catalogues fail
    report = (f"jobs: {j1} {states[j1]}, {j2} {states[j2]} (rebuild={rebuild})\n"
              f"concretize log: {l1}\nGPU build log: {l2}\n--- tail of GPU build log ---\n{gpu_tail}")
    print(report)
    _log(("OK" if ok else "FAILED") + "\n```\n" + report + "\n```")
    return 0 if ok else 1


def build_local(machine: str, rebuild: bool) -> int:
    from .spackenv import render
    steps = []

    def run(cmd: str) -> int:
        print("+", cmd, flush=True)
        rc = subprocess.run(["bash", "-c", cmd], cwd=REPO_DIR).returncode
        steps.append(f"{cmd} -> rc={rc}")
        return rc

    render(machine, None, os.cpu_count() or 8, True)
    env = f"spack env activate {sma_work() / 'env'} &&"
    rc = run(f"{env} spack concretize -f") or run(f"{env} spack fetch -m")
    if rc == 0 and rebuild:
        rc = run(f"{env} spack uninstall -y -a --dependents kokkos kokkos-kernels umpire camp hypre petsc trilinos")
    rc = rc or run(f"{env} spack install --fail-fast") or run("drivers/build.sh") or run("sma-tools catalog")
    _log(("OK" if rc == 0 else "FAILED") + "\n```\n" + "\n".join(steps) + "\n```")
    return rc


def build(mode: str, machine: str, rebuild: bool) -> int:
    if rebuild and os.environ.get("SMA_ALLOW_REBUILD") != "1":
        raise SystemExit("--rebuild uninstalls the GPU stack (~1 h): set SMA_ALLOW_REBUILD=1 to confirm")
    return build_slurm(machine, rebuild) if mode == "slurm" else build_local(machine, rebuild)
