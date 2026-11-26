"""Start / stop / inspect the bench worker of a campaign.

mode "slurm": submits machines/<machine>/worker.slurm (exclusive GPU node), records the job id.
mode "local": runs the dispatcher as a detached local process (private machine with a GPU).
Stopping always writes queue/STOP, then scancel / kill as a hard stop, so no GPU job is orphaned.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .. import REPO_DIR, sma_work


def _info_path(campaign_dir: Path) -> Path:
    return campaign_dir / "queue" / "worker.json"


def start(campaign_dir: Path, system: str, mode: str, machine: str | None, time_limit: str,
          idle_timeout: int) -> dict:
    q = campaign_dir / "queue"
    q.mkdir(parents=True, exist_ok=True)
    (q / "STOP").unlink(missing_ok=True)
    (q / "heartbeat.json").unlink(missing_ok=True)
    bindir = sma_work() / "bin"
    cmd = [sys.executable, "-m", "sma.worker.dispatcher", "--system", system, "--queue", str(q),
           "--bin", str(bindir), "--idle-timeout", str(idle_timeout)]
    if mode == "slurm":
        # each driver = its own 1-rank MPI step inside the allocation
        cmd += ["--launcher", "srun --overlap --ntasks=1 --cpus-per-task=24 --gres=gpu:1"]
    if mode == "local":
        log = open(q / "worker.log", "a")
        p = subprocess.Popen(cmd, cwd=REPO_DIR, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        info = {"mode": "local", "pid": p.pid, "started": time.time()}
    else:
        script = REPO_DIR / "machines" / machine / "worker.slurm"
        out = subprocess.run(
            ["timeout", "100", "sbatch", "--parsable", f"--time={time_limit}", f"--output={q}/worker-%j.out",
             str(script), *cmd],
            cwd=REPO_DIR, capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(f"sbatch failed: {out.stderr.strip()}")
        info = {"mode": "slurm", "job_id": out.stdout.strip().split(";")[0], "started": time.time()}
    _info_path(campaign_dir).write_text(json.dumps(info, indent=2) + "\n")
    return info


def status(campaign_dir: Path) -> dict:
    ip = _info_path(campaign_dir)
    st = {"worker": json.loads(ip.read_text()) if ip.exists() else None}
    hb = campaign_dir / "queue" / "heartbeat.json"
    if hb.exists():
        h = json.loads(hb.read_text())
        h["age_s"] = round(time.time() - h["time"], 1)
        st["heartbeat"] = h
    w = st["worker"]
    if w and w["mode"] == "slurm":
        out = subprocess.run(["timeout", "100", "squeue", "-h", "-j", w["job_id"], "-o", "%T %R"],
                             capture_output=True, text=True)
        st["slurm_state"] = out.stdout.strip() or "not in queue"
    st["pending_requests"] = len(list((campaign_dir / "queue" / "requests").glob("*.req"))) \
        if (campaign_dir / "queue" / "requests").exists() else 0
    return st


def stop(campaign_dir: Path, grace: float = 60) -> dict:
    q = campaign_dir / "queue"
    q.mkdir(parents=True, exist_ok=True)
    (q / "STOP").touch()
    ip = _info_path(campaign_dir)
    if not ip.exists():
        return {"stopped": "no worker recorded"}
    w = json.loads(ip.read_text())
    t_end = time.time() + grace
    hb = q / "heartbeat.json"
    while time.time() < t_end:
        if hb.exists() and json.loads(hb.read_text()).get("state") == "exited":
            break
        if w["mode"] == "slurm" and not hb.exists():
            break  # job never started: cancel right away
        time.sleep(2)
    if w["mode"] == "slurm":
        subprocess.run(["timeout", "100", "scancel", w["job_id"]], capture_output=True)
        return {"stopped": f"slurm job {w['job_id']} (STOP file + scancel)"}
    try:
        os.killpg(w["pid"], signal.SIGTERM)
    except ProcessLookupError:
        pass
    return {"stopped": f"local pid {w['pid']}"}
