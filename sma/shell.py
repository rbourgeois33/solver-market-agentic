"""sma-tools shell 'SCRIPT': run a shell script (loops, sed, python ...) confined by bubblewrap.

The whole filesystem is read-only except DIR/decks and DIR/work of the campaign; /tmp is private
and discarded; no network. The record (journal, runs, queue) is therefore out of reach, and nothing
outside the campaign can change. Benchmarks cannot run in here (the journal is read-only): use
`sma-tools run` directly. Typical use: generate deck variants in a loop.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

WRITABLE = ("decks", "work")


def run(campaign_dir: Path, script: str, timeout: float = 600) -> int:
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise SystemExit("sma-tools shell needs bubblewrap (bwrap), not found on this machine")
    binds = []
    for d in WRITABLE:
        p = campaign_dir / d
        p.mkdir(exist_ok=True)
        binds += ["--bind", str(p), str(p)]
    cmd = [bwrap, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", *binds,
           "--unshare-net", "--unshare-pid", "--die-with-parent", "--chdir", str(campaign_dir),
           "bash", "-c", script]
    try:
        return subprocess.run(cmd, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"sma-tools shell: timed out after {timeout:.0f} s")
        return 124
