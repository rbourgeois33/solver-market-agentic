"""Render spack/spack.yaml.in into $SMA_WORK/env/spack.yaml for this machine."""
import os
import subprocess
from pathlib import Path

from . import REPO_DIR, sma_work


def detect_cuda_arch() -> str:
    out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.split()
    return out[0].replace(".", "")


def render(machine: str, cuda_arch: str | None, build_jobs: int, force: bool) -> Path:
    work = sma_work()
    env_dir = work / "env"
    target = env_dir / "spack.yaml"
    if target.exists() and not force:
        raise SystemExit(f"{target} exists (it may hold logged tweaks); use --force to overwrite")
    arch = cuda_arch or detect_cuda_arch()
    stage = os.environ.get("SMA_STAGE_DIR", str(work / "stage"))
    # Machine externals: spack does not expand env variables in external paths, so the
    # prefixes of the loaded modules (SMA_*_PREFIX, set by machines/<m>/env.sh) are written in.
    pk_text = (REPO_DIR / "machines" / machine / "packages.yaml.in").read_text()
    for key in ("GCC", "CUDA", "MPI", "CMAKE"):
        tag = f"@{key}_PREFIX@"
        if tag in pk_text:
            val = os.environ.get(f"SMA_{key}_PREFIX")
            if not val or not Path(val).is_dir():
                raise SystemExit(f"SMA_{key}_PREFIX={val!r} is not a directory (source machines/{machine}/env.sh)")
            pk_text = pk_text.replace(tag, val)
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "packages.yaml").write_text(pk_text)
    subs = {
        "@MACHINE_PACKAGES@": str(env_dir / "packages.yaml"),
        "@REPO_DIR@": str(REPO_DIR),
        "@CUDA_ARCH@": arch,
        "@SMA_WORK@": str(work),
        "@STAGE_DIR@": stage,
        "@BUILD_JOBS@": str(build_jobs),
    }
    text = (REPO_DIR / "spack" / "spack.yaml.in").read_text()
    for k, v in subs.items():
        text = text.replace(k, v)
    env_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    log = env_dir / "build_log.md"
    if not log.exists():
        log.write_text("# Build log\n\nEvery change to spack.yaml (or recipes) with its reason.\n\n"
                       f"- rendered from spack/spack.yaml.in, machine={machine}, cuda_arch={arch}\n")
    return target
