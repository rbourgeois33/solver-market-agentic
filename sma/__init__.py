"""solver-market-agentic toolbox."""
import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent


def sma_work() -> Path:
    """Root of all generated state (spack, installs, data, campaigns)."""
    w = os.environ.get("SMA_WORK")
    if not w:
        raise SystemExit("SMA_WORK is not set (source machines/<machine>/env.sh)")
    return Path(w)
