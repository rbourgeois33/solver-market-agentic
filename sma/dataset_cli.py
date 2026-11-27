"""sma-dataset: the public dataset (matrices + every configuration run + best per GPU), separate from
campaigns. Bring your own matrices; nothing here generates them.

  sma-dataset add-matrix FILE --name NAME --origin TEXT [--blocks TEXT] [--rhs ones|zeros|FILE]
      FILE: MatrixMarket (.mtx) or PETSc binary (32/64-bit indices, a trailing Vec is ignored).
      Submits one CPU job: convert + analyze into $SMA_WORK/data/NAME, then package into the dataset.
  sma-dataset add-campaign CAMPAIGN_DIR...     add the configurations a campaign ran (light, login node)
  sma-dataset build                            regenerate best.jsonl, best/ and the card README.md
  sma-dataset publish [--repo OWNER/NAME]      upload to the Hugging Face dataset repo ($SMA_HF_REPO)
All take --dir DIR (default $SMA_WORK/dataset).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import REPO_DIR, sma_work
from . import dataset as D


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sma-dataset", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=None, help="dataset directory (default: $SMA_WORK/dataset)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("add-matrix", help="add your matrix (one CPU job: convert, analyze, package)")
    s.add_argument("file")
    s.add_argument("--name", required=True, help="dataset name of the matrix (your choice, unique)")
    s.add_argument("--origin", required=True, help="where it comes from: application, case, discretization...")
    s.add_argument("--blocks", default="", help="unknown blocks in row order, e.g. 'u: rows 0-999; p: rows 1000-1499'")
    s.add_argument("--rhs", default="ones", help="ones | zeros | rhs file (default ones)")
    s.add_argument("--here", action="store_true", help="run now in this process (inside the job)")
    s = sub.add_parser("add-campaign", help="add the configurations run by campaign(s)")
    s.add_argument("campaigns", nargs="+")
    sub.add_parser("build", help="regenerate best.jsonl, best/ and README.md")
    s = sub.add_parser("publish", help="upload to Hugging Face")
    s.add_argument("--repo", default=os.environ.get("SMA_HF_REPO"), help="OWNER/NAME (default $SMA_HF_REPO)")
    for sp in sub.choices.values():
        sp.add_argument("--dir", dest="subdir", default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    ds = a.subdir or a.dir or str(sma_work() / "dataset")

    if a.cmd == "add-matrix":
        f = str(Path(a.file).resolve())
        if a.here:
            print("matrix:", D.new_matrix(f, a.name, a.origin, a.blocks, a.rhs, ds))
            return 0
        script = REPO_DIR / "machines" / os.environ.get("SMA_MACHINE", "jean-zay") / "dataset_add_matrix.slurm"
        args = [f, "--name", a.name, "--origin", a.origin, "--rhs", a.rhs] + (["--blocks", a.blocks] if a.blocks else [])
        out = subprocess.run(["sbatch", "--parsable", f"--job-name=sma-mtx-{a.name}", str(script), *args,
                              "--dir", ds], cwd=REPO_DIR, capture_output=True, text=True)
        if out.returncode:
            raise SystemExit(out.stderr)
        print(f"{a.name}: job {out.stdout.strip()} (log: $SMA_WORK/logs/sma-mtx-{a.name}-<job>.out)")
    elif a.cmd == "add-campaign":
        for c in a.campaigns:
            n, x = D.add_campaign(c, ds)
            print(f"{c}: {n} runs added" + (f", {x} excluded (measured something else than their deck)" if x else ""))
        D.build(ds)
    elif a.cmd == "build":
        for b in D.build(ds):
            print(f"{b['matrix']} | {b['gpu']} | k={b['k']} tol={b['tol']:g} | {b['time_per_solve_ms']} ms, "
                  f"{b['iterations']} its, {b['lib']} ({b['configs_tested']} configs)")
    elif a.cmd == "publish":
        if not a.repo:
            raise SystemExit("--repo OWNER/NAME (or SMA_HF_REPO) is required")
        D.publish(ds, a.repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
