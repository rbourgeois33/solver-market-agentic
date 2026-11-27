"""Public dataset (Hugging Face layout): matrices + every configuration tested on them, per GPU, and the
best one per (matrix, GPU, k, tolerance). No hypotheses, comments or reasoning: decks and measurements only.

Layout of DATASET_DIR (a Hugging Face dataset repository):
  README.md                          dataset card (YAML configs for the viewer + leaderboard), generated
  matrices.jsonl                     one row per matrix: name, n, nnz, origin, files, sha256, properties
  runs.jsonl                         one row per configuration run (all campaigns, valid or not)
  best.jsonl                         best valid configuration per (matrix, gpu, k, tol)
  matrices/<name>/A.mtx.gz, b.mtx.gz (or rhs "ones"), A.csr + b.vec (binary CSR, sma/csr_io.py),
                                     info.json (provenance + analysis)
  best/<name>/<gpu>/k<k>_tol<tol>.yaml   the best deck, runnable with sma-tools run

Commands: `sma-tools dataset add-matrix SYSTEM --to DIR [--name N --origin TEXT]` (compresses the
matrix: run it on a compute node), `sma-tools dataset add CAMPAIGN... --to DIR` (runs only: light),
`sma-tools dataset build DIR` (best.jsonl, best decks, README.md). `sma campaign --dataset DIR` adds the
campaign's runs at the end.

Best = lowest cost among valid runs.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import re
import shutil
from pathlib import Path

import yaml

# Runs excluded because what they measured is not what their deck says (bugs fixed since).
# MueLu "half precision" was ignored by the driver before 2026-11-27 07:33 (driver rebuild, commit 6eda3aa).
HALF_PRECISION_FIX = dt.datetime(2026, 11, 27, 7, 33)

RUN_FIELDS = ["matrix", "gpu", "k", "tol", "lib", "lib_version", "solver", "deck", "valid", "cost_s",
              "t_setup_median", "t_setup_min", "t_setup_max", "t_solve_median", "t_solve_min", "t_solve_max",
              "iterations", "true_relres", "gpu_mem_peak_bytes", "operator_complexity", "grid_complexity",
              "levels", "level_rows", "level_nnz", "error", "setup_repeats", "solve_repeats", "date", "run_id",
              "campaign", "sma_commit"]


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=False) + "\n" for r in rows))
    tmp.replace(p)


def _run_date(run_id: str) -> dt.datetime | None:
    m = re.match(r"(\d{8}-\d{6})", run_id)
    return dt.datetime.strptime(m.group(1), "%Y%m%d-%H%M%S") if m else None


def _clean_deck(deck: dict) -> dict:
    d = {k: v for k, v in deck.items() if k not in ("comment", "why", "description")}
    return d


def _matrix_name(system: Path, ds: Path) -> str:
    for m in _read_jsonl(ds / "matrices.jsonl"):
        if m.get("system_sha256", {}).get("A.csr") == json.loads((system / "system.json").read_text())["sha256"]["A.csr"]:
            return m["name"]
    return system.name


# ------------------------------------------------------------------ matrices
def _write_mtx_gz(A, out: Path, chunk: int = 4_000_000) -> None:
    """MatrixMarket coordinate (1-based), gzip, from a CSR matrix, written in chunks of entries."""
    import numpy as np
    coo = A.tocoo()
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as fo:
        fo.write(f"%%MatrixMarket matrix coordinate real general\n{A.shape[0]} {A.shape[1]} {A.nnz}\n".encode())
        for k in range(0, A.nnz, chunk):
            blk = np.column_stack((coo.row[k:k + chunk] + 1, coo.col[k:k + chunk] + 1, coo.data[k:k + chunk]))
            np.savetxt(fo, blk, fmt=("%d", "%d", "%.17g"))
    tmp.replace(out)


def add_matrix(system: str, ds_dir: str, name: str | None = None, origin: str = "",
               also_from: list[str] | None = None, blocks: str = "") -> str:
    """Package a converted system. MatrixMarket sources are gzipped as is; other sources (PETSc binary)
    are written as MatrixMarket from the converted CSR. Matrices are deduplicated on the CSR sha256: an
    existing entry with the same matrix only gets `also_from` appended."""
    system, ds = Path(system), Path(ds_dir)
    meta = json.loads((system / "system.json").read_text())
    ana = json.loads((system / "analysis.json").read_text()) if (system / "analysis.json").exists() else {}
    name = name or system.name
    for m in _read_jsonl(ds / "matrices.jsonl"):  # same matrix already in the dataset
        if m.get("system_sha256", {}).get("A.csr") == meta["sha256"]["A.csr"]:
            m["also_from"] = sorted(set(m.get("also_from", [])) | {name, *(also_from or [])} - {m["name"]})
            rows = [x if x["name"] != m["name"] else m for x in _read_jsonl(ds / "matrices.jsonl")]
            _write_jsonl(ds / "matrices.jsonl", rows)
            return m["name"]
    mdir = ds / "matrices" / name
    mdir.mkdir(parents=True, exist_ok=True)
    src = Path(meta["source_matrix"])
    files = {}
    for f in ("A.csr", "b.vec"):  # binary CSR, fast to load (format: sma/csr_io.py)
        if not (mdir / f).exists():
            shutil.copyfile(system / f, mdir / f".{f}.tmp")
            (mdir / f".{f}.tmp").replace(mdir / f)
        files[f"{f[0]}_binary"] = f"matrices/{name}/{f}"
    if src.suffix == ".mtx":
        out = mdir / "A.mtx.gz"
        if not out.exists():
            with open(src, "rb") as fi, gzip.open(out.with_suffix(".tmp"), "wb", compresslevel=6) as fo:
                shutil.copyfileobj(fi, fo, 64 << 20)
            out.with_suffix(".tmp").replace(out)
        files["A"] = f"matrices/{name}/A.mtx.gz"
    else:
        from .csr_io import read_csr
        out = mdir / "A.mtx.gz"
        if not out.exists():
            _write_mtx_gz(read_csr(system / "A.csr"), out)
        files["A"] = f"matrices/{name}/A.mtx.gz"
    rhs = meta.get("source_rhs", "ones")
    if rhs in ("ones", "zeros"):
        files["b"] = rhs
    else:
        out = mdir / "b.mtx.gz"
        if not out.exists():
            with open(rhs, "rb") as fi, gzip.open(out, "wb", compresslevel=6) as fo:
                shutil.copyfileobj(fi, fo, 64 << 20)
        files["b"] = f"matrices/{name}/b.mtx.gz"
    st = ana.get("structure", {})
    mtx_bytes = src.stat().st_size if src.suffix == ".mtx" else None
    info = {"name": name, "nrows": meta["nrows"], "nnz": meta["nnz"], "origin": origin, "files": files,
            "blocks": blocks, "also_from": sorted(set(also_from or []) - {name}),
            "source_file": src.name, "mtx_bytes": mtx_bytes, "mtx_gz_bytes": (mdir / "A.mtx.gz").stat().st_size,
            "system_sha256": meta["sha256"],
            "symmetric": ana.get("symmetry", {}).get("numerically_symmetric"),
            "positive_diagonal": ana.get("diagonal", {}).get("negative_diagonal") == 0 and
            ana.get("diagonal", {}).get("zero_diagonal") == 0,
            "offdiag_positive": ana.get("offdiag_signs", {}).get("offdiag_positive"),
            "nnz_per_row_mean": st.get("nnz_per_row", {}).get("mean"), "bandwidth": st.get("bandwidth")}
    (mdir / "info.json").write_text(json.dumps({**info, "analysis": ana}, indent=1) + "\n")
    rows = [m for m in _read_jsonl(ds / "matrices.jsonl") if m["name"] != name] + [info]
    _write_jsonl(ds / "matrices.jsonl", sorted(rows, key=lambda m: m["name"]))
    return name


# ------------------------------------------------------------------ runs
def add_campaign(campaign: str, ds_dir: str) -> tuple[int, int]:
    """Append a campaign's runs (dedup by run_id). Returns (added, excluded)."""
    cdir, ds = Path(campaign), Path(ds_dir)
    ds.mkdir(parents=True, exist_ok=True)
    params = json.loads((cdir / "campaign.json").read_text())
    system = Path(params["system"])
    mname = _matrix_name(system, ds)
    rows = _read_jsonl(ds / "runs.jsonl")
    seen = {r["run_id"] for r in rows}
    added = excluded = 0
    for e in _read_jsonl(cdir / "journal.jsonl"):
        r, deck = e.get("result", {}), e.get("deck", {})
        if e["id"] in seen or not e.get("charged", True) or r.get("infra_error"):
            continue
        when = _run_date(e["id"])
        pp = deck.get("precond_params") or {}
        if deck.get("lib") == "muelu" and pp.get("half precision") and when and when < HALF_PRECISION_FIX:
            excluded += 1
            continue
        row = {"matrix": mname, "gpu": r.get("gpu") or "unknown",
               "k": params["solves_per_setup"], "tol": params["tol"], "lib": deck.get("lib"),
               "lib_version": r.get("lib_version"), "solver": deck.get("solver"),
               "deck": yaml.safe_dump(_clean_deck(deck), sort_keys=False), "valid": bool(e.get("valid")),
               "cost_s": e.get("cost"),
               **{k: r.get(k) for k in ("t_setup_median", "t_setup_min", "t_setup_max", "t_solve_median",
                                        "t_solve_min", "t_solve_max", "iterations", "true_relres",
                                        "gpu_mem_peak_bytes", "operator_complexity", "grid_complexity",
                                        "levels", "level_rows", "level_nnz")},
               "error": (r.get("error") or "")[:2000],
               "setup_repeats": params.get("setup_repeats"), "solve_repeats": params.get("solve_repeats"),
               "date": when.isoformat() if when else None, "run_id": e["id"], "campaign": cdir.name,
               "sma_commit": e.get("sma_commit") or ""}
        rows.append({k: row.get(k) for k in RUN_FIELDS})
        seen.add(e["id"])
        added += 1
    _write_jsonl(ds / "runs.jsonl", rows)
    return added, excluded


# ------------------------------------------------------------------ best + card
def build(ds_dir: str) -> list[dict]:
    ds = Path(ds_dir)
    runs = [r for r in _read_jsonl(ds / "runs.jsonl") if r["valid"] and r["cost_s"]]
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault((r["matrix"], r["gpu"], r["k"], r["tol"]), []).append(r)
    best = []
    shutil.rmtree(ds / "best", ignore_errors=True)
    for (m, gpu, k, tol), rs in sorted(groups.items()):
        b = min(rs, key=lambda r: r["cost_s"])
        same = [r for r in rs if r["deck"] == b["deck"]]
        path = ds / "best" / m / _slug(gpu) / f"k{k}_tol{tol:g}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(b["deck"])
        best.append({"matrix": m, "gpu": gpu, "k": k, "tol": tol, "lib": b["lib"],
                     "solver": b["solver"], "cost_s": b["cost_s"], "t_setup_median": b["t_setup_median"],
                     "t_solve_median": b["t_solve_median"], "iterations": b["iterations"],
                     "repeats_of_this_deck": len(same),
                     "cost_s_all_repeats": sorted(r["cost_s"] for r in same),
                     "deck": b["deck"], "deck_file": str(path.relative_to(ds)), "run_id": b["run_id"],
                     "configs_tested": len([r for r in _read_jsonl(ds / "runs.jsonl")
                                            if (r["matrix"], r["gpu"], r["k"], r["tol"]) == (m, gpu, k, tol)])})
    _write_jsonl(ds / "best.jsonl", best)
    (ds / "README.md").write_text(_card(ds, best))
    return best


def _card(ds: Path, best: list[dict]) -> str:
    mats = _read_jsonl(ds / "matrices.jsonl")
    runs = _read_jsonl(ds / "runs.jsonl")
    gpus = sorted({r["gpu"] for r in runs})
    lines = [
        "---",
        "license: other",
        "pretty_name: GPU AMG solver configurations",
        "tags: [sparse-matrix, linear-solver, algebraic-multigrid, gpu, hpc, cfd]",
        "configs:",
        "- config_name: best",
        "  data_files: best.jsonl",
        "  default: true",
        "- config_name: runs",
        "  data_files: runs.jsonl",
        "- config_name: matrices",
        "  data_files: matrices.jsonl",
        "---",
        "",
        "# GPU AMG solver configurations",
        "",
        "Sparse linear systems from CFD codes, and for each GPU every Krylov + algebraic multigrid configuration",
        "that was benchmarked on them (hypre BoomerAMG, Trilinos MueLu, PETSc GAMG), with the fastest valid one",
        "per matrix, GPU, solves-per-setup k and tolerance.",
        "",
        "## Best configuration per matrix and GPU",
        "",
        "| matrix | GPU | k | tol | time per solve | library | solver | iterations | deck |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for b in best:
        lines.append(f"| {b['matrix']} | {b['gpu']} | {b['k']} | {b['tol']:g} | {b['cost_s'] * 1e3:.2f} ms | "
                     f"{b['lib']} | {b['solver']} | {b['iterations']} | [{Path(b['deck_file']).name}]"
                     f"({b['deck_file']}) |")
    lines += ["", "## Matrices", "", "| matrix | rows | nonzeros | MatrixMarket size | origin |", "|---|---|---|---|---|"]
    for m in mats:
        size = f"{m['mtx_bytes'] / 1e9:.2f} GB" if m.get("mtx_bytes") else f"{m.get('mtx_gz_bytes', 0) / 1e9:.2f} GB (gz)"
        lines.append(f"| {m['name']} | {m['nrows']:,} | {m['nnz']:,} | {size} | {m['origin']} |")
    lines += [
        "", f"{len(runs)} configurations in total, GPUs: {', '.join(gpus)}.", "",
        "## How a configuration is measured", "",
        "- The matrix and right-hand side are uploaded once to the GPU; then 1 untimed warm-up (setup + solve),",
        "  S timed setups and N timed solves from x0 = 0 (`setup_repeats`, `solve_repeats`); medians are used.",
        "- `cost_s` = (t_setup + k t_solve) / k with k = solves per setup; k = 0 means the solve time only.",
        "- `valid`: the TRUE relative residual ||b - A x|| / ||b||, recomputed in double on the host, is <= `tol`",
        "  for every timed solve (and peak GPU memory within the limit).",
        "- Only GPU options are used (host-only smoothers etc. are refused; the coarsest-level solve may run on",
        "  the host; host setup options only when k = 0).",
        "",
        "## Files", "",
        "- `best.jsonl`: best valid configuration per (matrix, gpu, k, tol); the deck is also in `best/`.",
        "- `runs.jsonl`: every configuration run, valid or not (`error` says why not); `deck` is the YAML input.",
        "- `matrices.jsonl` + `matrices/<name>/`: `A.mtx.gz` (MatrixMarket, gzip), `A.csr` + `b.vec` (binary CSR,",
        "  format in solver-market-agentic `sma/csr_io.py`), right-hand side, `info.json`.",
        "",
        "A deck uses each library's own parameter names; it runs with the `solver-market-agentic` tools",
        "(`sma-tools run DECK`). Library versions: see `lib_version` (hypre 3.2.1, Trilinos 17.3.0, PETSc 3.26.1).",
        "",
    ]
    return "\n".join(lines)
