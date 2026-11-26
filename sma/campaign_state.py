"""Campaign directory: parameters, budget, journal, request submission and result scoring.

  <campaign>/campaign.json   parameters (fixed at init)
  <campaign>/journal.jsonl   one line per benchmarked configuration (the ONLY record of attempts)
  <campaign>/reasoning.md    free-form reasoning kept by the agent
  <campaign>/runs/<id>/      deck.yaml + native inputs + result.json
  <campaign>/queue/          worker queue (see sma/worker/dispatcher.py)
"""
from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from pathlib import Path

import yaml

from . import deck as deckmod

KRYLOV_FAMILIES = {
    # family -> per-lib solver names (lower-case match)
    "cg": {"hypre": {"pcg"}, "muelu": {"cg", "pseudoblock cg", "block cg", "pseudoblock stochastic cg"},
           "petsc": {"cg", "pipecg", "pipecgrr", "groppcg", "nash", "stcg", "gltr"}},
    "gmres": {"hypre": {"gmres", "fgmres"},
              "muelu": {"gmres", "pseudoblock gmres", "block gmres", "flexible gmres", "gcrodr"},
              "petsc": {"gmres", "fgmres", "lgmres", "dgmres", "pgmres", "pipefgmres", "gcr"}},
    "bicgstab": {"hypre": {"bicgstab"}, "muelu": {"bicgstab"}, "petsc": {"bcgs", "ibcgs", "fbcgs", "bcgsl"}},
}


def amg_family(d: dict) -> str:
    """'classical' (C/F selection) or 'aggregation'."""
    if d["lib"] == "hypre":
        return "classical"
    if d["lib"] == "muelu":
        return "aggregation"
    t = str(d["precond_params"].get("pc_gamg_type", "agg")).lower()
    return "classical" if t == "classical" else "aggregation"


def krylov_family(d: dict) -> str | None:
    s = d["solver"].lower()
    for fam, per_lib in KRYLOV_FAMILIES.items():
        if s in per_lib[d["lib"]]:
            return fam
    return None


class Campaign:
    def __init__(self, path: str | Path):
        self.dir = Path(path).resolve()
        self.params = json.loads((self.dir / "campaign.json").read_text())

    # ------------------------------------------------------------ creation
    @staticmethod
    def init(path: str | Path, system: str, budget: int, tol: float, solves_per_setup: int,
             libs: list[str], krylov: list[str], amg: list[str], mem_limit_gb: float | None,
             repeats: int, warmup: int) -> "Campaign":
        d = Path(path).resolve()
        if (d / "campaign.json").exists():
            raise SystemExit(f"{d} already holds a campaign")
        d.mkdir(parents=True, exist_ok=True)
        (d / "runs").mkdir(exist_ok=True)
        (d / "queue").mkdir(exist_ok=True)
        sysdir = Path(system).resolve()
        if not (sysdir / "A.csr").exists():
            raise SystemExit(f"{sysdir} is not a converted system (run sma-tools convert)")
        params = {
            "system": str(sysdir), "budget": budget, "tol": tol, "solves_per_setup": solves_per_setup,
            "libs": libs, "krylov": krylov, "amg": amg,
            "mem_limit_bytes": int(mem_limit_gb * 2**30) if mem_limit_gb else None,
            "repeats": repeats, "warmup": warmup, "created": time.time(),
        }
        (d / "campaign.json").write_text(json.dumps(params, indent=2) + "\n")
        (d / "journal.jsonl").touch()
        if not (d / "reasoning.md").exists():
            (d / "reasoning.md").write_text("# Reasoning log\n\n")
        return Campaign(d)

    # ------------------------------------------------------------ journal / budget
    def journal(self) -> list[dict]:
        p = self.dir / "journal.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    def used(self) -> int:
        """Budget used: configurations that actually ran. Infrastructure failures (driver could not
        start, worker down / no answer) are journaled with charged=false and cost nothing."""
        return sum(1 for e in self.journal() if e.get("charged", True))

    def remaining(self) -> int:
        return self.params["budget"] - self.used()

    def cost(self, res: dict) -> float | None:
        """Objective (seconds per solve). k = solves per setup; k = 0 -> solve time only."""
        ts, tv = res.get("t_setup_median"), res.get("t_solve_median")
        if ts is None or tv is None:
            return None
        k = self.params["solves_per_setup"]
        return tv if k == 0 else (ts + k * tv) / k

    def best(self) -> dict | None:
        ok = [e for e in self.journal() if e.get("valid")]
        return min(ok, key=lambda e: e["cost"]) if ok else None

    # ------------------------------------------------------------ one benchmark
    def check_deck(self, d: dict) -> None:
        p = self.params
        if d["lib"] not in p["libs"]:
            raise deckmod.DeckError(f"lib {d['lib']} not allowed in this campaign (allowed: {p['libs']})")
        fam = krylov_family(d)
        if fam is None:
            raise deckmod.DeckError(f"solver {d['solver']!r} is not a known {d['lib']} Krylov method "
                                    f"(families: {sorted(KRYLOV_FAMILIES)})")
        if fam not in p["krylov"]:
            raise deckmod.DeckError(f"Krylov family {fam} not allowed (allowed: {p['krylov']})")
        af = amg_family(d)
        if af not in p["amg"]:
            raise deckmod.DeckError(f"AMG family {af} not allowed (allowed: {p['amg']})")

    def submit_and_wait(self, deck_path: str, why: str, timeout: float = 900,
                        poll: float = 2.0) -> dict:
        # One benchmark at a time per campaign: makes the budget check race-free and matches the
        # worker, which never runs two timings concurrently.
        with open(self.dir / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._submit_and_wait(deck_path, why, timeout, poll)

    def _submit_and_wait(self, deck_path: str, why: str, timeout: float, poll: float) -> dict:
        if not why or len(why.strip()) < 10:
            raise SystemExit("--why is required: state the hypothesis this configuration tests")
        d = deckmod.load(deck_path)
        self.check_deck(d)
        if self.remaining() <= 0:
            raise SystemExit(f"budget exhausted ({self.params['budget']} configurations used)")
        rid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        rdir = self.dir / "runs" / rid
        rdir.mkdir(parents=True)
        (rdir / "deck.yaml").write_text(Path(deck_path).read_text())
        fields = deckmod.translate(d, self.params["tol"], rdir)
        q = self.dir / "queue"
        for sub in ("requests", "results"):
            (q / sub).mkdir(parents=True, exist_ok=True)
        result_path = rdir / "result.json"
        lines = {"id": rid, "out": str(result_path), "tol": self.params["tol"],
                 "repeats": self.params["repeats"], "warmup": self.params["warmup"],
                 "timeout": int(timeout), **fields}
        tmp = q / "requests" / f".{rid}.tmp"
        tmp.write_text("".join(f"{k} {v}\n" for k, v in lines.items()))
        os.replace(tmp, q / "requests" / f"{rid}.req")
        # Budget: a configuration that ran counts even if it fails (diverges, crashes, times out);
        # infrastructure failures (infra_error) do not.
        entry = {"id": rid, "submitted": time.time(), "why": why.strip(), "deck": d}
        hb = q / "heartbeat.json"
        t_end = time.time() + timeout + 120
        while not result_path.exists():
            if time.time() > t_end:
                res = {"id": rid, "converged": False, "infra_error": True,
                       "error": "no result: worker down or request timed out"}
                break
            if hb.exists() and json.loads(hb.read_text()).get("state") == "exited":
                res = {"id": rid, "converged": False, "infra_error": True,
                       "error": "worker exited (start it with sma-tools worker start)"}
                break
            time.sleep(poll)
        else:
            res = json.loads(result_path.read_text())
        entry["result"] = res
        entry["charged"] = not res.get("infra_error", False)
        entry["cost"] = self.cost(res)
        mem_ok = True
        lim = self.params["mem_limit_bytes"]
        if lim and res.get("gpu_mem_peak_bytes") is not None:
            mem_ok = res["gpu_mem_peak_bytes"] <= lim
        entry["mem_ok"] = mem_ok
        entry["valid"] = bool(res.get("converged")) and mem_ok and entry["cost"] is not None
        with open(self.dir / "journal.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry


def short(entry: dict, campaign: Campaign) -> dict:
    """Compact view printed to the agent after each run."""
    r = entry["result"]
    keys = ("converged", "lib_converged", "iterations", "true_relres", "t_setup_median", "t_solve_median",
            "gpu_mem_peak_bytes", "operator_complexity", "grid_complexity", "levels", "error",
            "unused_options", "converged_reason", "driver_load")
    out = {"id": entry["id"], "valid": entry["valid"], "charged": entry.get("charged", True),
           "cost_s": entry["cost"], "mem_ok": entry["mem_ok"]}
    out.update({k: r[k] for k in keys if k in r and r[k] not in ("", None)})
    err = out.get("error")
    if err and len(err) > 1200:  # e.g. MueLu dumps all valid parameters; full text stays in result.json
        out["error"] = err[:500] + "\n[... truncated, see runs/<id>/result.json ...]\n" + err[-600:]
    best = campaign.best()
    out["best_so_far"] = {"id": best["id"], "cost_s": best["cost"]} if best else None
    out["budget_remaining"] = campaign.remaining()
    return out
