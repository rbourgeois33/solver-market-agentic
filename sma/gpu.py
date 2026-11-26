"""GPU-only rule: classify a deck's option values as gpu / host_only / unknown (sma/gpu_support.yaml),
plus ruled-out methods (clearly bad on GPU) and host setup options (refused when k = 0).

`sma-tools run` refuses a deck with a host_only or ruled_out finding (not charged to the budget). The coarsest-
level solve is exempt (it may run on the host). Rules apply to EFFECTIVE values, so library defaults
that run on the host (e.g. MueLu's default Symmetric Gauss-Seidel smoother) are caught too.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml


@lru_cache
def rules() -> dict:
    return yaml.safe_load((Path(__file__).with_name("gpu_support.yaml")).read_text())


def _f(key, value, status, source, note=""):
    return {"key": key, "value": value, "status": status, "source": source, "note": note}


def _value_rule(key, value, r) -> dict:
    v = value if not isinstance(value, str) else value.strip()
    if v in r.get("gpu", []):
        st = "gpu"
    elif v in r.get("host_only", []):
        st = "host_only"
    else:
        st = r.get("otherwise", "unknown")
    return _f(key, value, st, r.get("source", ""), r.get("note", ""))


# ------------------------------------------------------------------ hypre
def _hypre(deck: dict) -> list[dict]:
    R = rules()["hypre"]
    out = []
    for key, value in deck.get("precond_params", {}).items():
        if key in R["exempt"]:
            out.append(_f(key, value, "exempt", "coarsest level: host or GPU allowed"))
            continue
        r = R["rules"].get(key.split(":")[0])
        if r:
            out.append(_value_rule(key, value, r))
    return out


# ------------------------------------------------------------------ petsc
def _petsc(deck: dict) -> list[dict]:
    R = rules()["petsc"]
    opts = {**deck.get("solver_params", {}), **deck.get("precond_params", {}), **deck.get("backend_params", {})}
    opts.setdefault("ksp_type", deck.get("solver"))
    out = []
    for key, value in opts.items():
        if key.startswith(R["exempt_prefix"]):
            out.append(_f(key, value, "exempt", "coarsest level: host or GPU allowed"))
            continue
        m = re.match(r"^((?:mg_levels_(?:\d+_)?)?)(mat_type|vec_type|pc_type|ksp_type)$", key)
        if not m:
            continue
        prefix, base = m.groups()
        r = R["rules"][base]
        if base == "pc_type" and value in r.get("factor", []):
            solver = opts.get(f"{prefix}pc_factor_mat_solver_type")
            st = "gpu" if solver in R["factor_gpu_solvers"] else "host_only"
            out.append(_f(key, value, st, r["source"],
                          f"factorization on the GPU needs {prefix}pc_factor_mat_solver_type "
                          f"{' or '.join(R['factor_gpu_solvers'])} (given: {solver})"))
        else:
            out.append(_value_rule(key, value, r))
    return out


# ------------------------------------------------------------------ muelu
def _smoother_defs(pl: dict, where: str):
    """(location, type, params) for every smoother defined in a MueLu parameter list level."""
    found = False
    for side in ("", "pre ", "post "):
        t = pl.get(f"smoother: {side}type")
        if t is not None:
            found = True
            yield f"{where}smoother: {side}type", t, pl.get(f"smoother: {side}params", {}) or \
                pl.get("smoother: params", {})
    if not found and where == "":
        yield "smoother: type (default)", rules()["muelu"]["smoother_type_default"], pl.get("smoother: params", {})


def _muelu(deck: dict) -> list[dict]:
    R = rules()["muelu"]
    pl = deck.get("precond_params", {})
    levels = [("", pl)] + [(f"{k} / ", v) for k, v in pl.items() if k.startswith("level ") and isinstance(v, dict)]
    out = []
    for where, lv in levels:
        for loc, stype, params in _smoother_defs(lv, where):
            st = R["smoother_types"].get(str(stype).upper())
            if st is None:
                out.append(_f(loc, stype, "unknown", "no rule for this smoother type"))
            elif st["status"] != "depends":
                out.append(_f(loc, stype, st["status"], st["source"]))
            elif st["depends_on"] == "relaxation: type":
                rt = params.get("relaxation: type")
                src = R["relaxation_types_source"]
                if rt is None:
                    rt, src = R["relaxation_type_default"], R["relaxation_type_default_source"] + "; " + src
                out.append(_f(f"{loc} / relaxation: type", rt, R["relaxation_types"].get(rt, "unknown"), src))
            else:  # RILUK
                ft = params.get("fact: type", R["fact_type_default"])
                out.append(_f(f"{loc} / fact: type", ft, R["fact_types"].get(ft, "unknown"), R["fact_types_source"]))
    for k, v in pl.items():
        if k.startswith(R["exempt_prefix"]):
            out.append(_f(k, v, "exempt", "coarsest level: host or GPU allowed"))
    return out


def _walk(d: dict, path: str = ""):
    """(location, key, value) for every leaf of a nested parameter dict."""
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _walk(v, f"{path}{k} / ")
        else:
            yield f"{path}{k}", k, v


def _extra(deck: dict, setup_measured: bool) -> list[dict]:
    """Ruled-out methods, and host setup options when the setup is not measured."""
    R, lib, out = rules(), deck["lib"], []
    params = {**deck.get("precond_params", {}), **deck.get("solver_params", {})}
    for loc, k, v in _walk(params):
        r = R.get("ruled_out", {}).get(lib, {}).get(k)
        if r and v in r["values"]:
            out.append(_f(loc, v, "ruled_out", r["evidence"], "ruled out: clearly bad on GPU"))
        r = R.get("setup_rules", {}).get(lib, {}).get(k)
        if r and v in r.get("host_only", []):
            out.append(_f(loc, v, "allowed_setup_cost" if setup_measured else "host_only", r["source"],
                          "setup on the host" + ("" if setup_measured else
                                                 ": refused while the setup is not measured (k = 0)")))
    return out


REFUSED = ("host_only", "ruled_out")


def classify(deck: dict, setup_measured: bool = False) -> list[dict]:
    base = {"hypre": _hypre, "petsc": _petsc, "muelu": _muelu}[deck["lib"]](deck)
    # smoother values already classified: mark ruled-out ones
    ro = rules().get("ruled_out", {}).get(deck["lib"], {})
    for f in base:
        key = f["key"].split(" / ")[-1]
        if key in ro and f["value"] in ro[key]["values"]:
            f.update(status="ruled_out", source=ro[key]["evidence"], note="ruled out: clearly bad on GPU")
    seen = {(f["key"], str(f["value"])) for f in base}
    return base + [f for f in _extra(deck, setup_measured) if (f["key"], str(f["value"])) not in seen
                   and not any(f["key"].endswith(b["key"].split(" / ")[-1]) and str(f["value"]) == str(b["value"])
                               for b in base)]


def refused(deck: dict, setup_measured: bool = False) -> list[dict]:
    return [f for f in classify(deck, setup_measured) if f["status"] in REFUSED]


def host_only(deck: dict) -> list[dict]:  # kept for callers that only need the GPU rule
    return [f for f in classify(deck) if f["status"] == "host_only"]


def show(lib: str) -> str:
    """Human-readable rules for one library (sma-tools params LIB --gpu): GPU support of values,
    host setup options refused when the setup is not measured, ruled-out methods."""
    R = rules()
    d = {lib: R[lib]}
    if lib in R.get("setup_rules", {}):
        d["setup_on_host (refused when k = 0)"] = R["setup_rules"][lib]
    if lib in R.get("ruled_out", {}):
        d["ruled_out (clearly bad on GPU)"] = R["ruled_out"][lib]
    return yaml.safe_dump(d, sort_keys=False, width=110, allow_unicode=True)
