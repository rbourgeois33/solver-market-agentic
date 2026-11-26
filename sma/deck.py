"""Input decks: YAML using each library's own parameter names.

  lib: hypre | muelu | petsc
  solver: <Krylov method, native name>
      hypre: pcg | gmres | fgmres | bicgstab
      muelu: a Belos solver name, e.g. "Pseudoblock CG", "Pseudoblock GMRES"
      petsc: a KSP type, e.g. cg, gmres, fgmres, bcgs
  solver_params:   {native Krylov parameters}       (optional)
  precond_params:  {native AMG parameters}          (optional)
  backend_params:  {library-wide GPU settings}      (optional)

Only keys present are passed; everything else keeps the library default.
Native names:
  hypre   solver_params:  suffix of HYPRE_ParCSR<Solver>Set<Name>  (Tol, MaxIter, KDim, TwoNorm, ...)
          precond_params: suffix of HYPRE_BoomerAMGSet<Name>        (CoarsenType, InterpType, ...)
                          two-argument setters use "Name:k", e.g. "CycleRelaxType:3": 9
          backend_params: SpGemmUseVendor, SpMVUseVendor, UseGpuRand
  muelu   solver_params:  Belos ParameterList keys ("Convergence Tolerance", "Num Blocks", ...)
          precond_params: MueLu ParameterList keys ("smoother: type", "aggregation: drop tol", ...);
                          nested dicts become sublists. Write reals as 0.0, not 0 (MueLu checks types).
                          MueLu keys contain ": " so they MUST be quoted in YAML: "smoother: type": CHEBYSHEV
  petsc   solver_params / precond_params / backend_params: PETSc option names without the leading
          dash (ksp_rtol, pc_type, pc_gamg_threshold, mg_levels_ksp_type, mat_type, ...).

The campaign tolerance is on the TRUE relative residual; if the deck gives no internal tolerance,
the campaign tolerance is used for the library's own stopping test.
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import quoteattr

import yaml

LIBS = ("hypre", "muelu", "petsc")
SECTIONS = ("solver_params", "precond_params", "backend_params")
HYPRE_SOLVERS = ("pcg", "gmres", "fgmres", "bicgstab")


class DeckError(ValueError):
    pass


def load(path: str | Path) -> dict:
    d = yaml.safe_load(Path(path).read_text())
    return validate(d)


def validate(d: dict) -> dict:
    if not isinstance(d, dict):
        raise DeckError("deck must be a mapping")
    extra = set(d) - {"lib", "solver", *SECTIONS, "comment"}
    if extra:
        raise DeckError(f"unknown top-level keys: {sorted(extra)}")
    if d.get("lib") not in LIBS:
        raise DeckError(f"lib must be one of {LIBS}")
    if not isinstance(d.get("solver"), str):
        raise DeckError("solver (native Krylov name) is required")
    if d["lib"] == "hypre" and d["solver"] not in HYPRE_SOLVERS:
        raise DeckError(f"hypre solver must be one of {HYPRE_SOLVERS}")
    for s in SECTIONS:
        d.setdefault(s, {})
        if d[s] is None:
            d[s] = {}
        if not isinstance(d[s], dict):
            raise DeckError(f"{s} must be a mapping")
    if d["lib"] == "muelu" and d["backend_params"]:
        raise DeckError("muelu has no backend_params (Kokkos/cuSPARSE choices are build-time)")
    return d


# ---------------------------------------------------------------- per-library translation
def _scalar(v) -> str:
    """hypre value: bool -> 1/0, list -> comma-separated (array setters), else str."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (list, tuple)):
        return ",".join(_scalar(x) for x in v)
    s = str(v)
    if any(c.isspace() for c in s):
        raise SystemExit(f"hypre value {v!r} contains spaces")
    return s


def to_hypre(d: dict, tol: float) -> str:
    """'section key value' lines, parsed by the hypre driver's setter table."""
    sp = dict(d["solver_params"])
    sp.setdefault("Tol", tol)
    lines = [f"solver {d['solver']}"]
    lines += [f"solver_param {k} {_scalar(v)}" for k, v in sp.items()]
    lines += [f"precond_param {k} {_scalar(v)}" for k, v in d["precond_params"].items()]
    lines += [f"backend_param {k} {_scalar(v)}" for k, v in d["backend_params"].items()]
    return "\n".join(lines) + "\n"


def _xml_param(name: str, v, indent: str) -> str:
    if isinstance(v, dict):
        body = "".join(_xml_param(k, x, indent + "  ") for k, x in v.items())
        return f'{indent}<ParameterList name={quoteattr(name)}>\n{body}{indent}</ParameterList>\n'
    if isinstance(v, bool):
        t, s = "bool", "true" if v else "false"
    elif isinstance(v, int):
        t, s = "int", str(v)
    elif isinstance(v, float):
        t, s = "double", repr(v)
    else:
        t, s = "string", str(v)
    return f'{indent}<Parameter name={quoteattr(name)} type="{t}" value={quoteattr(s)}/>\n'


def _xml_list(name: str, params: dict) -> str:
    return _xml_param(name, params, "")


def to_muelu(d: dict, tol: float) -> tuple[str, str, str]:
    """(belos solver name, Belos XML, MueLu XML)."""
    sp = dict(d["solver_params"])
    sp.setdefault("Convergence Tolerance", float(tol))
    return d["solver"], _xml_list("Belos", sp), _xml_list("MueLu", d["precond_params"])


def to_petsc(d: dict, tol: float) -> str:
    """PETSc options string for PetscOptionsInsertString (prefix-free)."""
    opts = {"ksp_type": d["solver"]}
    opts.update(d["backend_params"])
    opts.update(d["solver_params"])
    opts.setdefault("ksp_rtol", tol)
    opts.setdefault("pc_type", "gamg")
    opts.update(d["precond_params"])
    parts = []
    for k, v in opts.items():
        k = k.lstrip("-")
        if v is None or v is True:
            parts.append(f"-{k}")
        elif v is False:
            parts.append(f"-{k} 0")
        else:
            parts.append(f"-{k} {v}")
    return " ".join(parts)


def translate(d: dict, tol: float, out_dir: Path) -> dict:
    """Write the native inputs for the driver into out_dir; return the request fields."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = d["lib"]
    if lib == "hypre":
        (out_dir / "hypre.cfg").write_text(to_hypre(d, tol))
        return {"lib": lib, "config": str(out_dir / "hypre.cfg")}
    if lib == "muelu":
        name, belos, muelu = to_muelu(d, tol)
        (out_dir / "belos.xml").write_text(belos)
        (out_dir / "muelu.xml").write_text(muelu)
        return {"lib": lib, "solver": name, "belos_xml": str(out_dir / "belos.xml"),
                "muelu_xml": str(out_dir / "muelu.xml")}
    opts = to_petsc(d, tol)
    (out_dir / "petsc.opts").write_text(opts)  # no trailing newline (PETSc splits on spaces only)
    return {"lib": lib, "config": str(out_dir / "petsc.opts")}
