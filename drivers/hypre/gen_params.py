#!/usr/bin/env python3
"""Generate the hypre driver's parameter tables from the INSTALLED HYPRE_parcsr_ls.h.

Every HYPRE_BoomerAMGSet<Name>, HYPRE_{PCG,GMRES,FlexGMRES,LGMRES,COGMRES,BiCGSTAB}Set<Name> (generic Krylov,
HYPRE_krylov.h) and HYPRE_ParCSR{PCG,GMRES,FlexGMRES,LGMRES,COGMRES,BiCGSTAB}Set<Name> whose
arguments are plain values (or a value + level/cycle index, or a host array) becomes a deck
parameter <Name>. Outputs:
  hypre_params.inc   C++ table entries, included by hypre_driver.cpp
  hypre_params.json  catalogue for the agent (name, kind, C signature, the header's doc comment)

Kinds (deck syntax):
  int / real            Name: 3                    -> Set<Name>(solver, 3)
  int_k / real_k        "Name:k": 3                -> Set<Name>(solver, 3, k)     (level, cycle part)
  int_array/real_array  Name: [1, 2, 3]            -> Set<Name>(solver, array)    (hypre takes ownership)
  count_real_array      Name: [1e-2, 1e-3]         -> Set<Name>(solver, 2, array)
  flag                  Name: 1                    -> Set<Name>(solver)
Not exposed (listed in the json under "excluded", with the reason): per-row data, files, objects,
callbacks, and Setup.
usage: gen_params.py HYPRE_utilities.h HYPRE_krylov.h HYPRE_parcsr_ls.h OUT_DIR
"""
import json
import re
import sys
from pathlib import Path

SOLVERS = {"BoomerAMG": "amg", "ParCSRPCG": "pcg", "ParCSRGMRES": "gmres",
           "ParCSRFlexGMRES": "fgmres", "ParCSRLGMRES": "lgmres", "ParCSRCOGMRES": "cogmres",
           "ParCSRBiCGSTAB": "bicgstab",
           # generic Krylov interface (HYPRE_krylov.h): applies to the ParCSR solvers, holds the docs
           "PCG": "pcg", "GMRES": "gmres", "FlexGMRES": "fgmres", "LGMRES": "lgmres", "COGMRES": "cogmres",
           "BiCGSTAB": "bicgstab"}
KIND = {("HYPRE_Int",): "int", ("HYPRE_Real",): "real",
        ("HYPRE_Int", "HYPRE_Int"): "int_k", ("HYPRE_Real", "HYPRE_Int"): "real_k",
        ("HYPRE_Int *",): "int_array", ("HYPRE_Real *",): "real_array",
        ("HYPRE_Int", "HYPRE_Real *"): "count_real_array", (): "flag"}
PER_ROW = {"DofFunc": "one entry per matrix row", "CPoints": "list of rows", "CpointsToKeep": "list of rows",
           "FPoints": "list of rows", "IsolatedFPoints": "list of rows", "Coordinates": "per-row coordinates",
           "InterpVectors": "vectors of matrix size", "GridRelaxPoints": "per-row point ordering"}

# Global settings (HYPRE_utilities.h, deck backend_params).
# A request that sets any of them gets a fresh driver afterwards (they cannot be read back).
GLOBAL_OK = {"SpTransUseVendor", "SpMVUseVendor", "SpMVAlgorithm", "SpGemmUseVendor", "UseGpuRand",
             "LogLevel", "NumThreads"}
GLOBAL_WHY = {"MemoryLocation": "fixed: device memory (GPU only)", "ExecutionPolicy": "fixed: device (GPU only)",
              "GpuAwareMPI": "1 rank: no MPI communication", "PrintErrorMode": "error printing only",
              "PrintErrorVerbosity": "error printing only"}
GDECL = re.compile(r"HYPRE_Int\s+HYPRE_Set(\w+)\s*\(([^)]*)\)\s*;")
DECL = re.compile(r"HYPRE_Int\s+HYPRE_(" + "|".join(SOLVERS) + r")Set(\w+)\s*\(([^)]*)\)\s*;")



def conditions(text: str) -> list[tuple[int, list[str]]]:
    """For each line start offset, the stack of preprocessor conditions active there, as C
    expressions (include guards dropped). Used to guard generated entries the same way."""
    out, stack, pos = [], [], 0
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        out.append((pos, [c for c in stack if c is not None]))
        t = line.strip()
        m = re.match(r"#\s*(ifdef|ifndef|if|elif|else|endif)\b\s*(.*)", t)
        if m:
            kw, arg = m.group(1), re.sub(r"/\*.*?\*/|//.*", "", m.group(2)).strip()
            if kw == "ifndef" and i + 1 < len(lines) and re.match(r"#\s*define\s+" + re.escape(arg) + r"\b",
                                                                  lines[i + 1].strip()):
                stack.append(None)  # include guard
            elif kw == "ifdef":
                stack.append(f"defined({arg})")
            elif kw == "ifndef":
                stack.append(f"!defined({arg})")
            elif kw == "if":
                stack.append(f"({arg})")
            elif kw in ("else", "elif") and stack:
                prev = stack.pop()
                stack.append(None if prev is None else f"!{prev}" if kw == "else" else f"!{prev} && ({arg})")
            elif kw == "endif" and stack:
                stack.pop()
        pos += len(line)
    return out


def guard_at(conds: list, offset: int) -> list[str]:
    g = []
    for p, c in conds:
        if p > offset:
            break
        g = c
    return [c for c in g if c not in ("defined(__cplusplus)",)]


def doc_before(text: str, pos: int) -> str:
    """The /** ... **/ comment that ends right before the declaration at pos (or '')."""
    end = text.rfind("*/", 0, pos)
    if end < 0 or text[end + 2:pos].strip():
        return ""
    start = text.rfind("/*", 0, end)
    body = text[start + 2:end]
    lines = [re.sub(r"^\s*\*+ ?", "", l).rstrip() for l in body.strip("*").splitlines()]
    return "\n".join(lines).strip()


def decls(headers: list[str]):
    """(text, match, guard) for every setter; for Krylov, the generic HYPRE_krylov.h entry comes
    first (documented), the HYPRE_ParCSR* duplicate of the same name is then skipped."""
    for h in headers:
        text = Path(h).read_text()
        conds = conditions(text)
        for m in DECL.finditer(text):
            yield text, m, guard_at(conds, m.start())


def global_setters(header: str, params: list, excluded: list, rows: list) -> None:
    text = Path(header).read_text()
    for m in GDECL.finditer(text):
        name, args = m.group(1), re.sub(r"\s+", " ", m.group(2).strip())
        sig = f"HYPRE_Set{name}({args})"
        if name not in GLOBAL_OK:
            why = GLOBAL_WHY.get(name, "must be set before the first allocation (fixed at driver start)"
                                 if name.startswith("Umpire") else "not a tuning setting")
            excluded.append({"solver": "backend", "name": name, "signature": sig, "reason": why})
            continue
        params.append({"solver": "backend", "name": name, "kind": "int", "signature": sig, "requires": "",
                       "doc": doc_before(text, m.start())})
        rows.append(f'  {{"{name}", {{false, [](HYPRE_Solver, const std::string& v, HYPRE_Int) '
                    f'{{ return (HYPRE_Int)HYPRE_Set{name}(to_int(v)); }}}}}},')


def main(headers: list[str], out_dir: str) -> None:
    params, excluded, inc = [], [], {k: [] for k in dict.fromkeys(SOLVERS.values())}
    util = [h for h in headers if h.endswith("HYPRE_utilities.h")]
    headers = [h for h in headers if not h.endswith("HYPRE_utilities.h")]
    inc["backend"] = []
    for h in util:
        global_setters(h, params, excluded, inc["backend"])
    seen = set()
    for text, m, guard in decls(headers):
        if (SOLVERS[m.group(1)], m.group(2)) in seen:
            continue
        seen.add((SOLVERS[m.group(1)], m.group(2)))
        solver, name, args = SOLVERS[m.group(1)], m.group(2), m.group(3)
        argv = [re.sub(r"\s+", " ", a.strip()) for a in args.split(",")]
        types = tuple(re.sub(r"\s*\b\w+$", "", a).replace(" *", " *").strip() for a in argv[1:])
        types = tuple(t if not t.endswith("*") else t[:-1].strip() + " *" for t in types)
        fn = f"HYPRE_{m.group(1)}Set{name}"
        sig = f"{fn}({', '.join(argv)})"
        kind = KIND.get(types)
        reason = None
        if name == "up":
            reason = "setup routine"
        elif name in PER_ROW:
            reason = PER_ROW[name]
        elif kind is None:
            reason = "argument types " + ", ".join(types) + " (object, file or callback)"
        elif solver != "amg" and kind not in ("int", "real"):
            reason = "not a plain value"
        if reason:
            excluded.append({"solver": solver, "name": name, "signature": sig, "reason": reason})
            continue
        params.append({"solver": solver, "name": name, "kind": kind, "signature": sig,
                       "doc": doc_before(text, m.start()), "requires": " && ".join(guard)})
        call = {"int": f"{fn}(s, to_int(v))", "real": f"{fn}(s, to_real(v))",
                "int_k": f"{fn}(s, to_int(v), k)", "real_k": f"{fn}(s, to_real(v), k)",
                "int_array": f"{fn}(s, int_array(v))", "real_array": f"{fn}(s, real_array(v))",
                "count_real_array": f"{fn}(s, (HYPRE_Int)list_len(v), real_array(v))",
                "flag": f"(to_int(v) ? {fn}(s) : 0)"}[kind]
        indexed = "true" if kind.endswith("_k") else "false"
        row = (f'  {{"{name}", {{{indexed}, [](HYPRE_Solver s, const std::string& v, HYPRE_Int k) '
               f'{{ (void)k; return (HYPRE_Int){call}; }}}}}},')
        if guard:
            row = f"#if {' && '.join(guard)}\n{row}\n#endif"
        inc[solver].append(row)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    body = ["// Generated by gen_params.py from " + " ".join(headers) + ". Do not edit."]
    for solver, rows in inc.items():
        body.append(f"static const ParamTable& {solver}_table() {{\n  static const ParamTable t = {{")
        body += rows
        body.append("  };\n  return t;\n}")
    (out / "hypre_params.inc").write_text("\n".join(body) + "\n")
    (out / "hypre_params.json").write_text(json.dumps(
        {"library": "hypre", "source": headers, "params": params, "excluded": excluded}, indent=1) + "\n")
    n = {s: len(r) for s, r in inc.items()}
    print(f"hypre parameters exposed: {n}; excluded: {len(excluded)}")
    for e in excluded:
        print(f"  excluded {e['solver']}:{e['name']}  ({e['reason']})")


if __name__ == "__main__":
    main(sys.argv[1:-1], sys.argv[-1])
