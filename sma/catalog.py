"""Parameter catalogues: every parameter each installed library accepts, from the library itself.

Built once per library build (`sma-tools catalog`, on a GPU node: the drivers initialise CUDA):
  hypre  : hypre_params.json, generated from the installed headers at driver build time
           (drivers/hypre/gen_params.py): every setter, its C signature and the header's doc comment.
  muelu  : `sma_muelu --catalog`: MueLu::MasterList (all MueLu parameters + defaults),
           Ifpack2::getValidParameters (smoothers), getValidParameters() of every Belos solver,
           Amesos2 coarse solvers available in this build.
  petsc  : `sma_petsc --catalog`: PETSc's own -help text for every registered KSP type, every PC
           type, and GAMG with level smoothers, on a small matrix; parsed here.
Stored in $SMA_WORK/share/sma/params/<lib>.json; read by `sma-tools params`.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from . import sma_work


def params_dir() -> Path:
    return sma_work() / "share" / "sma" / "params"


# ------------------------------------------------------------------ build
def _run(launcher: str, exe: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [*shlex.split(launcher), str(exe), "--catalog", str(out)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _xml_list(node: ET.Element, prefix: str = "") -> list[dict]:
    out = []
    for c in node:
        if c.tag == "Parameter":
            out.append({"name": prefix + c.get("name"), "type": c.get("type"), "default": c.get("value"),
                        "doc": c.get("docString", "")})
        elif c.tag == "ParameterList":
            out.append({"name": prefix + c.get("name"), "type": "sublist", "default": None, "doc": ""})
            out += _xml_list(c, prefix + c.get("name") + " / ")
    return out


def _muelu(raw: Path) -> dict:
    belos = {}
    for line in (raw / "belos.txt").read_text().splitlines():
        if m := re.match(r"^(\d+) (.+)$", line):
            belos[m.group(2)] = _xml_list(ET.parse(raw / f"belos_{m.group(1)}.xml").getroot())
        elif m := re.match(r"^- (.+?) ERROR ", line):
            belos[m.group(1)] = {"error": "not registered in this build"}
    amesos2 = {l.split()[0]: l.split()[1] == "1" for l in (raw / "amesos2.txt").read_text().splitlines()}
    return {"library": "muelu",
            "muelu": _xml_list(ET.parse(raw / "muelu_master.xml").getroot()),
            "ifpack2": _xml_list(ET.parse(raw / "ifpack2.xml").getroot()),
            "belos": belos, "amesos2_available": amesos2}


OPT = re.compile(r"^\s*-(s\d+_)(\S+) <(.*?)>: (.*)$")


def _norm(name: str) -> str:
    return re.sub(r"mg_levels_\d+_", "mg_levels_", name)


def _petsc(raw: Path) -> dict:
    opts: dict[str, dict] = {}
    scenarios, errors, scen = [], {}, None
    section = ""
    for line in (raw / "petsc_help.txt").read_text(errors="replace").splitlines():
        if line.startswith("### SCENARIO "):
            _, _, pre, rest = line.split(" ", 3)
            scen = rest
            scenarios.append(scen)
            continue
        if line.startswith("### ERROR "):
            errors[scen] = line[len("### ERROR "):]
            continue
        m = OPT.match(line)
        if not m:
            t = line.strip()
            if t.endswith("-------------") or t.endswith("options:") or t.endswith("Options"):
                section = re.sub(r"\s*-+$", "", t).replace(m.group(1) if m else "", "")
                section = re.sub(r"\(s\d+_\S*\)", "", section).strip()
            continue
        name = _norm(m.group(2).rstrip(":"))
        val, desc = m.group(3), m.group(4).strip()
        default = val.split(" : formerly ")[-1] if " : formerly " in val else val.replace("now ", "", 1)
        man = re.search(r"\((\w+)\)\s*$", desc)
        e = opts.setdefault(name, {"name": name, "default": default, "doc": desc,
                                   "manpage": man.group(1) if man else "", "section": section, "seen_in": []})
        if scen not in e["seen_in"]:
            e["seen_in"].append(scen)
    n = len(scenarios)
    for e in opts.values():
        if len(e["seen_in"]) == n:
            e["seen_in"] = "all"
    return {"library": "petsc", "scenarios": scenarios, "scenario_errors": errors,
            "params": sorted(opts.values(), key=lambda e: e["name"])}


def build(launcher: str = "", libs=("hypre", "muelu", "petsc")) -> None:
    pd = params_dir()
    pd.mkdir(parents=True, exist_ok=True)
    bindir = sma_work() / "bin"
    for lib in libs:
        if lib == "hypre":
            if not (pd / "hypre_params.json").exists():
                raise SystemExit("hypre_params.json missing: rebuild the hypre driver (drivers/build.sh hypre)")
            d = json.loads((pd / "hypre_params.json").read_text())
        else:
            raw = pd / "raw" / lib
            _run(launcher, bindir / f"sma_{lib}", raw)
            d = _muelu(raw) if lib == "muelu" else _petsc(raw)
        (pd / f"{lib}.json").write_text(json.dumps(d, indent=1) + "\n")
        print(f"{lib}: {pd / (lib + '.json')}")


# ------------------------------------------------------------------ query
def _short(doc: str, full: bool) -> str:
    doc = doc or ""
    return doc if full else " ".join(doc.split())[:160]


def _petsc_host_only_scenario(scen: str) -> bool:
    from .gpu import rules
    r = rules()["petsc"]["rules"]["pc_type"]
    m = re.search(r"-pc_type (\S+)", scen)
    return bool(m) and m.group(1) in r.get("host_only", [])


def show(lib: str, grep: str | None = None, section: str | None = None, full: bool = False,
         all_: bool = False) -> str:
    p = params_dir() / f"{lib}.json"
    if not p.exists():
        raise SystemExit(f"no catalogue for {lib}: run `sma-tools catalog` after building (GPU node)")
    d = json.loads(p.read_text())
    rx = re.compile(grep, re.I) if grep else None
    keep = lambda e: rx is None or rx.search(e["name"]) or rx.search(e.get("doc") or "")
    out = []
    if lib in ("hypre", "muelu"):
        out.append(f"GPU rules for values (smoothers, coarsening, ...): sma-tools params {lib} --gpu")
    if lib == "hypre":
        out.append("deck key = <Name> (indexed kinds: \"<Name>:k\"); precond_params -> amg, "
                   "solver_params -> the chosen solver. kinds: int real int_k real_k int_array real_array "
                   "count_real_array flag")
        for e in d["params"]:
            if (section is None or e["solver"] == section) and keep(e):
                out.append(f"[{e['solver']}] {e['name']}  ({e['kind']})  {e['signature']}")
                if e["doc"]:
                    out.append("    " + _short(e["doc"], full).replace("\n", "\n    "))
        out.append(f"excluded: " + ", ".join(f"{e['solver']}:{e['name']} ({e['reason']})" for e in d["excluded"]))
    elif lib == "muelu":
        parts = {"muelu": d["muelu"], "ifpack2": d["ifpack2"]}
        parts.update({f"belos:{k}": v for k, v in d["belos"].items() if isinstance(v, list)})
        for name, entries in parts.items():
            if section and not name.startswith(section):
                continue
            for e in entries:
                if keep(e):
                    out.append(f"[{name}] \"{e['name']}\"  {e['type']}  default={e['default']}"
                               + (f"\n    {_short(e['doc'], full)}" if e["doc"] else ""))
        if section in (None, "amesos2"):
            out.append("amesos2 coarse solvers available: "
                       + ", ".join(k for k, v in d["amesos2_available"].items() if v))
    else:
        out.append("GPU rules for values (pc_type, ksp_type, mat_type ...): sma-tools params petsc --gpu")
        hidden = 0
        for e in d["params"]:
            if not all_ and e["seen_in"] != "all" and all(_petsc_host_only_scenario(x) for x in e["seen_in"]):
                hidden += 1
                continue
            if (section is None or section.lower() in e["section"].lower()) and keep(e):
                ctx = "" if e["seen_in"] == "all" else f"  [with: {'; '.join(e['seen_in'][:3])}" + \
                      (f" +{len(e['seen_in']) - 3}" if len(e["seen_in"]) > 3 else "") + "]"
                out.append(f"-{e['name']}  default={e['default']}  {_short(e['doc'], full)}{ctx}")
        if hidden:
            out.append(f"({hidden} options of host-only PC types hidden; --all to show)")
    return "\n".join(out)
