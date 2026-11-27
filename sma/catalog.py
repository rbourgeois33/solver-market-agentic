"""Parameter catalogues: every parameter each installed library accepts, from the library itself.

Built once per library build (`sma-tools catalog`, on a GPU node: the drivers initialise CUDA):
  hypre  : hypre_params.json, generated from the installed headers at driver build time
           (drivers/hypre/gen_params.py): every setter, its C signature and the header's doc comment.
  muelu  : `sma_muelu --catalog`: MueLu::MasterList (all MueLu parameters + defaults),
           Ifpack2::getValidParameters (smoothers), getValidParameters() of every Belos solver,
           Amesos2 coarse solvers available in this build.
  petsc  : `sma_petsc --catalog`: PETSc's own -help text for every registered KSP type, every PC
           type, and GAMG with level smoothers, on a small matrix; parsed here.
Trilinos parts are then enriched from the sources of the built version (sma/srcscan.py): allowed values
of string parameters (user guide tables + values the code compares them to), keys the code reads but
does not declare ("source-only"), and declared MueLu keys that no code reads (no effect).
Stored in $SMA_WORK/share/sma/params/<lib>.json; read by `sma-tools params`.
The sources of the built versions (user guides included) are extracted from the spack source cache
into $SMA_WORK/share/sma/src/<lib>-<version>/ for the agent to read offline.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from . import sma_work


def params_dir() -> Path:
    return sma_work() / "share" / "sma" / "params"


def src_dir() -> Path:
    return sma_work() / "share" / "sma" / "src"


# ------------------------------------------------------------------ library sources (offline docs)
# Extracted from the spack source cache (the exact versions built), for the agent to read:
# user guides, parameter lists and implementation. Trilinos: only the packages used here.
SRC_MEMBERS = {"trilinos": ["packages/muelu/", "packages/ifpack2/", "packages/belos/", "packages/amesos2/"]}


def _pinned() -> dict[str, tuple[str, str]]:
    """lib -> (version, sha256) from spack/spack.yaml.in + the overlay package recipes."""
    from . import REPO_DIR
    spec = (REPO_DIR / "spack" / "spack.yaml.in").read_text()
    out = {}
    for lib in ("hypre", "petsc", "trilinos"):
        ver = re.search(rf"-\s*{lib}@([\w.]+)", spec).group(1)
        pkg = (REPO_DIR / "spack" / "repo" / "spack_repo" / "sma" / "packages" / lib / "package.py").read_text()
        sha = re.search(rf'version\("{re.escape(ver)}",\s*sha256="([0-9a-f]{{64}})"', pkg).group(1)
        out[lib] = (ver, sha)
    return out


def extract_sources() -> None:
    import tarfile
    cache = Path(os.environ.get("SPACK_USER_CONFIG_PATH", sma_work() / "spack-user")) / "source-cache" / "_source-cache" / "archive"
    for lib, (ver, sha) in _pinned().items():
        dest = src_dir() / f"{lib}-{ver}"
        if dest.exists():
            continue
        tgz = cache / sha[:2] / f"{sha}.tar.gz"
        if not tgz.exists():
            print(f"{lib}: source archive not in the spack cache ({tgz}): skipped")
            continue
        keep = SRC_MEMBERS.get(lib)
        tmp = dest.with_name(dest.name + ".tmp")
        with tarfile.open(tgz) as t:
            members = []
            for m in t:
                rel = m.name.split("/", 1)[1] if "/" in m.name else ""
                if not (m.isfile() or m.isdir()) or (keep and not any(rel.startswith(k) for k in keep)):
                    continue
                m.name = rel or "."
                members.append(m)
            t.extractall(tmp, members=members, filter="data")
        tmp.rename(dest)
        print(f"{lib}: sources {dest}")


def _muelu_descriptions() -> dict[str, str]:
    ver = _pinned()["trilinos"][0]
    xml = src_dir() / f"trilinos-{ver}" / "packages" / "muelu" / "doc" / "UsersGuide" / "masterList.xml"
    if not xml.exists():
        return {}
    out = {}
    for p in ET.parse(xml).getroot().iter("parameter"):
        name, desc = p.findtext("name"), p.findtext("description")
        if name and desc:
            out[name.strip()] = " ".join(desc.split())
    return out


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
    master = _xml_list(ET.parse(raw / "muelu_master.xml").getroot())
    desc = _muelu_descriptions()  # MueLu user guide (masterList.xml) of the installed version
    for e in master:
        e["doc"] = e["doc"] or desc.get(e["name"], "")
    d = {"library": "muelu",
         "muelu": master,
         "ifpack2": _xml_list(ET.parse(raw / "ifpack2.xml").getroot()),
         "belos": belos, "amesos2_available": amesos2}
    _enrich_trilinos(d)
    return d


# Forwarding statements (the interpreter copies the key into a factory list, or a factory declares it
# valid): they do not count as a reader when looking for keys with no effect.
_FORWARD = re.compile(r"(test_and_set_param_2list<[^>]*>|SET_VALID_ENTRY|MUELU_TEST_AND_SET_PARAM_2LIST|"
                      r"test_and_set_var_from_masterlist)\s*\(")


def _enrich_trilinos(d: dict) -> None:
    from . import srcscan
    ver = _pinned()["trilinos"][0]
    src = src_dir() / f"trilinos-{ver}" / "packages"
    if not src.exists():
        d["enriched"] = False
        return
    tables = srcscan.tex_tables(src / "muelu" / "doc" / "UsersGuide")
    belos_entries = [e for v in d["belos"].values() if isinstance(v, list) for e in v]
    parts = {"muelu": d["muelu"], "ifpack2": d["ifpack2"], "belos": belos_entries}
    for part, entries in parts.items():
        skeys = sorted({e["name"].split(" / ")[-1] for e in entries if e["type"] == "string"})
        found = srcscan.scan_values(src, part, skeys)
        for e in entries:
            if e["type"] != "string":
                continue
            key = e["name"].split(" / ")[-1]
            vals, origin, partial = {}, [], True
            for lab in re.findall(r"\\ref\{([^}]+)\}", e.get("doc") or ""):
                t = tables.get(lab)
                if t:
                    vals.update({v: desc for v, desc in t["values"]})
                    origin.append(f"MueLu user guide table {lab} ({t['file']})")
                    partial = partial and not t["exhaustive"]
            if key in found:
                for v in found[key]:
                    vals.setdefault(v, "")
                origin.append("values compared to in the sources (" + ", ".join(sorted(set(found[key].values()))[:4])
                              + (", ..." if len(set(found[key].values())) > 4 else "") + ")")
            if part == "muelu" and key == "coarse: type":
                avail = [k for k, ok in d["amesos2_available"].items() if ok]
                vals.update({k: "Amesos2 direct solver available in this build" for k in avail})
                origin.append("Amesos2 solvers of this build; any smoother type is also accepted")
            if part == "muelu" and key == "coarse: type":  # direct solvers first
                vals = {**{k: v for k, v in vals.items() if v.startswith("Amesos2")}, **vals}
            # one spelling per value: the code often accepts case / space / underscore variants
            seen, uniq = set(), {}
            for v, desc in vals.items():
                n = re.sub(r"[\s_]", "", v.upper())
                if n not in seen:
                    seen.add(n)
                    uniq[v] = desc
            vals = uniq
            if vals:
                dflt = (e.get("default") or "").strip('"')
                norm = lambda v: re.sub(r"[\s_]", "", v.upper())
                if dflt and norm(dflt) not in {norm(v) for v in vals}:
                    vals = {dflt: "default", **vals}
                e["values"] = vals
                e["values_from"] = "; ".join(origin)
                if e.get("doc"):
                    e["doc"] = re.sub(r"(see\s+)?Table\s*~?\s*\\ref\{[^}]+\}", "see values", e["doc"], flags=re.I)
    # declared MueLu keys that nothing reads (accepted by MueLu, no effect)
    files = [(p, p.read_text(errors="replace")) for p in srcscan._files(src, ["muelu/src", "muelu/adapters"])
             if "MasterList" not in p.name]
    for e in d["muelu"]:
        if e["type"] == "sublist":
            continue
        q = f'"{e["name"]}"'
        live = any(not _FORWARD.search(srcscan._stmt(t, m.start()))
                   for _, t in files if q in t for m in re.finditer(re.escape(q), t))
        if not live:
            e["no_effect"] = (f"no code of Trilinos {ver} reads this key (it is only declared/forwarded): "
                              "accepted, but has no effect")
    d["source_only"] = {part: srcscan.scan_hidden(src, part, {e["name"].split(" / ")[-1] for e in entries})
                        for part, entries in parts.items()}
    d["enriched"] = True


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


def build(launcher: str = "", libs=("hypre", "muelu", "petsc"), parse_only: bool = False) -> None:
    """parse_only: rebuild <lib>.json from the raw dumps already in params/raw (no driver run, no GPU)."""
    pd = params_dir()
    pd.mkdir(parents=True, exist_ok=True)
    if not parse_only:
        extract_sources()
    bindir = sma_work() / "bin"
    for lib in libs:
        if lib == "hypre":
            if not (pd / "hypre_params.json").exists():
                raise SystemExit("hypre_params.json missing: rebuild the hypre driver (drivers/build.sh hypre)")
            d = json.loads((pd / "hypre_params.json").read_text())
        else:
            raw = pd / "raw" / lib
            if not parse_only:
                _run(launcher, bindir / f"sma_{lib}", raw)
            d = _muelu(raw) if lib == "muelu" else _petsc(raw)
        (pd / f"{lib}.json").write_text(json.dumps(d, indent=1) + "\n")
        print(f"{lib}: {pd / (lib + '.json')}")


# ------------------------------------------------------------------ query
def _short(doc: str, full: bool) -> str:
    doc = doc or ""
    return doc if full else " ".join(doc.split())[:160]


_TAGS = {"gpu": "gpu", "depends": "gpu depends on its sub-type", "exempt": "coarsest level: host or GPU",
         "host_setup": "host setup: allowed only when k = 0", "unknown": "no GPU rule",
         "host_only": "HOST ONLY: refused", "do_not_try": "DO NOT TRY: refused"}


def _values(e: dict, full: bool, lib: str, all_: bool) -> str:
    """Allowed values of a string parameter, tagged with their GPU status (sma/gpu_support.yaml). Host-only
    and do-not-try values are hidden unless --all (only GPU options are explored)."""
    from .gpu import value_status
    vals = e.get("values")
    if not vals:
        return ""
    key = e["name"].split(" / ")[-1]
    shown, hidden = [], 0
    for v, desc in vals.items():
        st = value_status(lib, key, v)
        if st in ("host_only", "do_not_try") and not all_:
            hidden += 1
            continue
        shown.append((v, desc, st))
    tag = lambda st: f" [{_TAGS[st]}]" if st else ""
    more = f"  ({hidden} host-only / do-not-try value(s) hidden; --all shows them)" if hidden else ""
    if full:
        lines = "".join(f"\n      {v}{tag(st)}" + (f": {_short(desc, False)}" if desc else "") for v, desc, st in shown)
        return f"\n    values ({e.get('values_from', '')}):{lines}" + (f"\n    {more.strip()}" if more else "")
    names = " | ".join(f"{v}{tag(st)}" for v, _, st in shown)
    return f"\n    values: {names if len(names) < 500 else names[:500] + ' ...'}{more}"


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
    keep = lambda e: rx is None or rx.search(e["name"]) or rx.search(e.get("doc") or "") or \
        any(rx.search(v) for v in (e.get("values") or {}))
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
                               + (f"\n    {_short(e['doc'], full)}" if e["doc"] else "")
                               + (f"\n    NO EFFECT: {e['no_effect']}" if e.get("no_effect") else "")
                               + _values(e, full, "muelu", all_))
        for part, keys in (d.get("source_only") or {}).items():
            if section and not part.startswith(section.split(":")[0]):
                continue
            for k, e in sorted(keys.items()):
                if rx is None or rx.search(k):
                    out.append(f"[{part}:source-only] \"{k}\"  {e['type']}"
                               + (f"  default={e['default']}" if e["default"] else "")
                               + f"\n    read by the code but not in the library's list of valid parameters: "
                               + ", ".join(e["where"]))
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
