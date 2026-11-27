"""Mine the sources of the built Trilinos version ($SMA_WORK/share/sma/src) for what the libraries'
own parameter lists do not say:

- allowed values of STRING parameters: the MueLu user guide tables that a description points to
  ("Possible values: see Table~\\ref{tab:coarse_solvers}"), and the string literals the sources
  compare the parameter to (test_param_2list(..., "key", "value"), `var == "value"` where var was read
  from "key", Teuchos::tuple<std::string>(...) validators next to "key");
- keys the sources READ but do not declare in their valid-parameter lists (e.g. Ifpack2's
  "chebyshev: algorithm", read with isParameter/get and absent from Ifpack2::getValidParameters).

Heuristic (regular expressions over the code), so every value and key carries the file:line it came
from. Used by sma/catalog.py when the MueLu catalogue is built.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

CODE_EXT = {".hpp", ".cpp", ".h", ".cc"}
SKIP_DIR = re.compile(r"/(test|tests|example|examples|doc|research|deprecated)/", re.I)

# Where each catalogue part reads its parameters (relative to packages/).
SCOPES = {
    "muelu": ["muelu/src", "muelu/adapters/tpetra", "muelu/adapters/xpetra"],
    "ifpack2": ["ifpack2/src"],
    "belos": ["belos/src", "belos/tpetra/src"],
}
# Not user options: ML-syntax translator, other entry points, test drivers, hypre pass-through args.
HIDDEN_SKIP_FILES = re.compile(r"ML2MueLu|MLParameterList|ShiftedLaplacian|Facade|AdaptiveSaML|TestFramework|"
                               r"Ifpack2_Hypre|Hiptmair")
# Hidden keys are only searched where the user's list is read: MueLu's interpreter (the factories'
# own lists use MasterList names), all of Ifpack2 and Belos.
HIDDEN_SCOPES = {
    "muelu": ["muelu/src/Interface", "muelu/adapters/tpetra"],
    "ifpack2": ["ifpack2/src"],
    "belos": ["belos/src", "belos/tpetra/src"],
}


def _files(root: Path, rels: list[str]):
    for rel in rels:
        base = root / rel
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix in CODE_EXT and p.is_file() and not SKIP_DIR.search("/" + str(p.relative_to(root))):
                yield p


def _text(p: Path) -> str:
    return p.read_text(errors="replace")


def _line(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


# ------------------------------------------------------------------ MueLu user guide tables
_TEX_CMD = [(re.compile(r"\\verb(.)(.+?)\1"), r"\2"), (re.compile(r"\{\\tt\s+([^}]*)\}"), r"\1"),
            (re.compile(r"\\texttt\{([^}]*)\}"), r"\1"), (re.compile(r"~?\\cite\{[^}]*\}"), ""),
            (re.compile(r"\\(\w+)\{\}"), r"\1"), (re.compile(r"\$([^$]*)\$"), r"\1")]
_FIRST_VALUES = re.compile(r"\\verb(.)(.+?)\1|\{\\tt\s+([^}]+)\}|\\texttt\{([^}]+)\}")


def _detex(s: str) -> str:
    for rx, rep in _TEX_CMD:
        s = rx.sub(rep, s)
    return " ".join(s.replace("\\\\", " ").split())


def tex_tables(guide_dir: Path) -> dict[str, dict]:
    """label -> {"values": [(value, description)], "file": "options.tex:LINE", "exhaustive": bool}."""
    out = {}
    for tex in sorted(guide_dir.glob("*.tex")):
        t = _text(tex)
        for m in re.finditer(r"\\begin\{table\}(.*?)\\end\{table\}", t, re.S):
            body = m.group(1)
            lab = re.search(r"\\label\{([^}]+)\}", body)
            tab = re.search(r"\\begin\{tabular\}\{.*?\}\n(.*?)\\end\{tabular\}", body, re.S)
            if not lab or not tab:
                continue
            caption = re.search(r"\\caption\{(.*?)\}\s*\n", body, re.S)
            rows, vals = re.split(r"\\\\", tab.group(1)), []
            for row in rows:
                row = re.sub(r"\\(toprule|midrule|bottomrule|hline)", "", row).strip()
                if "&" not in row:
                    continue
                first, _, rest = row.partition("&")
                names = [next(g for g in mm.groups()[1:] if g) for mm in _FIRST_VALUES.finditer(first)]
                desc = _detex(rest.split("&")[-1])
                for n in names:
                    n = n.strip().strip('"')
                    if n and ":" not in n:  # a header cell names the parameter ("coarse: type")
                        vals.append((n, desc))
            if vals:
                cap = _detex(caption.group(1)) if caption else ""
                out[lab.group(1)] = {"values": vals, "file": f"{tex.name}:{_line(t, m.start())}",
                                     "exhaustive": not re.search(r"commonly used", cap, re.I)}
    return out


# ------------------------------------------------------------------ values compared in the code
# Values decided elsewhere: (file, regex whose group 1 is a value). The smoother names MueLu passes on
# to Ifpack2's factory; Belos's string -> enum helpers and its orthogonalization names.
_IFP2_TYPES = [("ifpack2/src/Ifpack2_Details_OneLevelFactory_def.hpp", r'precTypeUpper\s*==\s*"([^"]+)"'),
               ("ifpack2/src/Ifpack2_Factory_def.hpp", r'precTypeUpper\s*==\s*"([^"]+)"')]
_SCALE = [("belos/src/BelosTypes.cpp", r'scaleType\s*==\s*"([^"]+)"')]
EXTRA_VALUES = {
    **{("muelu", k): _IFP2_TYPES for k in ("smoother: type", "smoother: pre type", "smoother: post type",
                                           "coarse: type")},
    **{("belos", k): _SCALE for k in ("Implicit Residual Scaling", "Explicit Residual Scaling", "Residual Scaling")},
    ("belos", "Residual Norm"): [("belos/src/BelosTypes.cpp", r'normType\s*==\s*"([^"]+)"')],
    ("belos", "Orthogonalization"): [("belos/src/BelosOrthoManagerFactory.hpp", r'theList_\[[^\]]*\]\s*=\s*"([^"]+)"')],
}
_LIT = r'"([^"\n]*)"'


def _stmt(t: str, pos: int) -> str:
    start = max(t.rfind(";", 0, pos), t.rfind("{", 0, pos), t.rfind("}", 0, pos)) + 1
    end = t.find(";", pos)
    return t[start:end if end > 0 else len(t)]


def _compared(t: str, var: str) -> list[tuple[str, int]]:
    v = re.escape(var)
    rx = re.compile(r"\b" + v + r"\s*[!=]=\s*" + _LIT + r"|" + _LIT + r"\s*[!=]=\s*" + v + r"\b")
    return [(m.group(1) if m.group(1) is not None else m.group(2), m.start()) for m in rx.finditer(t)]


def scan_values(src: Path, part: str, keys: list[str]) -> dict[str, dict[str, str]]:
    """key -> {value: "file:line"} for string keys, from the code of one catalogue part. Patterns, in the
    statement that names the key: test_param_2list(..., "key", "value"); get<std::string>("key") ==
    "value"; tuple<std::string>("a", "b") validators; setStringToIntegralParameter("key", ..., names())
    with names[i] = "value"; and `var == "value"` anywhere in the file when var was read from the key
    (one level of propagation into a member: member_ = var)."""
    found: dict[str, dict[str, str]] = defaultdict(dict)
    files = [(p, _text(p)) for p in _files(src, SCOPES[part])]
    for p, t in files:
        rel = str(p.relative_to(src))
        add = lambda key, v, pos: found[key].setdefault(v, f"{rel}:{_line(t, pos)}")
        occs = [(key, m) for key in keys if f'"{key}"' in t for m in re.finditer(re.escape(f'"{key}"'), t)]
        for key, m in occs:
            st = _stmt(t, m.start())
            k = re.escape(key)
            for mm in re.finditer(r'(?:test_param_2list<std::string>|MUELU_TEST_PARAM_2LIST)\s*\(\s*\w+\s*,\s*\w+\s*,\s*"'
                                  + k + r'"\s*,\s*(?:std::string\s*,\s*)?' + _LIT, st):
                add(key, mm.group(1), m.start())
            for mm in re.finditer(r'get<std::string>\(\s*"' + k + r'"\s*\)\s*[!=]=\s*' + _LIT, st):
                add(key, mm.group(1), m.start())
            for mm in re.finditer(r"tuple<std::string>\s*\(([^)]*)\)", st):
                for v in re.findall(_LIT, mm.group(1)):
                    add(key, v, m.start())
            mm = re.search(r'setStringToIntegralParameter<[^>]*>\s*\(\s*"' + k + r'"[^;]*?,\s*(\w+)\s*\(\s*\)', st)
            if mm:
                for a in re.finditer(re.escape(mm.group(1)) + r"\s*\[\s*\d+\s*\]\s*=\s*" + _LIT, t):
                    add(key, a.group(1), a.start())
            names = set()
            if re.search(r'get<std::string>\(\s*"' + k + r'"|set_var_2list<std::string>\(|get\(\s*"' + k
                         + r'"\s*,\s*(?:std::string\()?"', st):
                pre = re.sub(r"//[^\n]*", "", st[:st.find(f'"{key}"')])
                a = re.findall(r"\b(\w+)\s*=(?!=)", pre)
                if a:
                    names.add(a[-1])
            a = re.search(r'MUELU_SET_VAR_2LIST\([^;]*"' + k + r'"\s*,\s*std::string\s*,\s*(\w+)\)', st)
            if a:
                names.add(a.group(1))
            names -= {"if", "return", "const", "auto", "else"}
            for n in list(names):
                names |= set(re.findall(r"\b(\w+_)\s*=\s*" + re.escape(n) + r"\s*;", t))
            for n in names:
                for v, pos in _compared(t, n):
                    add(key, v, pos)
                # std::map<std::string, ...> M; M["v"] = ...; used as M.count(n) / M[n] / M.find(n)
                for mp in set(re.findall(r"\b(\w+)\s*(?:\.count|\.find|\.at)?\s*[(\[]\s*" + re.escape(n) + r"\s*[)\]]", t)):
                    for a in re.finditer(r"\b" + re.escape(mp) + r'\s*\[\s*"([^"]+)"\s*\]\s*=', t):
                        add(key, a.group(1), a.start())
    for (pt, key), where in EXTRA_VALUES.items():
        if pt != part or key not in keys:
            continue
        for rel, rx in where:
            f = src / rel
            if f.exists():
                t = _text(f)
                for m in re.finditer(rx, t):
                    found[key].setdefault(m.group(1), f"{rel}:{_line(t, m.start())}")
    return {k: {v: w for v, w in d.items() if v.strip()} for k, d in found.items()
            if any(v.strip() for v in d)}


# ------------------------------------------------------------------ keys read but not declared
_READ = re.compile(r'\b(?:get|isParameter|isType|getEntryPtr|getPtr|set_var_2list|test_param_2list|'
                   r'isSublist)\s*(?:<\s*([\w:<>, ]+?)\s*>)?\s*\(\s*(?:\w+\s*,\s*\w+\s*,\s*)?"([^"\n]+)"\s*([,)])'
                   r'\s*([^;\n]{0,80})')


def _type_of(tmpl: str | None, default: str) -> str:
    if tmpl:
        t = tmpl.replace("std::", "").replace(" ", "")
        return {"string": "string", "bool": "bool", "int": "int", "double": "double", "float": "double",
                "LO": "int", "GO": "int", "magnitude_type": "double", "MT": "double", "SC": "double",
                "Scalar": "double", "size_t": "int"}.get(t, t)
    d = default.strip()
    if d.startswith('"') or d.startswith("std::string("):
        return "string"
    if d in ("true", "false"):
        return "bool"
    if re.fullmatch(r"-?\d+", d):
        return "int"
    if re.fullmatch(r"-?[\d.]+(e-?\d+)?f?", d):
        return "double"
    return "?"


def scan_hidden(src: Path, part: str, declared: set[str]) -> dict[str, dict]:
    """key -> {"type", "default", "where"}: keys read in the code of a part, absent from its lists."""
    out: dict[str, dict] = {}
    for p in _files(src, HIDDEN_SCOPES[part]):
        if HIDDEN_SKIP_FILES.search(p.name):
            continue
        t = _text(p)
        rel = str(p.relative_to(src))
        for m in _READ.finditer(t):
            tmpl, key, sep, after = m.group(1), m.group(2), m.group(3), m.group(4)
            if key in declared or len(key) < 3 or "%" in key or key.startswith((" ", "-")):
                continue
            if re.search(r"(Timer|Time|timer|Monitor)", key) or key in ("Hierarchy", "Factories"):
                continue
            if part == "muelu" and ": " not in key:  # MueLu user keys are "area: name"
                continue
            if tmpl and re.search(r"RCP|map_type|_type$|matrix|Vector|parts", tmpl):  # objects, not options
                continue
            default = ""
            if sep == ",":
                dm = re.match(r'\s*("[^"]*"|[^,)]+)', after)
                default = dm.group(1).strip() if dm else ""
            e = out.setdefault(key, {"type": "?", "default": "", "where": []})
            ty = _type_of(tmpl, default)
            if e["type"] == "?" and ty != "?":
                e["type"] = ty
            literal = re.fullmatch(r'"[^"]*"|true|false|-?[\d.]+(e-?\d+)?f?', default.strip())
            if literal and not e["default"]:
                e["default"] = default.strip().strip('"')
            w = f"{rel}:{_line(t, m.start())}"
            if len(e["where"]) < 3 and w not in e["where"]:
                e["where"].append(w)
    return out
