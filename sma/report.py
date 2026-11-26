"""Campaign report: evolution chart (report.png) and ranked table (report.md).

Chart: one point per benchmarked configuration in submission order, cost on a log axis,
colour = library (fixed per library, never by rank), best-so-far as a step line.
Configurations that did not produce a valid result are drawn as hollow markers in a
separate "not valid" strip above the plot (their cost does not exist).
Infrastructure failures (charged = false) are not plotted; they are counted in the table.
"""
from __future__ import annotations

import math
from pathlib import Path

from .campaign_state import Campaign

# Reference palette, categorical slots 1-3 (validated all-pairs for scatter, light mode).
LIB_COLOR = {"hypre": "#257de0", "muelu": "#f47230", "petsc": "#00b894"}
LIB_LABEL = {"hypre": "hypre BoomerAMG", "muelu": "MueLu", "petsc": "PETSc GAMG"}
TEXT, TEXT2, GRID, SURFACE = "#191919", "#666666", "#e5e5e5", "#ffffff"


def _rows(c: Campaign):
    rows = []
    n = 0
    for e in c.journal():
        if not e.get("charged", True):
            continue
        n += 1
        r = e["result"]
        rows.append({"n": n, "id": e["id"], "lib": e["deck"]["lib"], "solver": e["deck"]["solver"],
                     "valid": e["valid"], "cost": e["cost"] if e["valid"] else None,
                     "its": r.get("iterations"), "setup": r.get("t_setup_median"),
                     "solve": r.get("t_solve_median"), "opcx": r.get("operator_complexity"),
                     "mem": r.get("gpu_mem_peak_bytes"), "relres": r.get("true_relres"),
                     "why": e.get("why", ""), "error": (r.get("error") or "").splitlines()[0][:80]
                     if r.get("error") else ""})
    return rows


def chart(c: Campaign, rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    k = c.params["solves_per_setup"]
    ylabel = "time per solve (ms, log)" if k == 0 else f"(setup + {k} solves) / {k}  (ms, log)"
    valid = [r for r in rows if r["valid"]]
    invalid = [r for r in rows if not r["valid"]]

    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT2,
                         "xtick.color": TEXT2, "ytick.color": TEXT2})
    fig, (ax_bad, ax) = plt.subplots(2, 1, sharex=True, figsize=(10, 5.6),
                                     gridspec_kw={"height_ratios": [1, 9], "hspace": 0.06})
    for a in (ax, ax_bad):
        a.set_facecolor(SURFACE)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
    fig.patch.set_facecolor(SURFACE)

    # best-so-far step line (drawn first, under the points)
    xs, ys, best = [], [], math.inf
    for r in rows:
        if r["valid"] and r["cost"] < best:
            best = r["cost"]
        if best < math.inf:
            xs.append(r["n"]); ys.append(best * 1e3)
    if xs:
        ax.step(xs, ys, where="post", color=TEXT2, linewidth=2, zorder=1)

    for lib, col in LIB_COLOR.items():
        pts = [r for r in valid if r["lib"] == lib]
        if pts:
            ax.scatter([r["n"] for r in pts], [r["cost"] * 1e3 for r in pts], s=64, color=col,
                       edgecolors=SURFACE, linewidths=1.5, zorder=3)
        bad = [r for r in invalid if r["lib"] == lib]
        if bad:
            ax_bad.scatter([r["n"] for r in bad], [0.5] * len(bad), s=64, facecolors="none",
                           edgecolors=col, linewidths=2, zorder=3)

    ax.set_yscale("log")
    ax.grid(True, which="major", axis="y", color=GRID, linewidth=1)
    ax.grid(True, which="minor", axis="y", color=GRID, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("configuration (submission order)")
    ax_bad.set_ylim(0, 1)
    ax_bad.set_yticks([])
    ax_bad.set_ylabel("not\nvalid", rotation=0, ha="right", va="center", color=TEXT2)
    ax_bad.tick_params(axis="x", length=0)
    ax_bad.spines["left"].set_visible(False)

    # direct label on the final best only
    if valid:
        b = min(valid, key=lambda r: r["cost"])
        ax.annotate(f"best {b['cost'] * 1e3:.1f} ms  ({LIB_LABEL[b['lib']]}, #{b['n']})",
                    xy=(b["n"], b["cost"] * 1e3), xytext=(12, 34), textcoords="offset points",
                    ha="left", va="bottom", color=TEXT, fontsize=10,
                    arrowprops={"arrowstyle": "-", "color": TEXT2, "linewidth": 1, "shrinkB": 5})

    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, color=col, label=LIB_LABEL[lib])
               for lib, col in LIB_COLOR.items() if any(r["lib"] == lib for r in rows)]
    handles.append(Line2D([], [], color=TEXT2, linewidth=2, label="best so far"))
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor="none",
                          markeredgecolor=TEXT2, markeredgewidth=2, label="not valid (diverged, crashed, rejected)"))
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)

    system = Path(c.params["system"]).name
    fig.suptitle(f"Tuning campaign on {system}: {len(rows)} configurations, tol {c.params['tol']:g} "
                 f"(true relative residual)", x=0.07, ha="left", fontsize=12, color=TEXT)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def table(c: Campaign, rows: list[dict], out: Path, top: int = 15) -> None:
    ms = lambda x: "" if x is None else f"{x * 1e3:.2f}"
    uncharged = sum(1 for e in c.journal() if not e.get("charged", True))
    lines = [f"# Campaign report: {Path(c.params['system']).name}", "",
             f"Budget {c.params['budget']}, used {c.used()}, valid {sum(r['valid'] for r in rows)}, "
             f"infrastructure failures not charged: {uncharged}. Tolerance {c.params['tol']:g} on the true "
             f"relative residual, solves per setup k = {c.params['solves_per_setup']}.", "",
             "![evolution](report.png)", "", "## Best per library", "",
             "| library | best cost (ms) | its | config # | id |", "|---|---|---|---|---|"]
    for lib in LIB_COLOR:
        v = [r for r in rows if r["lib"] == lib and r["valid"]]
        if v:
            b = min(v, key=lambda r: r["cost"])
            lines.append(f"| {LIB_LABEL[lib]} | {ms(b['cost'])} | {b['its']} | {b['n']} | [{b['id']}](runs/{b['id']}/deck.yaml) |")
    lines += ["", f"## Top {top} configurations", "",
              "| rank | # | library | solver | cost (ms) | setup (ms) | solve (ms) | its | op cx | GPU mem (GB) | why |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(sorted([r for r in rows if r["valid"]], key=lambda r: r["cost"])[:top], 1):
        mem = f"{r['mem'] / 2**30:.1f}" if r["mem"] else ""
        opcx = f"{r['opcx']:.2f}" if isinstance(r["opcx"], (int, float)) else ""
        lines.append(f"| {i} | {r['n']} | {r['lib']} | {r['solver']} | {ms(r['cost'])} | {ms(r['setup'])} | "
                     f"{ms(r['solve'])} | {r['its']} | {opcx} | {mem} | {r['why'][:70]} |")
    lines += ["", "## Not valid", "", "| # | library | reason |", "|---|---|---|"]
    for r in rows:
        if not r["valid"]:
            reason = r["error"] or (f"true relres {r['relres']:.1e} > tol" if r["relres"] else "no result")
            lines.append(f"| {r['n']} | {r['lib']} | {reason.replace('|', '/')} |")
    lines += ["", "## All configurations (submission order)", "",
              "| # | library | cost (ms) | its | id |", "|---|---|---|---|---|"]
    link = lambda i: f"[{i}](runs/{i}/deck.yaml) ([result](runs/{i}/result.json))"
    lines += [f"| {r['n']} | {r['lib']} | {ms(r['cost']) or 'not valid'} | {r['its']} | {link(r['id'])} |" for r in rows]
    out.write_text("\n".join(lines) + "\n")


def report(campaign_dir: str) -> tuple[Path, Path]:
    c = Campaign(campaign_dir)
    rows = _rows(c)
    png, md = c.dir / "report.png", c.dir / "report.md"
    chart(c, rows, png)
    table(c, rows, md)
    return png, md
