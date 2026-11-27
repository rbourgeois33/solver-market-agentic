"""sma campaign: run an agent on a tuning campaign until it is done.

    sma campaign DIR --system SYSDIR [campaign-init options] --agent claude --model MODEL
                     [--mode slurm|local] [--no-exclusive] [--max-sessions N]

1. Creates the campaign (or resumes it if DIR/campaign.json exists).
2. Writes the agent's prompt (prompts/agent.md, filled in) and its permissions
   (DIR/.claude/settings.json: sma-tools, read-only commands, Slurm queries (cluster) or nvidia-smi
   (local), reading the campaign / repo docs / installed libraries and sources, writing only in DIR,
   web access to the libraries' documentation sites).
3. Runs agent sessions until: DONE file written, budget exhausted (+ final summary), 2 consecutive
   sessions without a new run, or --max-sessions. Transcripts: DIR/transcripts/session-NN.*
4. GPU worker on demand: started when a request is pending and no worker is alive; it exits after
   --idle-timeout without requests. Always stopped at the end, whatever the reason (also Ctrl-C,
   SIGTERM, errors).
5. Final outputs: report.png / report.md, best.yaml, index.md (links to everything).

Agents: "claude" (Claude Code CLI, `claude -p`), or "cmd:TEMPLATE" for any other CLI agent, with
{prompt_file}, {workdir}, {model} substituted (run through bash, in DIR).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import string
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import REPO_DIR, sma_work
from .campaign_state import Campaign
from .worker import control

DOC_DOMAINS = ["hypre.readthedocs.io", "petsc.org", "trilinos.github.io", "docs.trilinos.org",
               "github.com", "raw.githubusercontent.com", "gitlab.com"]
NO_PROGRESS_SESSIONS = 2
READ_ONLY_CMDS = ["cd", "pwd", "ls", "cat", "head", "tail", "grep", "wc", "echo", "sort", "uniq", "cut", "tr",
                  "diff", "comm", "column", "nl", "basename", "dirname", "realpath", "stat", "du", "date",
                  "true"]


# ------------------------------------------------------------------ permissions
def read_roots(cdir: Path, system: Path) -> list[Path]:
    return [cdir, system, REPO_DIR / "docs", REPO_DIR / "decks", REPO_DIR / "prompts", REPO_DIR / "sma",
            REPO_DIR / "README.md", sma_work() / "share", sma_work() / "install", sma_work() / "view",
            sma_work() / "env" / "build_log.md", sma_work() / "logs"]


def add_dirs(cdir: Path, system: Path) -> list[Path]:
    """Extra working directories given to Claude Code (--add-dir), read-only through the rules: without
    them, `cd <dir>; cat ...` outside the campaign directory is refused as a compound command."""
    return [REPO_DIR] + [r for r in read_roots(cdir, system) if r != cdir and r.is_dir() and REPO_DIR not in r.parents]


def settings(cdir: Path, system: Path, mode: str) -> dict:
    a = lambda p: "/" + str(p)  # Claude Code absolute-path rule prefix: //abs/path
    read_roots_ = read_roots(cdir, system)
    # Plain read-only commands, chainable with ; && | (Claude Code checks each part). Loops (for/while)
    # cannot be allowed by a prefix rule: they go inside `sma-tools shell '...'` (bwrap, writes only in
    # decks/ and work/), as do sed/awk/python/find, which could write anywhere if allowed directly.
    allow = ["Bash(sma-tools:*)"] + [f"Bash({c}:*)" for c in READ_ONLY_CMDS]
    allow += ["Bash(squeue:*)", "Bash(sacct:*)"] if mode == "slurm" else ["Bash(nvidia-smi:*)"]
    for r in read_roots_:
        allow.append(f"Read({a(r)})")  # the path itself (e.g. Grep rooted at a directory)
        if r.suffix == "":
            allow.append(f"Read({a(r)}/**)")  # Read rules cover Read, Glob, Grep
    allow.append(f"Edit({a(cdir)}/**)")  # Edit rules cover Edit and Write
    allow += [f"WebFetch(domain:{d})" for d in DOC_DOMAINS]
    record = ["journal.jsonl", "campaign.json", "runs/**", "queue/**", "transcripts/**", ".claude/**"]
    deny = [f"Edit({a(cdir)}/{r})" for r in record]
    deny += ["Bash(sma-tools worker start:*)", "Bash(sma-tools worker stop:*)", "Bash(sma-tools campaign-init:*)",
             "Bash(sma-tools catalog:*)"]
    return {"permissions": {"allow": allow, "deny": deny, "defaultMode": "dontAsk"}}


# ------------------------------------------------------------------ prompt
def prompt(c: Campaign, mode: str, session: int) -> str:
    p = c.params
    sysmeta = json.loads((Path(p["system"]) / "system.json").read_text())
    k = p["solves_per_setup"]
    objective = ("t_solve (time of one solve; setup is free)" if k == 0 else
                 f"(t_setup + {k} t_solve) / {k}: time per solve when one setup serves {k} solves")
    lim = p.get("mem_limit_bytes")
    extra = ("| `squeue --me`, `sacct` | the Slurm queue (the worker is a Slurm job) |" if mode == "slurm" else
             "| `nvidia-smi` | the GPU |")
    note = "" if session == 1 else (
        f"\n**This is session {session}: you were relaunched.** Read `reasoning.md` and `sma-tools status` "
        "first, then continue. If you were already done, write the final summary and `DONE`.")
    t = string.Template((REPO_DIR / "prompts" / "agent.md").read_text())
    return t.substitute(
        budget=p["budget"], campaign=c.dir, system=p["system"], nrows=sysmeta.get("nrows"), nnz=sysmeta.get("nnz"),
        tol=f"{p['tol']:g}", mem_limit=f"{lim / 2**30:.1f} GiB" if lim else "the GPU capacity",
        objective=objective, libs=", ".join(p["libs"]), krylov=", ".join(p["krylov"]), amg=", ".join(p["amg"]),
        setup_repeats=p["setup_repeats"], solve_repeats=p["solve_repeats"], extra_tools=extra,
        repo=REPO_DIR, share=sma_work() / "share" / "sma", session_note=note,
        setup_rule=("The setup is not measured here (k = 0): setup work on the host is allowed." if k == 0 else
                    "The setup is measured (k > 0), so it must run on the GPU too: host setup options are "
                    "refused, library defaults included (e.g. MueLu's default \"serial\" aggregation coloring)."))


# ------------------------------------------------------------------ agents
def agent_cmd(agent: str, model: str | None, prompt_file: Path, settings_file: Path, cdir: Path,
              extra_dirs: list[Path] = ()) -> list[str]:
    if agent == "claude":
        cmd = ["claude", "-p", prompt_file.read_text(), "--permission-mode", "dontAsk",
               "--settings", str(settings_file), "--setting-sources", "", "--strict-mcp-config",
               *[x for d in extra_dirs for x in ("--add-dir", str(d))],
               "--tools", "Bash,Read,Write,Edit,Glob,Grep,WebFetch",
               "--output-format", "stream-json", "--verbose"]
        return cmd + (["--model", model] if model else [])
    if agent.startswith("cmd:"):
        tpl = agent[4:].format(prompt_file=shlex.quote(str(prompt_file)), workdir=shlex.quote(str(cdir)),
                               model=shlex.quote(model or ""))
        return ["bash", "-c", tpl]
    raise SystemExit(f"unknown agent {agent!r} (claude | cmd:TEMPLATE)")


# ------------------------------------------------------------------ worker on demand
class WorkerKeeper(threading.Thread):
    """Starts a worker when a request is pending and none is alive."""

    def __init__(self, c: Campaign, mode: str, machine: str, time_limit: str, idle: int, exclusive: bool, log):
        super().__init__(daemon=True)
        self.c, self.mode, self.machine, self.time_limit, self.idle, self.exclusive = c, mode, machine, time_limit, idle, exclusive
        self.log, self.stop_evt, self.starts = log, threading.Event(), 0

    def alive(self) -> bool:
        st = control.status(self.c.dir)
        w, hb = st.get("worker"), st.get("heartbeat")
        if not w or (hb and hb.get("state") == "exited"):
            return False
        if w["mode"] == "slurm":
            return st.get("slurm_state", "not in queue") != "not in queue"
        try:
            os.kill(w["pid"], 0)
            return True
        except ProcessLookupError:
            return False

    def run(self):
        reqs = self.c.dir / "queue" / "requests"
        while not self.stop_evt.wait(15):
            try:
                if reqs.exists() and any(reqs.glob("*.req")) and not self.alive():
                    info = control.start(self.c.dir, self.c.params["system"], self.mode, self.machine,
                                         self.time_limit, self.idle, self.exclusive)
                    self.starts += 1
                    self.log(f"worker started: {info}")
            except BaseException as e:  # keep the keeper alive; the request then times out as infra error
                self.log(f"worker start failed: {e}")


# ------------------------------------------------------------------ final outputs
def finalize(c: Campaign, end_reason: str, sessions: list[dict], log) -> None:
    from .report import report
    try:
        png, md = report(str(c.dir))
    except Exception as e:
        log(f"report failed: {e}")
        png = md = None
    best = c.best()
    if best:
        (c.dir / "best.yaml").write_text((c.dir / "runs" / best["id"] / "deck.yaml").read_text())
    j = c.journal()
    lines = [f"# Campaign {c.dir.name}", "",
             f"- system: `{c.params['system']}`; tolerance {c.params['tol']:g} (true relative residual); "
             f"solves per setup k = {c.params['solves_per_setup']}",
             f"- budget {c.params['budget']}, used {c.used()}, valid {sum(e['valid'] for e in j)}",
             f"- end: {end_reason}; sessions: {len(sessions)}", ""]
    if best:
        r = best["result"]
        lines += ["## Best configuration", "",
                  f"`best.yaml` (run `{best['id']}`, {best['deck']['lib']}): cost {best['cost'] * 1e3:.2f} ms, "
                  f"{r.get('iterations')} iterations, setup {r.get('t_setup_median', 0) * 1e3:.1f} ms, "
                  f"solve {r.get('t_solve_median', 0) * 1e3:.2f} ms", "", f"> {best.get('why', '')}", ""]
    lines += ["## Files", "",
              "| what | where |", "|---|---|",
              "| performance evolution (log) | [report.png](report.png) |",
              "| ranked table, failures, every configuration | [report.md](report.md) |",
              "| the agent's reasoning and final summary | [reasoning.md](reasoning.md) |",
              "| early-stop reason | [DONE](DONE) |" if (c.dir / "DONE").exists() else "| early-stop reason | (none) |",
              "| build suggestions | [suggestions.md](suggestions.md) |" if (c.dir / "suggestions.md").exists()
              else "| build suggestions | (none) |",
              "| every configuration: deck, native inputs, full result | `runs/<id>/` (ids in report.md) |",
              "| raw record | [journal.jsonl](journal.jsonl) |",
              "| agent transcripts | `transcripts/` |", "",
              "## Sessions", "", "| # | start | duration (min) | exit code | new runs |", "|---|---|---|---|---|"]
    lines += [f"| {s['n']} | {time.strftime('%Y-%m-%d %H:%M', time.localtime(s['start']))} | "
              f"{(s['end'] - s['start']) / 60:.1f} | {s['rc']} | {s['new_runs']} |" for s in sessions]
    (c.dir / "index.md").write_text("\n".join(lines) + "\n")
    log(f"final outputs: {c.dir / 'index.md'}")


# ------------------------------------------------------------------ main loop
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sma campaign", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--system", help="converted system directory (required for a new campaign)")
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--solves-per-setup", type=int, default=0)
    ap.add_argument("--libs", default="hypre,muelu,petsc")
    ap.add_argument("--krylov", default="cg,gmres,bicgstab")
    ap.add_argument("--amg", default="classical,aggregation")
    ap.add_argument("--mem-limit-gb", type=float, default=None)
    ap.add_argument("--setup-repeats", type=int, default=None)
    ap.add_argument("--solve-repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--agent", default="claude", help="claude | cmd:TEMPLATE")
    ap.add_argument("--model", default=None, help="model passed to the agent")
    ap.add_argument("--mode", choices=["slurm", "local"], default="slurm")
    ap.add_argument("--machine", default=os.environ.get("SMA_MACHINE"))
    ap.add_argument("--no-exclusive", action="store_true", help="worker on a shared node (tests only)")
    ap.add_argument("--worker-time", default="02:00:00", help="Slurm time limit of each worker job")
    ap.add_argument("--idle-timeout", type=int, default=600, help="worker exits after this many idle seconds")
    ap.add_argument("--max-sessions", type=int, default=20)
    a = ap.parse_args(argv)

    cdir = Path(a.dir).resolve()
    if not (cdir / "campaign.json").exists():
        if not a.system:
            raise SystemExit("new campaign: --system is required")
        Campaign.init(cdir, a.system, a.budget, a.tol, a.solves_per_setup, a.libs.split(","), a.krylov.split(","),
                      a.amg.split(","), a.mem_limit_gb, a.setup_repeats, a.solve_repeats, a.warmup)
    c = Campaign(cdir)
    system = Path(c.params["system"])
    if not (system / "analysis.json").exists():
        raise SystemExit(f"{system}/analysis.json missing: run sma-tools analyze on a compute node first")
    if a.mode == "slurm" and not a.machine:
        raise SystemExit("--machine (or SMA_MACHINE) is required in slurm mode")

    (cdir / ".claude").mkdir(exist_ok=True)
    sfile = cdir / ".claude" / "settings.json"
    sfile.write_text(json.dumps(settings(cdir, system, a.mode), indent=2) + "\n")
    tdir = cdir / "transcripts"
    tdir.mkdir(exist_ok=True)
    olog = open(cdir / "orchestrator.log", "a")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        olog.write(line + "\n")
        olog.flush()

    env = dict(os.environ, SMA_CAMPAIGN=str(cdir), SMA_WORK=str(sma_work()),
               PATH=f"{REPO_DIR / 'bin'}:{os.environ.get('PATH', '')}")
    env.pop("SMA_ALLOW_REBUILD", None)
    keeper = WorkerKeeper(c, a.mode, a.machine, a.worker_time, a.idle_timeout, not a.no_exclusive, log)
    agent_proc: list[subprocess.Popen] = []

    def on_signal(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_signal)

    sessions, end_reason, idle_sessions = [], "max sessions reached", 0
    first = 1 + len(list(tdir.glob("session-*.prompt.md")))
    (cdir / "queue").mkdir(exist_ok=True)
    (cdir / "queue" / "ORCHESTRATED").touch()
    keeper.start()
    log(f"campaign {cdir}: agent {a.agent} model {a.model} mode {a.mode} budget {c.params['budget']}, "
        f"used {c.used()}")
    try:
        for n in range(first, first + a.max_sessions):
            if (cdir / "DONE").exists():
                end_reason = "agent wrote DONE"
                break
            summary_only = c.remaining() <= 0
            if summary_only and n > first and sessions and sessions[-1].get("summary_only"):
                end_reason = "budget exhausted"
                break
            pfile = tdir / f"session-{n:02d}.prompt.md"
            text = prompt(c, a.mode, n)
            if summary_only:
                text += ("\n**The budget is exhausted.** Do not run anything: write the final summary in "
                         "`reasoning.md` and stop.\n")
            pfile.write_text(text)
            before = c.used()
            s = {"n": n, "start": time.time(), "summary_only": summary_only}
            log(f"session {n} start (budget used {before}/{c.params['budget']})")
            with open(tdir / f"session-{n:02d}.jsonl", "w") as out, open(tdir / f"session-{n:02d}.err", "w") as err:
                p = subprocess.Popen(agent_cmd(a.agent, a.model, pfile, sfile, cdir, add_dirs(cdir, system)), cwd=cdir, env=env,
                                     stdout=out, stderr=err, stdin=subprocess.DEVNULL, start_new_session=True)
                agent_proc[:] = [p]
                s["rc"] = p.wait()
                agent_proc.clear()
            s["end"], s["new_runs"] = time.time(), c.used() - before
            sessions.append(s)
            log(f"session {n} end: rc {s['rc']}, {s['new_runs']} new runs, used {c.used()}/{c.params['budget']}")
            if summary_only:
                end_reason = "budget exhausted"
                break
            idle_sessions = idle_sessions + 1 if s["new_runs"] == 0 else 0
            if (cdir / "DONE").exists():
                end_reason = "agent wrote DONE"
                break
            if idle_sessions >= NO_PROGRESS_SESSIONS:
                end_reason = f"{NO_PROGRESS_SESSIONS} consecutive sessions without a new run"
                break
            if c.remaining() <= 0 and "Final summary" in (cdir / "reasoning.md").read_text():
                end_reason = "budget exhausted"
                break
    except KeyboardInterrupt as e:
        end_reason = f"interrupted ({e})"
        log(end_reason)
    finally:
        for p in agent_proc:  # agent still running (interrupted): stop it and its children
            try:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=30)
            except Exception:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:
                    pass
        keeper.stop_evt.set()
        (cdir / "queue" / "ORCHESTRATED").unlink(missing_ok=True)
        try:
            log(f"worker stop: {control.stop(cdir)}")
        except BaseException as e:
            log(f"worker stop failed: {e} -- check squeue/scancel by hand")
        finalize(c, end_reason, sessions, log)
        log(f"end: {end_reason}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "campaign":
        sys.argv.pop(1)
    sys.exit(main())
