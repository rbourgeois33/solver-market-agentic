"""sma-tools: command line toolbox used by humans and agents."""
import argparse
import sys


def cmd_spack_env(a):
    from .spackenv import render
    print(render(a.machine, a.cuda_arch, a.jobs, a.force))


def cmd_convert(a):
    import json
    from .convert import convert
    print(json.dumps(convert(a.matrix, a.rhs, a.out, a.force), indent=2))


def cmd_analyze(a):
    import json
    from pathlib import Path
    from .analyze import analyze, summary
    from .convert import sha256
    out = Path(a.out) if a.out else Path(a.csr).with_name("analysis.json")
    h = sha256(Path(a.csr))
    old = json.loads(out.read_text()) if out.exists() else None
    if (not a.force and old and old.get("csr_sha256") == h
            and ("eigen_estimates" in old or not a.eigs)):
        print(summary(old))
        print(f"full report: {out} (already analysed: same A.csr; --force to redo)")
        return
    r = analyze(a.csr, eigs=a.eigs)
    r["csr_sha256"] = h
    out.write_text(json.dumps(r, indent=2) + "\n")
    print(summary(r))
    print(f"full report: {out}")


def _campaign(a):
    import os
    from .campaign_state import Campaign
    c = a.campaign or os.environ.get("SMA_CAMPAIGN")
    if not c:
        raise SystemExit("--campaign (or SMA_CAMPAIGN) is required")
    return Campaign(c)


def cmd_campaign_init(a):
    from .campaign_state import Campaign
    c = Campaign.init(a.dir, a.system, a.budget, a.tol, a.solves_per_setup, a.libs.split(","),
                      a.krylov.split(","), a.amg.split(","), a.mem_limit_gb, a.setup_repeats, a.solve_repeats, a.warmup)
    print(f"campaign ready: {c.dir}")


def cmd_run(a):
    import json
    from .campaign_state import short
    from .deck import DeckError
    c = _campaign(a)
    try:
        e = c.submit_and_wait(a.deck, a.why, timeout=a.timeout)
    except DeckError as err:
        raise SystemExit(f"deck rejected (no budget used): {err}")
    print(json.dumps(short(e, c), indent=2))


def cmd_status(a):
    import json
    from .campaign_state import short
    c = _campaign(a)
    j = c.journal()
    best = c.best()
    print(json.dumps({"budget": c.params["budget"], "used": len(j), "remaining": c.remaining(),
                      "valid_runs": sum(e["valid"] for e in j),
                      "best": short(best, c) if best else None, "params": c.params}, indent=2))


def cmd_worker(a):
    import json
    from .worker import control
    c = _campaign(a)
    if a.action == "start":
        r = control.start(c.dir, c.params["system"], a.mode, a.machine, a.time, a.idle_timeout,
                          exclusive=not a.no_exclusive)
    elif a.action == "stop":
        r = control.stop(c.dir)
    else:
        r = control.status(c.dir)
    print(json.dumps(r, indent=2))


def cmd_report(a):
    from .report import report
    png, md = report(str(_campaign(a).dir))
    print(f"chart: {png}\ntable: {md}")


def cmd_catalog(a):
    from .catalog import build
    build(a.launcher, a.libs.split(","))


def cmd_params(a):
    from .catalog import show
    print(show(a.lib, a.grep, a.section, a.full))


def main(argv=None):
    p = argparse.ArgumentParser(prog="sma-tools", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("spack-env", help="render the spack environment for this machine")
    s.add_argument("--machine", required=True)
    s.add_argument("--cuda-arch", help="e.g. 90; default: detected with nvidia-smi")
    s.add_argument("--jobs", type=int, default=24)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_spack_env)

    s = sub.add_parser("convert", help="convert matrix (+rhs) to the fast binary format")
    s.add_argument("--matrix", required=True, help=".mtx, PETSc binary or .csr")
    s.add_argument("--rhs", default="ones", help="file (.mtx, PETSc binary, .vec) or 'ones'/'zeros'")
    s.add_argument("--out", required=True, help="output system directory")
    s.add_argument("--force", action="store_true", help="convert even if --out already holds this system")
    s.set_defaults(func=cmd_convert)

    s = sub.add_parser("analyze", help="matrix properties and tuning hints")
    s.add_argument("csr", help="A.csr file")
    s.add_argument("--eigs", action="store_true", help="also Lanczos eigenvalue estimates (slower)")
    s.add_argument("--out", help="json report path (default: analysis.json next to A.csr)")
    s.add_argument("--force", action="store_true", help="analyse even if the report matches this A.csr")
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("campaign-init", help="create a campaign directory")
    s.add_argument("dir")
    s.add_argument("--system", required=True, help="converted system directory (A.csr, b.vec)")
    s.add_argument("--budget", type=int, default=100, help="max number of configurations")
    s.add_argument("--tol", type=float, default=1e-8, help="target TRUE relative residual")
    s.add_argument("--solves-per-setup", type=int, default=0,
                   help="k: cost = (t_setup + k t_solve)/k; 0 = solve time only")
    s.add_argument("--libs", default="hypre,muelu,petsc")
    s.add_argument("--krylov", default="cg,gmres,bicgstab")
    s.add_argument("--amg", default="classical,aggregation")
    s.add_argument("--mem-limit-gb", type=float, default=None, help="default: GPU capacity")
    s.add_argument("--setup-repeats", type=int, default=None,
                   help="timed setups per configuration (default: 1 if k = 0, else 5)")
    s.add_argument("--solve-repeats", type=int, default=10, help="timed solves on the last setup")
    s.add_argument("--warmup", type=int, default=1, help="untimed setup + solve before timing")
    s.set_defaults(func=cmd_campaign_init)

    s = sub.add_parser("run", help="benchmark ONE deck (uses 1 unit of budget)")
    s.add_argument("deck")
    s.add_argument("--why", required=True, help="hypothesis this configuration tests")
    s.add_argument("--campaign")
    s.add_argument("--timeout", type=float, default=900)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("status", help="budget and best configuration so far")
    s.add_argument("--campaign")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("report", help="evolution chart (report.png) + ranked table (report.md)")
    s.add_argument("--campaign")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("catalog", help="build the parameter catalogues from the installed libraries (GPU node)")
    s.add_argument("--launcher", default="", help="prefix for the drivers, e.g. 'srun --ntasks=1' in a Slurm job")
    s.add_argument("--libs", default="hypre,muelu,petsc")
    s.set_defaults(func=cmd_catalog)

    s = sub.add_parser("params", help="every parameter a library accepts (names, types, defaults, docs)")
    s.add_argument("lib", choices=["hypre", "muelu", "petsc"])
    s.add_argument("--grep", help="regex on name or doc")
    s.add_argument("--section", help="hypre: amg|pcg|gmres|fgmres|bicgstab; muelu: muelu|ifpack2|belos[:Solver]|"
                                     "amesos2; petsc: substring of the help section title")
    s.add_argument("--full", action="store_true", help="full doc text")
    s.set_defaults(func=cmd_params)

    s = sub.add_parser("worker", help="start/stop/status of the GPU bench worker")
    s.add_argument("action", choices=["start", "stop", "status"])
    s.add_argument("--campaign")
    s.add_argument("--mode", choices=["slurm", "local"], default="slurm")
    s.add_argument("--machine", default=None)
    s.add_argument("--time", default="02:00:00", help="slurm time limit")
    s.add_argument("--idle-timeout", type=int, default=1800)
    s.add_argument("--no-exclusive", action="store_true",
                   help="share the node (tests only; benchmarks are exclusive by default)")
    s.set_defaults(func=cmd_worker)

    a = p.parse_args(argv)
    if getattr(a, "machine", "x") is None:
        import os
        a.machine = os.environ.get("SMA_MACHINE")
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
