# Task

Find the fastest valid **GPU** configuration (Krylov method + AMG preconditioner) for one linear system
on this machine, within a budget of **$budget configurations**. You choose the library, the method and
every parameter. Reason explicitly: each configuration tests a hypothesis, and you record why.

## The campaign
- Campaign directory (your working directory): `$campaign`
- System: `$system` (n = $nrows, nnz = $nnz). Analysis (symmetry, M-matrix pattern, row sums, block
  structure, hints): `$system/analysis.json`. Read it first.
- Validity: true relative residual ||b - A x|| / ||b|| <= $tol, recomputed by the tools on the host
  for every timed solve; peak GPU memory <= $mem_limit.
- Objective (minimise): $objective
- Allowed: libraries $libs; Krylov families $krylov; AMG families $amg.
- Measurement (fixed, done by the tools): 1 untimed warm-up, $setup_repeats timed setup(s), $solve_repeats timed
  solves on the last setup; min / median / max are reported, the median is used. Configurations whose
  ranges overlap are a tie: do not draw conclusions from differences inside the noise.
- GPU only: values that run on the host are refused before running (not charged), see
  `sma-tools params LIB --gpu`. The coarsest-level solve is exempt (host direct solve or GPU smoother
  are both allowed there). $setup_rule A few values are on a do-not-try list (shown there too):
  they are refused.

## Tools (the only way to benchmark)
| command | what |
|---|---|
| `sma-tools status` | budget used / remaining, best so far, campaign parameters |
| `sma-tools params LIB [--grep REGEX] [--section S] [--full]` | every parameter LIB accepts (from the installed library), allowed values of string parameters, keys that have NO EFFECT in this version, and keys the code reads without declaring them (`source-only`) |
| `sma-tools params LIB --gpu` | which values run on the GPU (with the doc/source evidence), and the do-not-try list |
| `sma-tools gpu-check DECK` | classify a deck's values (no run, no budget) |
| `sma-tools shell 'SCRIPT'` | any shell script (loops, heredocs, sed, awk, python...), writable only in `decks/` and `work/`; e.g. generate deck variants in a loop |
| `sma-tools run DECK --why "hypothesis"` | benchmark one deck: **1 unit of budget**; prints the result |
| `sma-tools report` | evolution chart + ranked table (`report.png`, `report.md`) |
| `sma-tools build` | build the libraries + drivers + catalogues on this machine (spack reuses what is installed) |
| `sma-tools worker status` | state of the GPU worker (managed for you: do not start or stop it) |
$extra_tools
Shell: plain read-only commands (`cd`, `ls`, `cat`, `head`, `tail`, `grep`, `wc`, `sort`, `diff`...) can be
chained directly with `;`, `&&` and `|`. Anything else (`for`/`while` loops, heredocs that write files,
`sed`, `awk`, `python`, `find`) must go inside ONE `sma-tools shell '...'` call; outside it, it is refused.
Write several decks at once with one `sma-tools shell` script instead of one file at a time.
`sma-tools run` cannot run inside `sma-tools shell` (the record is read-only there): one run per call.

Each result shows the hierarchy (`hierarchy`: rows, nnz and nnz/row per level, finest first) and
`runs/<id>/lib.log`: the library's own output for that run. Raise the library's verbosity in the deck to
see more there (MueLu `"verbosity": high` prints the level table, smoothers and eigenvalue estimates;
hypre `PrintLevel`; PETSc `ksp_view`).
The catalogue (`sma-tools params`) lists everything the installed versions accept: before concluding
that a feature or option is not available, search it (`--grep`).

Before the first run, check that the libraries are built: if `sma-tools params hypre` says there is no
catalogue, or a run reports that a driver cannot start, run `sma-tools build` (it can take an hour).
You cannot change the build recipe. If you think it should change (a variant, a version, a patch),
write the proposal and its reason in `suggestions.md`: the maintainers implement it for later campaigns.

## Decks
One YAML file per configuration, in `decks/` of the campaign directory, using each library's own
parameter names (only the keys present are set; the rest keeps the library default). Reference:
`$repo/docs/decks.md`; examples: `$repo/decks/examples/`. Documentation of the installed versions
(user guides, parameter lists, sources): `$repo/docs/references.md` and `$share/src/`. You may also read
the official documentation online (hypre, PETSc, Trilinos sites).

## Record (required)
Keep `reasoning.md` up to date, in this structure:
1. **Analysis**: what the matrix is and what it implies for the methods.
2. **Plan**: how you will spend the budget.
3. One section per batch: **hypotheses -> runs (ids) -> results -> conclusions**. Write the hypotheses
   before running, the conclusions after.
4. **Final summary** (when you stop): the best configuration and why it wins, what did not work and
   why, ideas left untried, build suggestions.

Every `sma-tools run` needs `--why`: the hypothesis this configuration tests, in one sentence.

## Stopping
- The budget is a maximum, not a target. You may stop early when further runs are not expected to
  improve the result: write the reason in a file `DONE` (one paragraph), after the final summary.
- When the budget is exhausted, write the final summary and stop.
- You may be relaunched in a new session (with this prompt and your files): then read `reasoning.md`
  and `sma-tools status` first, and continue where you stopped.

## Rules
- Benchmark only through `sma-tools run`. Do not run the drivers or the libraries directly.
- Write only inside the campaign directory, and never edit `journal.jsonl`, `campaign.json`, `runs/`
  or `queue/` (they are the record).
$session_note
