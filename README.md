# solver-market-agentic

An AI agent **chooses and tunes GPU algebraic multigrid** (Krylov + AMG preconditioner) for one linear
system, **on the machine where it will run**. You give it a matrix and a budget of configurations; it
benchmarks configurations one by one on a GPU node, writes why it tried each one, and gives you the
fastest valid one.

| library | AMG | Krylov | GPU backend |
|---|---|---|---|
| Trilinos 17.3 (Tpetra) | MueLu: aggregation (SA, emin, ...) | Belos (CG, GMRES, ...) | Kokkos 5.2 + cuSPARSE |
| hypre 3.2.1 | BoomerAMG: classical C/F selection | PCG, GMRES, FlexGMRES, BiCGSTAB | CUDA (device memory) |
| PETSc 3.26.1 | GAMG: aggregation (or classical) | KSP (cg, gmres, ...) | aijkokkos / aijcusparse |

Only GPU options are explored (setup excepted when it is not measured), see [GPU-only rules](#gpu-only-rules).
Design and every decision taken so far: [docs/design.md](docs/design.md).

Results can be shared in a public dataset (matrices + every configuration tested + best per GPU), managed
separately from campaigns with `sma-dataset`: see [docs/dataset.md](docs/dataset.md).

---

## Quick start (Jean Zay)

Everything below is run **from the repository root, on a login node**. Heavy work always goes to Slurm
jobs; the login node only runs the orchestrator and the agent.

### 0. Every new shell

```bash
cd $WORK/solver-market-agentic
source machines/jean-zay/env.sh        # modules, PATH (sma, sma-tools), SMA_WORK=$WORK/sma-work
```

### 1. Build the libraries (once, ~1 h; skip if already done)

```bash
sma-tools build      # concretize + download on compil_h100, then build on an H100 node; waits for both jobs
```
Spack reuses what is installed, so re-running it is cheap. Check: `sma-tools params muelu | head` prints
parameters (no "no catalogue" message).

### 2. Prepare a system (matrix + right-hand side)

```bash
sbatch machines/jean-zay/prepare_system.slurm $WORK/aij.mtx ones $SMA_WORK/data/mysys
```
- matrix: `.mtx` (MatrixMarket) or PETSc binary; rhs: a file (`.mtx`, PETSc binary) or `ones` / `zeros`;
- output: `$SMA_WORK/data/mysys/` with `A.csr`, `b.vec`, `system.json` and `analysis.json` (symmetry,
  M-matrix pattern, near-null space... with hints for the agent);
- a few minutes on an H100 node (never on the login node: the analysis is heavy). Log in
  `$SMA_WORK/logs/sma-prepare-<jobid>.out`.

### 3. Run a campaign with the agent

Run it inside **tmux** (or screen): the campaign stops if your terminal closes (SIGHUP), and it can last
from minutes to hours.

```bash
tmux new -s sma                        # later: tmux attach -t sma
source machines/jean-zay/env.sh
sma campaign $SMA_WORK/campaigns/mysys-k0 --system $SMA_WORK/data/mysys \
    --budget 40 --tol 1e-8 --solves-per-setup 0
```
That is all. It starts the agent (Claude Code CLI, `claude -p`, account default model), starts a GPU
worker job when the agent submits its first configuration, relaunches the agent if a session ends early,
stops the worker at the end, and writes the final report. Prerequisite: `claude` is installed and logged in
on the login node.

Common variants:

| you want | add |
|---|---|
| a specific model | `--model claude-opus-5-5` |
| setup cost included (e.g. 10 solves per setup) | `--solves-per-setup 10` |
| only some libraries | `--libs muelu,hypre` |
| a cheaper, noisier test (shared GPU node) | `--no-exclusive` |
| a memory cap | `--mem-limit-gb 40` |
| another CLI agent | `--agent cmd:'mycli --prompt-file {prompt_file}'` |
| tell the agent something (e.g. a target) | `--note "The record to beat is 19 ms per solve."` (kept on resume) |

Budget: 1 configuration = 1 unit. The agent may stop early (it writes `DONE` with the reason). For
reference, a 40-run campaign on a 2.6M-row 2D Poisson matrix took ~17 min of agent time and ~10 min of GPU.
By default the worker runs on an **exclusive** H100 node (4 GPUs billed) for clean timings.

### 4. Follow it, stop it, resume it

From another terminal (with `env.sh` sourced):

```bash
C=$SMA_WORK/campaigns/mysys-k0
tail -f $C/orchestrator.log                     # sessions, worker start/stop, end reason
sma-tools status --campaign $C                  # budget used, best so far
less $C/reasoning.md                            # the agent's hypotheses and conclusions, batch by batch
sma-tools report --campaign $C                  # refresh report.png / report.md at any time
squeue -u $USER                                 # the worker job (sma-worker)
```
- **Stop**: Ctrl-C in the tmux window (or `kill` the `sma campaign` process). The agent and the worker
  job are always stopped with it.
- **Resume**: run the same `sma campaign <dir> ...` command again. The campaign parameters (system, budget,
  tolerance, k, libraries) come from `<dir>/campaign.json`; repeat the run options you used (`--model`,
  `--no-exclusive`, `--agent`). A campaign whose agent wrote `DONE` is finished: delete `DONE` to continue it.

### 5. Read the results

Open `$C/index.md`: it links everything.

| file | content |
|---|---|
| `best.yaml` | the fastest valid configuration (a deck you can re-run) |
| `report.png` | cost of every configuration in submission order, per library, best-so-far line |
| `report.md` | best per library, ranked table, failures with their reason, every configuration |
| `reasoning.md` | analysis, plan, hypotheses -> runs -> conclusions, final summary |
| `suggestions.md` | what the agent would change in the build (versions, variants, patches) |
| `runs/<id>/` | one configuration: `deck.yaml`, native inputs, `result.json` (all timings, iterations, residual, memory, rows/nnz per level), `lib.log` (the library's own output) |
| `journal.jsonl` | the raw record (deck, hypothesis, full result per configuration) |
| `transcripts/` | the agent sessions (stream-json) |

---

## Running a campaign by hand (no agent)

The agent uses exactly these tools; you can too.

```bash
C=$SMA_WORK/campaigns/byhand
sma-tools campaign-init $C --system $SMA_WORK/data/mysys --budget 20 --tol 1e-8 --solves-per-setup 0
export SMA_CAMPAIGN=$C                             # or --campaign $C on every command
sma-tools worker start --time 02:00:00            # exclusive H100 node (--no-exclusive: shared, tests)
sma-tools run decks/examples/muelu/cg_sa_cheby.yaml --why "baseline: SA + Chebyshev"
sma-tools status
sma-tools report
sma-tools worker stop                              # also stops by itself after 30 idle minutes
```

| command | what |
|---|---|
| `sma-tools run DECK --why "..."` | benchmark one deck (1 unit of budget); prints the result, with the hierarchy per level |
| `sma-tools params LIB [--grep RE] [--section S] [--full] [--all]` | every parameter LIB accepts, from the installed library: type, default, doc, allowed values (tagged with GPU status), NO EFFECT keys, source-only keys |
| `sma-tools params LIB --gpu` | GPU rules: which values run on the GPU (with evidence), host setup options, do-not-try list |
| `sma-tools gpu-check DECK` | classify a deck's values without running it |
| `sma-tools shell 'SCRIPT'` | a shell script confined to the campaign's `decks/` and `work/` (e.g. generate variants in a loop) |
| `sma-tools status` / `report` | budget and best so far / chart + ranked table |
| `sma-tools worker start\|stop\|status` | the GPU worker job |
| `sma-tools catalog [--parse-only]` | rebuild the parameter catalogues (after a build; `--parse-only` needs no GPU) |

Examples: `sma-tools params muelu --grep "coloring"`, `sma-tools params hypre --section amg --grep Relax`,
`sma-tools params petsc --grep gamg_threshold`.

---

## Decks: one configuration, in each library's own words

A deck uses the parameter names the library documents. Only the keys present are set; everything else keeps the
library default. Full reference: [docs/decks.md](docs/decks.md); examples: [decks/examples](decks/examples).

```yaml
lib: muelu
solver: Pseudoblock CG                  # Belos name
solver_params: {Maximum Iterations: 1000}
precond_params:                         # MueLu ParameterList (quote keys: they contain ": ")
  "multigrid algorithm": sa
  "smoother: type": CHEBYSHEV
  "smoother: params": {"chebyshev: degree": 1, "chebyshev: ratio eigenvalue": 7.0}
  "coarse: max size": 2000
  "coarse: type": KLU2
  "half precision": true                # float preconditioner inside the double-precision CG
  "verbosity": high                     # more detail in runs/<id>/lib.log
```
```yaml
lib: hypre
solver: pcg
solver_params: {MaxIter: 1000, TwoNorm: 1}   # HYPRE_ParCSRPCGSet<Name>
precond_params: {CoarsenType: 8, InterpType: 6, PMaxElmts: 4, RelaxType: 7, RelaxWt: 0.9}  # HYPRE_BoomerAMGSet<Name>
```
```yaml
lib: petsc
solver: cg
solver_params: {ksp_max_it: 1000, ksp_norm_type: unpreconditioned}
precond_params: {pc_type: gamg, mg_levels_ksp_type: chebyshev, mg_levels_pc_type: jacobi, mg_levels_ksp_max_it: 2}
backend_params: {mat_type: aijkokkos, vec_type: kokkos}
```
- Misspelled names are reported: hypre driver (unknown setter), MueLu (rejects with "did you mean"), PETSc
  (`unused_options` in the result).
- MueLu `"half precision": true` is implemented by our driver (float copy of A, float hierarchy, applied
  inside the double-precision Krylov), as the Stratimikos adapter does upstream; only the preconditioner
  is in float, the convergence test and the true residual stay in double.

## GPU-only rules

Checked by `sma-tools run` before running; refused decks are **not charged**. Rules and evidence:
`sma/gpu_support.yaml`, shown by `sma-tools params LIB --gpu`.

- Values that run on the host are refused (e.g. MueLu's default smoother, Symmetric Gauss-Seidel). Rules
  apply to effective values, library defaults included. Values with no rule are allowed.
- The coarsest-level solve is exempt (host direct solve or GPU smoother).
- Setup on the host: **allowed when the setup is not measured (k = 0), refused when it is (k > 0)**,
  defaults included (MueLu's default `"aggregation: coloring algorithm": serial` colors on the host).
- A short do-not-try list (no justification given to the agent).

## How a configuration is measured

For each configuration the driver runs, on the matrix already resident on the GPU:

- **1 warm-up** setup + solve (not timed: first-touch allocations, memory pools, JIT, eigenvalue estimates);
- **S timed setups** (fresh preconditioner each time, only the last is kept), then **N timed solves** from x0 = 0
  on it, without reloading anything. Defaults: k = 0 -> S = 1, N = 10; k > 0 -> S = 5, N = 10
  (`--setup-repeats / --solve-repeats`). `cudaDeviceSynchronize` / `Kokkos::fence` before each timer stops.
- Reported: min / median / max of setups and of solves (all values kept); the **median** is used.
- **Cost** (what the agent minimises) = `(t_setup + k t_solve) / k` with k = solves per setup;
  k = 0 (default) means the solve time only.
- **Valid** only if the **true** relative residual ‖b − Ax‖/‖b‖, recomputed on the host from the original matrix
  (independent of each library's own stopping test), is ≤ tol for every repetition, and peak GPU memory ≤ the limit.
- Peak GPU memory: sampled every ~1 ms with `cudaMemGetInfo` during setup + solve.

## How a campaign works

```
            login node                                    GPU node (Slurm job, or local)
 ┌──────────────────────────────┐               ┌─────────────────────────────────────────┐
 │ agent (Claude / local LLM)   │               │ worker (sma/worker/dispatcher.py)       │
 │   reads docs + lib sources   │   queue dir   │   one driver server alive at a time:    │
 │   writes decks (YAML)        │  on Lustre    │   sma_muelu | sma_hypre | sma_petsc     │
 │   sma-tools run deck --why ──┼──requests────►│   matrix read + uploaded ONCE,          │
 │   ◄── result (cost, its, ...)┼───results─────┼── then serves many configurations      │
 │ journal.jsonl, reasoning.md  │               │                                         │
 └──────────────────────────────┘               └─────────────────────────────────────────┘
```

`sma campaign` writes the agent's prompt (`prompts/agent.md`, filled in) and permissions
(`.claude/settings.json`), then runs agent sessions until the agent writes `DONE`, the budget is spent, or
2 sessions in a row make no run. The GPU worker starts when a request is waiting and exits after 10 idle
minutes; it is always stopped at the end (also on Ctrl-C / errors).

What the agent may do: run `sma-tools` (not worker start/stop); read-only shell commands (`cd`, `cat`,
`grep`, ... chainable), anything else inside `sma-tools shell`; Slurm queries (cluster) or `nvidia-smi`
(local); read the campaign, the repo, the installed libraries and their sources; write only in the campaign
directory (never the record: `journal.jsonl`, `runs/`, `queue/`); fetch the libraries' documentation sites.
It cannot change the build: proposals go to `suggestions.md`.

## Troubleshooting

| symptom | what to do |
|---|---|
| `no catalogue for ...` | `sma-tools build` (or `sbatch machines/jean-zay/catalog.slurm` if only the catalogues are missing) |
| a run says a driver cannot start | look at `$C/queue/drivers/<lib>.stderr.log`; rebuild the drivers: `sbatch machines/jean-zay/build_drivers.slurm` |
| the campaign waits a long time before the first run | the worker job is queued: `squeue -u $USER` (waiting is normal) |
| `--campaign (or SMA_CAMPAIGN) is required` | add `--campaign $C` or `export SMA_CAMPAIGN=$C` |
| a deck is "refused (not charged)" | a host-only / do-not-try value: `sma-tools gpu-check DECK` says which |

## Layout

```
bin/             sma (campaign with an agent), sma-tools (toolbox), sma-dataset (public dataset, separate)
prompts/         agent.md: the agent's instructions (filled in per campaign)
sma/             python toolbox (sma-tools); campaign.py (sma campaign); gpu_support.yaml (GPU rules)
  csr_io.py        binary CSR / vector format          convert.py   .mtx / PETSc binary -> binary
  analyze.py       matrix properties + hints             deck.py      YAML deck -> native library input
  campaign_state.py budget, journal, cost, run           report.py    chart + table
  catalog.py       parameter catalogues                  srcscan.py   values / hidden keys from the sources
  dataset.py, dataset_cli.py   public dataset (sma-dataset): matrices, runs, best, card, Hugging Face
  worker/          dispatcher (GPU node) + start/stop/status
drivers/         C++ bench servers: common/ (CSR reader, timers, GPU memory sampler, true residual, JSON),
                 hypre/, petsc/, muelu/  (built by drivers/build.sh)
spack/           spack.yaml.in (pinned versions) + overlay repo (versions newer than spack's builtin recipes)
machines/        per machine: env.sh, packages.yaml.in (externals), Slurm scripts
decks/examples/  example decks per library
docs/            design.md (decisions), decks.md (deck reference), references.md (where parameters are documented),
                 dataset.md (the public dataset and sma-dataset)
bench/           (phase 3) reference systems, expert decks, results
```
