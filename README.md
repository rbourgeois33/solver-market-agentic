# solver-market-agentic

Tooling for an AI agent that **chooses and tunes GPU algebraic multigrid** (Krylov + AMG preconditioner)
for one linear system, **on the hardware where it will run**. The agent builds the libraries itself,
benchmarks configurations under a fixed budget, and logs why it tried each one.

GPU only, portable libraries only:

| library | AMG | Krylov | GPU backend |
|---|---|---|---|
| Trilinos 17.3 (Tpetra) | MueLu: aggregation (SA, emin, unsmoothed) | Belos (CG, GMRES, ...) | Kokkos 5.2 + cuSPARSE |
| hypre 3.2.1 | BoomerAMG: classical C/F selection | PCG, GMRES, FlexGMRES, BiCGSTAB | CUDA (device memory) |
| PETSc 3.26.1 | GAMG: aggregation (or classical) | KSP (cg, gmres, ...) | aijkokkos / aijcusparse |

Status: phase 1 (toolbox) and phase 2 (`sma campaign`: the agent runs the campaign). Phase 3: reference
systems + expert configurations + scoring.
Design and every decision taken so far: [docs/design.md](docs/design.md).

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

1. `sma-tools convert`: matrix + rhs to a raw binary CSR (fast to read, provenance + sha256).
2. `sma-tools analyze`: symmetry, diagonal, M-matrix pattern, row sums / near-null space, block size -> hints.
3. `sma-tools campaign-init`: budget, tolerance, cost model, allowed libraries / Krylov / AMG families.
4. `sma-tools worker start`: the GPU worker (exclusive node by default).
5. `sma-tools run DECK --why "hypothesis"`: one configuration = one unit of budget, appended to `journal.jsonl`.
6. `sma-tools report`: evolution chart + ranked table.
7. `sma-tools worker stop` (also happens on idle timeout and Slurm time limit: no orphan GPU job).

## Running a campaign with an agent

```bash
source machines/jean-zay/env.sh                     # puts sma and sma-tools on PATH
sma campaign $SMA_WORK/campaigns/c1 --system $SMA_WORK/data/mysys --budget 100 --tol 1e-8 \
    --agent claude --model <model>                  # or --agent cmd:'mycli --prompt-file {prompt_file}'
```
Runs on the login node (cluster) or the GPU machine (`--mode local`). It writes the agent's prompt
(`prompts/agent.md`, filled in) and permissions (`.claude/settings.json`), then runs agent sessions until the
agent writes `DONE`, the budget is spent, or 2 sessions in a row make no run (relaunched otherwise, with its
files). The GPU worker is started when a request is waiting and exits after 10 idle minutes; it is always
stopped at the end (also on Ctrl-C / errors). The same command resumes an interrupted campaign.

The agent may: run `sma-tools` (not worker start/stop), read-only shell commands, Slurm queries (cluster) or
`nvidia-smi` (local); read the campaign, the repo docs and decks, the installed libraries and their sources;
write only in the campaign directory (never the record: `journal.jsonl`, `runs/`, `queue/`); fetch the
libraries' documentation sites. It cannot change the build: proposals go to `suggestions.md`.

At the end, in the campaign directory: `index.md` (entry point), `report.png` (performance evolution),
`report.md` (ranking + every configuration, linked to `runs/<id>/`), `best.yaml`, `reasoning.md` (with the
agent's final summary), `transcripts/`.

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
Misspelled names are reported: hypre driver (unknown setter), MueLu (rejects with "did you mean"), PETSc
(`unused_options` in the result). `sma-tools params LIB` lists every accepted parameter (from the installed library).
GPU only: values that run on the host are refused before running (`sma/gpu_support.yaml`, with the doc or
source evidence; the coarsest-level solve is exempt), as are the values on a short do-not-try list.
The catalogue also gives the allowed values of string parameters (MueLu user guide tables + the values the
sources compare them to), flags declared keys that no code reads (no effect), and lists keys the code reads
without declaring them. MueLu `"half precision": true` is implemented by the driver (float hierarchy inside
the double-precision Krylov), as the Stratimikos adapter does upstream.

## How a configuration is measured

For each configuration the driver runs, on the matrix already resident on the GPU:

- **1 warm-up** setup + solve (not timed: first-touch allocations, memory pools, JIT, eigenvalue estimates);
- **S timed setups** (fresh preconditioner each time, only the last is kept), then **N timed solves** from x0 = 0
  on it, without reloading anything. Defaults: k = 0 -> S = 1, N = 10; k > 0 -> S = 5, N = 10
  (`campaign-init --setup-repeats / --solve-repeats`). `cudaDeviceSynchronize` / `Kokkos::fence` before each timer stops.
- Reported: min / median / max of setups and of solves (all values kept); the **median** is used.
- **Cost** (what the agent minimises) = `(t_setup + k t_solve) / k` with k = solves per setup;
  k = 0 (default) means the solve time only.
- **Valid** only if the **true** relative residual ‖b − Ax‖/‖b‖, recomputed on the host from the original matrix
  (independent of each library's own stopping test), is ≤ tol for every repetition, and peak GPU memory ≤ the limit.
- Peak GPU memory: sampled every ~1 ms with `cudaMemGetInfo` during setup + solve.

Benchmarks run on an exclusive node by default (`--no-exclusive` is for tests only).

## Output

`sma-tools report` writes, in the campaign directory:
- `report.png`: every configuration in submission order, cost on a log axis, colour per library, best-so-far line,
  invalid configurations in a separate strip;
- `report.md`: best per library, top configurations (cost, setup, solve, iterations, complexity, memory, hypothesis),
  failures with their reason, and the full list.

The raw record is `journal.jsonl` (deck, hypothesis, full result per configuration) and `runs/<id>/` (deck, native
inputs given to the library, result.json with the rows/nnz of every level, lib.log with the library's own output).
The agent's reasoning is in `reasoning.md`.

## Quick start (Jean Zay)

```bash
source machines/jean-zay/env.sh                   # modules, SMA_WORK=$WORK/sma-work, spack
# libraries + drivers + parameter catalogues: concretize + download on compil_h100 (internet), build on an
# H100 node; waits for both jobs. spack reuses what is installed.
sma-tools build

# a system
sbatch machines/jean-zay/prepare_system.slurm aij.mtx ones $SMA_WORK/data/mysys   # convert + analyze

# a campaign
# by hand (what the agent does):
sma-tools campaign-init $SMA_WORK/campaigns/c1 --system $SMA_WORK/data/mysys \
    --budget 100 --tol 1e-8 --solves-per-setup 0 --libs hypre,muelu,petsc
export SMA_CAMPAIGN=$SMA_WORK/campaigns/c1
sma-tools worker start --time 02:00:00            # exclusive H100 node; --no-exclusive for tests
sma-tools run decks/examples/muelu/cg_sa_cheby.yaml --why "baseline: SA + Chebyshev"
sma-tools status
sma-tools report
sma-tools worker stop
```

Libraries are built on the GPU node: configure-time test programs that link CUDA need the GPU driver.

## Layout

```
bin/             sma (campaign with an agent), sma-tools (toolbox)
prompts/         agent.md: the agent's instructions (filled in per campaign)
sma/             python toolbox (sma-tools); campaign.py (sma campaign); gpu_support.yaml (GPU rules)
  csr_io.py        binary CSR / vector format          convert.py   .mtx / PETSc binary -> binary
  analyze.py       matrix properties + hints             deck.py      YAML deck -> native library input
  campaign_state.py budget, journal, cost, run           report.py    chart + table
  worker/          dispatcher (GPU node) + start/stop/status
drivers/         C++ bench servers: common/ (CSR reader, timers, GPU memory sampler, true residual, JSON),
                 hypre/, petsc/, muelu/  (built by drivers/build.sh)
spack/           spack.yaml.in (pinned versions) + overlay repo (versions newer than spack's builtin recipes)
machines/        per machine: env.sh, packages.yaml.in (externals), Slurm scripts
decks/examples/  example decks per library
docs/            design.md (decisions), decks.md (deck reference), references.md (where parameters are documented)
bench/           (phase 3) reference systems, expert decks, results
```
