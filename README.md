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

Status: **phase 1 (toolbox) done and tested on an H100**. Phase 2 (`sma campaign`: launching the agent with a
fixed prompt) and phase 3 (reference systems + expert configurations + scoring) are next.
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

1. `sma-tools convert`: matrix + rhs to a raw binary CSR (read in 0.05 s instead of 0.6 s for a 475 MB .mtx).
2. `sma-tools analyze`: symmetry, diagonal, M-matrix pattern, row sums / near-null space, block size -> hints.
3. `sma-tools campaign-init`: budget, tolerance, cost model, allowed libraries / Krylov / AMG families.
4. `sma-tools worker start`: the GPU worker (exclusive node by default).
5. `sma-tools run DECK --why "hypothesis"`: one configuration = one unit of budget, appended to `journal.jsonl`.
6. `sma-tools report`: evolution chart + ranked table.
7. `sma-tools worker stop` (also happens on idle timeout and Slurm time limit: no orphan GPU job).

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
(`unused_options` in the result).

## How a configuration is measured

For each configuration the driver runs, on the matrix already resident on the GPU:

- **1 warm-up** setup + solve (not timed: first-touch allocations, memory pools, JIT, eigenvalue estimates);
- **R timed repetitions** (default 3, `campaign-init --repeats`), each = a fresh preconditioner **setup** followed
  by one **solve** from x0 = 0. Setup and solve are both repeated, and each is timed separately
  (`cudaDeviceSynchronize` / `Kokkos::fence` before stopping each timer).
- Reported: the **median** of the R setups and the median of the R solves (all R values are kept in the result).
- **Cost** (what the agent minimises) = `(t_setup + k t_solve) / k` with k = solves per setup;
  k = 0 (default) means the solve time only.
- **Valid** only if the **true** relative residual ‖b − Ax‖/‖b‖, recomputed on the host from the original matrix
  (independent of each library's own stopping test), is ≤ tol for every repetition, and peak GPU memory ≤ the limit.
- Peak GPU memory: sampled every ~1 ms with `cudaMemGetInfo` during setup + solve.

Noise: on a shared node, re-running the same configuration varied by ±3-5%. Rankings closer than that need an
exclusive node (the default) and more repetitions.

## Output

`sma-tools report` writes, in the campaign directory:
- `report.png`: every configuration in submission order, cost on a log axis, colour per library, best-so-far line,
  invalid configurations in a separate strip;
- `report.md`: best per library, top configurations (cost, setup, solve, iterations, complexity, memory, hypothesis),
  failures with their reason, and the full list.

The raw record is `journal.jsonl` (deck, hypothesis, full result per configuration) and `runs/<id>/` (deck, native
inputs given to the library, result.json). The agent's reasoning is in `reasoning.md`.

## First result (2026-11-26, H100, TRUST 2D pressure matrix n = 2.59 M, tol 1e-8, solve time only)

| library | default config | best found | best config |
|---|---|---|---|
| MueLu + CG | 4150 ms | **~23 ms** | SA, Chebyshev(1), KLU2 coarse ≤ 2000 rows |
| hypre PCG + BoomerAMG | 46.6 ms | 36.5 ms | PMIS, ext+i, PMaxElmts 4, Jacobi w = 0.9 |
| PETSc CG + GAMG | 47.1 ms | 43.2 ms | agg, Chebyshev/Jacobi 2 sweeps |

65 of 100 configurations used (Claude in a session played the agent; phase 2 will automate this).

## Quick start (Jean Zay)

```bash
source machines/jean-zay/env.sh                   # modules, SMA_WORK=$WORK/sma-work, spack
# libraries, once per machine: resolve + download on compil_h100 (internet), BUILD ON AN H100 NODE
sbatch machines/jean-zay/concretize.slurm
sbatch machines/jean-zay/build_libs.slurm         # first time only: fills the spack source cache
sbatch machines/jean-zay/build_libs_gpu.slurm --rebuild   # GPU stack + the 3 drivers, on the GPU node

# a system
sbatch machines/jean-zay/prepare_system.slurm aij.mtx ones $SMA_WORK/data/mysys   # convert + analyze

# a campaign
sma-tools campaign-init $SMA_WORK/campaigns/c1 --system $SMA_WORK/data/mysys \
    --budget 100 --tol 1e-8 --solves-per-setup 0 --libs hypre,muelu,petsc
export SMA_CAMPAIGN=$SMA_WORK/campaigns/c1
sma-tools worker start --time 02:00:00            # exclusive H100 node; --no-exclusive for tests
sma-tools run decks/examples/muelu/cg_sa_cheby.yaml --why "baseline: SA + Chebyshev"
sma-tools status
sma-tools report
sma-tools worker stop
```

Why build on the GPU node: configure-time test programs that link CUDA cannot run without the GPU driver;
on a GPU-less node PETSc silently assumed 64-bit BLAS integers and crashed later. Jean Zay notes:
[machines/jean-zay](machines/jean-zay).

## Layout

```
sma/             python toolbox (sma-tools)
  csr_io.py        binary CSR / vector format          convert.py   .mtx / PETSc binary -> binary
  analyze.py       matrix properties + hints             deck.py      YAML deck -> native library input
  campaign_state.py budget, journal, cost, run           report.py    chart + table
  worker/          dispatcher (GPU node) + start/stop/status
drivers/         C++ bench servers: common/ (CSR reader, timers, GPU memory sampler, true residual, JSON),
                 hypre/, petsc/, muelu/  (built by drivers/build.sh)
spack/           spack.yaml.in (pinned versions) + overlay repo (versions newer than spack's builtin recipes)
machines/        per machine: env.sh, packages.yaml.in (externals), Slurm scripts
decks/examples/  example decks per library
docs/            design.md (decisions), decks.md (deck reference)
bench/           (phase 3) reference systems, expert decks, results
```
