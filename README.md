# solver-market-agentic

Tooling for an AI agent that **chooses and tunes GPU algebraic multigrid** (Krylov + AMG) for a
given linear system, on the hardware where it will run. GPU only, portable libraries only:

| library | AMG | Krylov used |
|---|---|---|
| Trilinos 17.3 (Tpetra) | MueLu (aggregation) | Belos |
| hypre 3.2.1 | BoomerAMG (classical C/F selection) | hypre PCG / GMRES / FlexGMRES / BiCGSTAB |
| PETSc 3.26.1 | GAMG (aggregation, or classical) | KSP |

Design decisions: [docs/design.md](docs/design.md). Deck format: [docs/decks.md](docs/decks.md).

## Status
Phase 1 (toolbox): binary format + converter, matrix analysis, spack environment, 3 GPU bench
drivers (persistent servers), YAML decks, `sma-tools run` with budget and journal, GPU worker
(Slurm or local). Phase 2: `sma campaign` (agent orchestration). Phase 3: benchmark DB + scoring.

## Quick start (Jean Zay)
```bash
source machines/jean-zay/env.sh                  # modules + SMA_WORK + spack
# libraries (once per machine): concretize, then build, both on compil_h100
sbatch machines/jean-zay/concretize.slurm
sbatch machines/jean-zay/build_libs.slurm
drivers/build.sh                                 # in a job on compil_h100 / an H100 node

# a system: matrix + rhs -> fast binary (+ provenance, sha256)
sma-tools convert --matrix aij.mtx --rhs ones --out $SMA_WORK/data/mysys
sma-tools analyze $SMA_WORK/data/mysys/A.csr     # properties + hints (--eigs: slower)

# a campaign: budget, tolerance on the true residual, cost model, allowed methods
sma-tools campaign-init $SMA_WORK/campaigns/c1 --system $SMA_WORK/data/mysys \
    --budget 100 --tol 1e-8 --solves-per-setup 0 --libs hypre,muelu,petsc
export SMA_CAMPAIGN=$SMA_WORK/campaigns/c1
sma-tools worker start --mode slurm --time 02:00:00   # exclusive H100 node, idle timeout 30 min
sma-tools run decks/examples/hypre/pcg_gpu_classic.yaml --why "baseline GPU BoomerAMG"
sma-tools status
sma-tools worker stop                                 # STOP file + scancel
```

## Layout
```
sma/            python toolbox (sma-tools): csr_io, convert, analyze, deck, campaign_state, worker/
drivers/        C++ bench servers: common/ (CSR reader, timers, GPU mem sampler, true residual),
                hypre/, petsc/, muelu/
spack/          spack.yaml.in + overlay repo (newer versions than spack's builtin recipes)
machines/       per-machine env.sh, externals (packages.yaml.in), Slurm scripts
decks/examples/ example decks per library
bench/          (phase 3) reference systems + expert decks + results
```
