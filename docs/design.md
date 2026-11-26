# Design

## Goal
Given a matrix, an rhs and campaign constraints, an agent (Claude or a local LLM) builds GPU AMG libraries on the
target machine and tunes Krylov + AMG (MueLu, BoomerAMG, GAMG) under a budget of N configurations, logging its
reasoning. A benchmark set of reference systems with expert configurations measures whether it beats experts.

## Principles
- **Tune for the hardware you run on**: libraries are built on the benchmark node, configurations are measured there.
- **GPU only, portable libraries only**: Trilinos (Tpetra stack; Epetra is deprecated), hypre, PETSc.
- **Native parameters**: decks use each library's documented parameter names; nothing is abstracted or renamed.
- **The matrix is read once**: drivers are persistent servers; the disk read is paid once per library switch.
- **Budget and record are enforced by the tools**, not by the agent's good will.

## Campaign parameters
| parameter | default |
|---|---|
| budget (configurations) | 100 |
| tolerance on the true relative residual | 1e-8 |
| solves per setup k | 0 (solve time only) |
| GPU memory limit | GPU capacity |
| libraries | hypre, muelu, petsc |
| Krylov families | cg, gmres, bicgstab |
| AMG families | classical (C/F selection), aggregation |

## Decisions
| topic | decision |
|---|---|
| Objective | runtime: cost = (t_setup + k t_solve)/k; k = 0 means t_solve only. Memory is a constraint only. |
| Validity | true relative residual ‖b − Ax‖/‖b‖ recomputed on the host from the original matrix ≤ tol for every repetition, and peak GPU memory ≤ limit. |
| Measurement | 1 untimed warm-up, then R repetitions of (fresh setup + one solve from x0 = 0); medians reported, all values kept; device sync before each timer stops. |
| Krylov | each library uses its own (Belos, hypre ParCSR solvers, KSP). |
| Drivers | 3 C++ servers (`drivers/`): load matrix once, serve requests over stdin, write result JSON. |
| Matrix format | own raw binary CSR (128-byte header, 64-byte aligned arrays: int64 rowptr, int32/64 colind, float64 values); converters from .mtx and PETSc binary. |
| Decks | YAML: `lib`, `solver`, `solver_params`, `precond_params`, `backend_params`, native names, only present keys set. Translated by `sma/deck.py` into ParameterList XML (MueLu), option strings (PETSc), setter lines (hypre). |
| Worker | python dispatcher on the GPU node; exactly one driver alive at a time (it owns the GPU during a measurement); each driver is its own 1-rank job step. Switching library reloads the matrix from the binary file. |
| Cluster mode | orchestrator + agent on the login node (GPU nodes are offline); worker = Slurm job on an exclusive GPU node, fed by a queue directory on the shared filesystem. |
| Local mode | same worker as a local process. |
| No orphan GPU job | the worker stops on a STOP file, on an idle timeout, and at its Slurm time limit; the orchestrator writes STOP and cancels the job when the agent exits for any reason. |
| Budget | `sma-tools run` is the only way to benchmark; each run needs `--why`; refused once the budget is spent. A configuration that ran is charged even if it failed; infrastructure failures (driver cannot start, worker down) are journaled with `charged: false`. |
| Record | `journal.jsonl` (deck, hypothesis, full result per configuration), `runs/<id>/`, agent's `reasoning.md`, agent transcript (phase 2). |
| Agent launch (phase 2) | CLI adapters: `--agent claude` runs `claude -p` with a fixed prompt and restricted settings in the campaign directory; `--agent cmd:'<template>'` for any other CLI agent. |
| Expert data (phase 3) | `bench/` in this repo; the agent is told not to read it and Claude settings deny the path. |
| Matrix store (phase 3) | public Hugging Face dataset: zstd-compressed binary CSR + manifest with sha256. |
| Build | spack environment `spack/spack.yaml.in`, pinned versions, overlay repo `spack/repo` for versions newer than spack's recipes. The agent may edit its copy of the environment and must log every change in `build_log.md`. |
| Versions | hypre 3.2.1, PETSc 3.26.1 (+kokkos), Trilinos 17.3.0 (Tpetra, MueLu, Belos, Ifpack2, Amesos2; +cusparse, +float), Kokkos / Kokkos Kernels 5.2.2. |
| BLAS/LAPACK | netlib-lapack (only CPU-side small dense work). |
| GPUs | 1 GPU / 1 MPI rank in v1; drivers use distributed objects so multi-GPU is a later option. |

## Build rules
- Resolve versions and download sources where there is internet; **build on a GPU node**: configure-time test
  programs that link CUDA need the GPU driver.
- Spack does not expand environment variables in external paths: `sma-tools spack-env` writes them in from the
  loaded modules.
- Drivers keep their RPATH and are configured from scratch on every build (library install paths change).
