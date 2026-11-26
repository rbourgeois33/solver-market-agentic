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
| Measurement | 1 untimed warm-up (setup + solve), then S timed setups (only the last is kept), then N timed solves from x0 = 0 on it; the matrix is never reloaded. Defaults: k = 0 -> S = 1, N = 10; k > 0 -> S = 5, N = 10 (setup is repeated only when it is in the cost). Min / median / max reported, the median is used; device sync before each timer stops; true residual checked after every timed solve. |
| Parameters | every parameter the installed library documents is accepted; catalogue from the library itself (`sma-tools params`). |
| GPU only | values that run on the host are refused (`sma/gpu_support.yaml`: rule + evidence per value; unknown = allowed). Exception: the coarsest-level solve (host direct solve or GPU smoother). When the setup is not measured (k = 0), host setup options are refused too; when k > 0 they are allowed (their time counts). |
| Ruled out | methods that run but are clearly bad on GPU, refused with the evidence (our runs): MueLu unsmoothed aggregation, MT Gauss-Seidel smoothers. No runtime host/device-traffic check. |
| Ordering | the system is benchmarked in the application's ordering: no reordering (RCM etc.), since the application solves in its own ordering. |
| Agent shell | `sma-tools shell 'SCRIPT'`: shell scripts (loops, sed, python) run under bubblewrap, writable only in the campaign's `decks/` and `work/`, no network; the record stays read-only. (Claude Code's own sandbox needs socat, absent on Jean Zay.) |
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
| Agent launch | `sma campaign`: `--agent claude` runs `claude -p` (model = `--model`) with `prompts/agent.md` filled in, in the campaign directory, permission mode dontAsk + generated settings, no user settings; `--agent cmd:'<template>'` for any other CLI agent. |
| Agent permissions | sma-tools (not worker start/stop, campaign-init, catalog), read-only shell commands, squeue/sacct (cluster) or nvidia-smi (local); read: campaign, system, repo docs/decks/prompts/sma, installed libraries + sources + catalogues, build log; write: campaign directory except the record (journal, runs, queue, transcripts); web: hypre / PETSc / Trilinos documentation sites, github, gitlab. Benchmarks only through `sma-tools run`. |
| Sessions | relaunched until: `DONE` (agent's early stop, with its reason), budget spent (+ a summary-only session if needed), 2 consecutive sessions without a run, or `--max-sessions`. Transcripts kept. |
| Worker on demand | started by the orchestrator when a request is pending and no worker is alive; exits after `--idle-timeout` (10 min); each job ≤ `--worker-time`. `sma-tools run` waits for it under the orchestrator. Stopped at the end in all cases. |
| Final outputs | `index.md`, `report.png` (log evolution), `report.md` (ranking, failures, every configuration linked to `runs/<id>/`), `best.yaml`, `reasoning.md` (final summary), `suggestions.md`, `transcripts/`. |
| Record of the agent | `reasoning.md` in a fixed structure: analysis, plan, per batch hypotheses -> runs -> results -> conclusions, final summary. |
| Expert data (phase 3) | `bench/` in this repo (public information; no access restriction). |
| Matrix store (phase 3) | public Hugging Face dataset: zstd-compressed binary CSR + manifest with sha256. |
| Build | spack environment `spack/spack.yaml.in`, pinned versions, overlay repo `spack/repo` for versions newer than spack's recipes. The agent does not edit the build: it may write suggestions, which we implement (the recipe converges over time). |
| Versions | hypre 3.2.1, PETSc 3.26.1 (+kokkos), Trilinos 17.3.0 (Tpetra, MueLu, Belos, Ifpack2, Amesos2; +cusparse, +float), Kokkos / Kokkos Kernels 5.2.2. |
| BLAS/LAPACK | netlib-lapack (only CPU-side small dense work). |
| GPUs | 1 GPU / 1 MPI rank in v1; drivers use distributed objects so multi-GPU is a later option. |

## Build rules
- Resolve versions and download sources where there is internet; **build on a GPU node**: configure-time test
  programs that link CUDA need the GPU driver.
- Spack does not expand environment variables in external paths: `sma-tools spack-env` writes them in from the
  loaded modules.
- Drivers keep their RPATH and are configured from scratch on every build (library install paths change).
