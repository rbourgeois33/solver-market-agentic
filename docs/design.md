# solver-market-agentic: design (agreed 2026-11-26)

## Context
We want a Python app where the user passes a matrix, an rhs, an agent and some campaign constraints. The app launches the agent (Claude Code or a local CLI agent). The agent builds GPU AMG libraries on the target hardware. It then tunes Krylov (CG/GMRES) + AMG: MueLu, BoomerAMG or GAMG, on GPU only. It works under a budget of N configurations and logs its reasoning explicitly. A benchmark DB of TRUST systems with expert-tuned decks measures whether the agent beats the experts.
- Tooling repo: `git@github.com:rbourgeois33/solver-market-agentic.git` (empty, cloned read-only so far).
- Knowledge repo: `git@github.com:rbourgeois33/skills.git`. Add `solver-market-agentic.md` there and link it from README.

Facts found:
- TRUST `Solv_Petsc` can dump systems: `save_matrix_petsc_format`, `save_matrix_mtx_format` (`trust-code/src/Kernel/Math/SolvSys/Solv_Petsc.cpp:755-776`, `1790-1838`).
- First test matrix: `$WORK/aij_2592000.mtx` (475 MB text, MatrixMarket "general"). Use rhs = ones.
- Jean Zay: spack is not installed. GPU nodes have no internet; login and compil_h100 do.

## Decisions taken with the user
| Topic | Decision |
|---|---|
| Where things run | **Cluster:** orchestrator + agent on the login node. A long sbatch job on an exclusive H100 node runs a persistent bench worker, fed through a Lustre queue directory. **Private machine:** everything local, same worker run as a subprocess. |
| Drivers | 3 persistent C++ servers: MueLu+Belos (Tpetra/Kokkos), hypre BoomerAMG+PCG/GMRES, PETSc KSP+GAMG. Each loads the matrix to the GPU once, then serves many configs. Each server uses its own library's Krylov. |
| Binary format | Own raw CSR: header (magic, version, nrows, ncols, nnz, idx width, value type, symmetric flag) + rowptr/colind/values. Converters from .mtx and PETSc binary. |
| Decks | YAML with native parameter names (MueLu ParameterList keys, `HYPRE_BoomerAMGSet*` names, PETSc options). Only keys present are set; everything else keeps the library default. Expert decks use the same format. |
| Objective | k = solves per setup. cost = (t_setup + k·t_solve)/k; k=0 (default) means t_solve only. Each time is the median of R repeats after a warm-up. |
| Convergence | User gives a relative tol on the **true** residual ‖b−Ax‖/‖b‖, which the driver recomputes itself. Configs that hit maxit, NaN or OOM are rejected. |
| Memory limit | Driver records peak GPU memory (cudaMemGetInfo around setup/solve, minus baseline) and also the AMG operator/grid complexity. `--mem-limit` default = GPU capacity. |
| Analysis tool | Python numpy/scipy on the binary CSR (memmap): structural and numerical symmetry, diagonal sign/dominance, zero diagonal entries, nnz/row stats, block-size detection, bandwidth, optional Lanczos extreme eigenvalues. |
| Build | Repo ships a spack env (`spack.yaml` + lock). Pinned latest stable hypre/PETSc/Trilinos, +cuda, cuda_arch taken from the detected GPU, cuda/mpi/gcc as externals. The agent clones spack and fetches sources on the login node, then builds on the GPU node (or compil_h100 with an explicit arch). Versions are recorded in every result. |
| GPUs | v1 runs on 1 GPU with 1 MPI rank. Drivers use distributed objects (Tpetra, ParCSR, MPIAIJ) so multi-GPU is a later flag. |
| Agent launch | CLI adapters: `--agent claude` runs `claude -p <fixed prompt>` in a campaign workdir with a restricted settings.json. `--agent cmd:'<template>'` runs any other CLI agent (local LLM). The toolbox is plain shell commands. |
| Budget + log | `sma-tools run` is the only way to benchmark. It requires `--why` (the hypothesis) and refuses once the budget is spent. Everything is appended to `journal.jsonl`. The agent keeps `reasoning.md`, and the orchestrator saves the transcript. |
| Anti-cheat | Expert DB = `bench/` subdir of the same repo (public info anyway). We trust the agent. The prompt says not to read `bench/`, and the Claude settings deny rule blocks that path (cheap). |
| Matrix store | Public Hugging Face dataset: free best-effort public storage, no per-file 100 MB limit, versioned. Files are zstd-compressed binary CSR + manifest with sha256. Download happens on the login node. |
| First matrix | aij_2592000 directly, with rhs = ones. |

## Clarifications after review
- **Decks:** keys are exactly the parameters each library exposes in its docs/API. Nothing is invented or mapped. Only the keys present are set. Example:
  ```yaml
  lib: hypre
  krylov: {type: pcg, tol: 1e-8, maxit: 200}
  amg: {CoarsenType: 8, InterpType: 6, StrongThreshold: 0.5, RelaxType: 18, AggNumLevels: 1}
  # -> HYPRE_BoomerAMGSetCoarsenType(8), ...
  ```
  MueLu uses its ParameterList keys (`"aggregation: drop tol"`, `"smoother: type"`, ...). GAMG uses PETSc options (`-pc_gamg_threshold`, `-mg_levels_ksp_type`, ...).
- **Objective = runtime** (cost formula above). Memory is only a constraint.
- **Build freedom:** the agent may edit the spack env (variants, versions, patches) in its campaign copy. It must log each tweak with its reason in `build_log.md`, and that file is promoted to the repo/skills for later sessions.
- **No orphan GPU job:** the orchestrator owns the worker job. A try/finally + SIGINT/SIGTERM handler runs `scancel` whenever the agent process exits: normal end, error, token/rate limit, or Ctrl-C. Safety nets in case the orchestrator itself dies: the worker exits after an idle timeout (no request for X min, configurable) and has a capped sbatch walltime.

## Proposed details I still need you to confirm (shown here so you can veto)
1. **Deck → driver translation in Python.** `sma-tools run` turns the YAML into each library's native input: a ParameterList XML for MueLu, an options string for PETSc, `key value` lines for hypre (a mapping table in the driver calls `HYPRE_BoomerAMGSet*`). The C++ drivers then need no YAML/JSON library and write their result JSON with printf.
2. **Worker:** one dispatcher per sbatch job. It starts the requested library's server on demand and keeps it alive. Requests are serialized on GPU 0 so no timings run concurrently.
3. **Python env:** Jean Zay `python/3.11.5` module (numpy, scipy, pyyaml) plus a venv for the extras (zstandard, huggingface_hub).

## Repo layout (solver-market-agentic)
```
pyproject.toml            # package `sma`, entry points: sma, sma-tools
sma/
  csr_io.py               # binary CSR read/write (numpy memmap)
  convert.py              # mtx / petsc-binary -> .csr (+ rhs), streaming for large files
  analyze.py              # matrix properties -> analysis.json + human summary
  deck.py                 # YAML deck schema check + per-lib translation
  run.py                  # sma-tools run: budget, --why, journal, submit to worker, wait for result
  worker/                 # dispatcher (queue dir protocol), slurm job template, local mode
  campaign.py             # (phase 2) sma campaign: workdir, prompt, agent adapters
  score.py                # (phase 3) compare against bench/ expert results
drivers/
  common/                 # csr reader, timers, peak-mem sampler, true residual, result JSON
  muelu/  hypre/  petsc/  # one server executable each, CMake
spack/spack.yaml          # env + lock
decks/examples/{muelu,hypre,gamg}/*.yaml   # defaults + a few typical GPU configs
prompts/agent.md          # (phase 2) fixed instructions + tool docs
bench/                    # (phase 3) cases/<id>/case.yaml, expert/*.yaml, results/<hw>/*.json
docs/                     # design.md (this plan), deck reference, protocol
```

## Phase 1 (this execution): toolbox, tested by hand on H100
1. Skeleton + `pyproject`, README, design doc. Commit on a branch.
2. `csr_io` + `convert` (streaming .mtx parser, PETSc binary reader). Convert aij_2592000 and time the reads (text vs binary).
3. `analyze` on aij_2592000.
4. `spack/spack.yaml`, plus Jean Zay scripts: fetch on login, build via sbatch.
5. Drivers: common layer + 3 servers. Each reads .csr, builds the matrix on GPU, and loops: read request → setup → solve ×R → true residual → write result JSON (t_setup, t_solve, iters, true relres, converged, peak mem, complexities, lib versions, GPU name).
6. Example decks per library, and Python translation.
7. `sma-tools run` + journal + budget, plus the worker (Slurm + local).
8. Write `skills/solver-market-agentic.md` (and README link) on the skills repo: purpose, layout, decisions, Jean Zay recipes learned. Update it at the end of each session (general.md rule).

Phase 2: `sma campaign` (prompt, adapters, Claude settings, transcript capture, final report).
Phase 3: HF dataset uploader/downloader, `bench/` schema + expert decks (from TRUST PETSc settings), `sma score`.

## Verification (phase 1, together on H100)
- Round-trip test: mtx → csr → compare with scipy; also time the reads.
- Analysis report on aij_2592000 makes sense (TRUST pressure matrix: SPD expected?).
- Spack build succeeds on Jean Zay, versions recorded.
- Each driver solves aij_2592000 with rhs=1 to tol 1e-8 (true residual verified), with a CG+AMG default deck. Results JSON appear in the journal.
- Budget refusal works, and a server serves several configs without reloading (load time logged once).
- Every sbatch run uses `-A trt@h100 -C h100 --exclusive` (benchmark), builds use compil_h100, and we never circumvent Slurm.

## Implementation notes (phase 1, 2026-11-26)
- Spack 1.2.2. Overlay repo `spack/repo` adds hypre 3.2.1, petsc 3.26.1, trilinos 17.3.0,
  kokkos/kokkos-kernels 5.2.2 (bundled versions of Trilinos 17.3). PETSc recipe copied: its
  `kokkos+cuda_lambda` requirement no longer exists in Kokkos 5.
- Trilinos 17 removed Epetra/AztecOO/ML/Ifpack/Amesos: never name those variants (they pin @:16).
  Tpetra stack only. `+cusparse`: Kokkos Kernels uses cuSPARSE for SpMV & co.
- Jean Zay: env rendered and libraries built ON compil_h100 (module paths and CPU target differ
  from the login node). Spack does not expand env variables in external paths: they are written
  in by `sma-tools spack-env`. `SPACK_PYTHON=/usr/bin/python3` (python/3.11.5 module lacks os.O_PATH).
- Worker: python dispatcher in the batch step; each driver is its own `srun --overlap` 1-rank step.
- PETSc driver caches one Mat per `mat_type` (default aijkokkos) and reports unused options;
  MueLu driver reports unused ParameterList entries; hypre driver rejects unknown names.
- True residual computed on the host from the original CSR, independent of the libraries.
