# Decks

A deck is one configuration, in YAML, using **each library's own parameter names** (exactly what
the library documents). Only the keys present are set; everything else keeps the library default.
The full reference is the docstring of `sma/deck.py`; examples in `decks/examples/`.

```yaml
lib: hypre            # hypre | muelu | petsc
solver: pcg           # native Krylov name
solver_params: {...}  # native Krylov parameters
precond_params: {...} # native AMG parameters
backend_params: {...} # library-wide GPU settings (hypre, petsc)
```

| lib | solver | solver_params | precond_params | backend_params |
|---|---|---|---|---|
| hypre | pcg, gmres, fgmres, lgmres, cogmres, bicgstab | `HYPRE_<Solver>Set<Name>` suffix: Tol, MaxIter, KDim, TwoNorm, RecomputeResidual... | `HYPRE_BoomerAMGSet<Name>` suffix; `"Name:k"` for indexed setters (CycleRelaxType:3, LevelRelaxWt:2); arrays as lists (`NumGridSweeps: [1, 1, 1, 1]`) | `HYPRE_Set<Name>` suffix: SpGemmUseVendor, SpMVUseVendor, SpMVAlgorithm, SpTransUseVendor, UseGpuRand, LogLevel, NumThreads (the driver restarts after a request that sets one: they are library-global) |
| muelu | Belos name ("Pseudoblock CG", "Pseudoblock GMRES", ...) | Belos ParameterList keys | MueLu ParameterList keys, **quoted** (`"smoother: type": CHEBYSHEV`), nested dict = sublist, reals as `0.0` | none |
| petsc | KSP type (cg, gmres, fgmres, bcgs...) | PETSc options without dash (ksp_max_it...) | PETSc options (pc_type, pc_gamg_*, mg_levels_*) | mat_type (aijkokkos default, aijcusparse), vec_type |

Every accepted name, with type, default and doc text, comes from the installed library itself:
`sma-tools params hypre|muelu|petsc [--grep REGEX] [--section S] [--full]` (built by `sma-tools catalog`
on the GPU node after each build). hypre's tables are generated from its headers at driver build time,
so every setter taking plain values is accepted; the few that are not (per-row data, files,
callbacks) are listed with the reason.

**GPU only**: values that run on the host are refused by `sma-tools run` (not charged), using the rules
and evidence in `sma/gpu_support.yaml` (`sma-tools params LIB --gpu`; `sma-tools gpu-check DECK` to
check without running). Rules apply to effective values, library defaults included (MueLu's default
smoother is Symmetric Gauss-Seidel, which runs on the host). The coarsest-level solve is exempt: a host
direct solve or a GPU smoother are both allowed there.

Stopping: the campaign `--tol` is on the TRUE relative residual ||b-Ax||/||b||, recomputed by the
driver. If the deck gives no internal tolerance, the campaign tolerance is used for the library's
own test; a deck may change it (e.g. a tighter internal tol) to make the true residual pass.
