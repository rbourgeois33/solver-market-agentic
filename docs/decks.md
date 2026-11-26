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
| hypre | pcg, gmres, fgmres, bicgstab | `HYPRE_ParCSR<Solver>Set<Name>` suffix: Tol, MaxIter, KDim, TwoNorm... | `HYPRE_BoomerAMGSet<Name>` suffix; `"Name:k"` for per-level-type setters (CycleRelaxType:3) | SpGemmUseVendor, SpMVUseVendor, UseGpuRand |
| muelu | Belos name ("Pseudoblock CG", "Pseudoblock GMRES", ...) | Belos ParameterList keys | MueLu ParameterList keys, **quoted** (`"smoother: type": CHEBYSHEV`), nested dict = sublist, reals as `0.0` | none |
| petsc | KSP type (cg, gmres, fgmres, bcgs...) | PETSc options without dash (ksp_max_it...) | PETSc options (pc_type, pc_gamg_*, mg_levels_*) | mat_type (aijkokkos default, aijcusparse), vec_type |

Stopping: the campaign `--tol` is on the TRUE relative residual ||b-Ax||/||b||, recomputed by the
driver. If the deck gives no internal tolerance, the campaign tolerance is used for the library's
own test; a deck may change it (e.g. a tighter internal tol) to make the true residual pass.
