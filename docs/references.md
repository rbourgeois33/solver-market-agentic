# References

Where the parameters and their GPU support are documented. In order of trust: the **installed version**
first (catalogue and sources below are exactly what runs), then the official online docs (which may
describe another version).

## Installed version (offline, on every machine after `sma-tools build`)
| what | where |
|---|---|
| every accepted parameter, type, default, doc text | `sma-tools params hypre\|muelu\|petsc [--grep R] [--full]` |
| GPU support of values (doc/source evidence) and the do-not-try list | `sma-tools params LIB --gpu` (`sma/gpu_support.yaml`); `sma-tools gpu-check DECK` |
| sources + user guides of the built versions | `$SMA_WORK/share/sma/src/{hypre-3.2.1,petsc-3.26.1,trilinos-17.3.0}/` |
| MueLu: all parameters with descriptions | `trilinos-17.3.0/packages/muelu/doc/UsersGuide/masterList.xml`, `options_*.tex` |
| Ifpack2 (MueLu smoothers): options | `trilinos-17.3.0/packages/ifpack2/doc/UsersGuide/options.tex`; `Ifpack2_Relaxation_def.hpp`, `Ifpack2_Chebyshev_def.hpp`, ... in the install `include/` |
| Belos solver parameters | `sma-tools params muelu --section belos:<SOLVER NAME>` |
| hypre: every setter with its doc comment | `hypre-3.2.1/src/parcsr_ls/HYPRE_parcsr_ls.h`, `src/krylov/HYPRE_krylov.h`, `src/utilities/HYPRE_utilities.h` |
| PETSc: options of each object | `sma-tools params petsc`; manual pages = comments in `petsc-3.26.1/src/ksp/pc/impls/*/`, `src/ksp/ksp/impls/*/` |

## Official online documentation (checked reachable 2026-11-26)
| library | page |
|---|---|
| hypre | BoomerAMG, incl. "GPU-supported Options": https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html |
| hypre | ParCSR solver API: https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html |
| hypre | sources: https://github.com/hypre-space/hypre |
| PETSc | KSP/PC manual chapter: https://petsc.org/release/manual/ksp/ |
| PETSc | PCGAMG: https://petsc.org/release/manualpages/PC/PCGAMG/ ; Chebyshev: https://petsc.org/release/manualpages/KSP/KSPCHEBYSHEV/ (manual pages: https://petsc.org/release/manualpages/) |
| PETSc | GPU status: https://petsc.org/release/overview/gpu_roadmap/ ; solver table: https://petsc.org/release/overview/linear_solve_table/ |
| PETSc | sources: https://gitlab.com/petsc/petsc |
| MueLu | documentation index: https://trilinos.github.io/muelu_documentation.html ; user's guide: https://trilinos.github.io/pdfs/mueluguide.pdf |
| Trilinos | doxygen https://docs.trilinos.org (not reachable from Jean Zay login nodes: use the installed sources) |
