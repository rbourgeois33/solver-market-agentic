#!/bin/bash
# Build the AMG libraries with spack on compil_h100 (internet, no GPU, same CPU as the H100 nodes).
# Prerequisite: machines/jean-zay/concretize.slurm succeeded (env rendered on that node).
# Submit from the repo root: sbatch machines/jean-zay/build_libs.slurm
#SBATCH --job-name=sma-build
#SBATCH --partition=compil_h100
#SBATCH -A trt@h100
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/fswork/projects/rech/pri/ujq29su/sma-work/logs/%x-%j.out
source machines/jean-zay/env.sh
spack env activate $SMA_WORK/env
# Concretize again (no-op if nothing changed); stop here on failure instead of installing a stale lock.
spack concretize > $SMA_WORK/env/concretize-build.log 2>&1 || { echo "concretize FAILED"; tail -30 $SMA_WORK/env/concretize-build.log; exit 1; }
time spack install --fail-fast || exit 1
spack find -lv hypre petsc trilinos kokkos kokkos-kernels
