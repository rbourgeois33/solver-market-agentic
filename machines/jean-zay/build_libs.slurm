#!/bin/bash
# First build on compil_h100 (has internet): fills the spack source cache for the offline GPU nodes.
# The libraries used for benchmarks are then (re)built on a GPU node: build_libs_gpu.slurm --rebuild.
# Prerequisite: machines/jean-zay/concretize.slurm succeeded.
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
