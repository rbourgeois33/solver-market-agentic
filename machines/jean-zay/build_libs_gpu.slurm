#!/bin/bash
# Build the AMG libraries ON an H100 node: configure-time test programs that link CUDA need the
# GPU driver (libcuda), absent on compil_h100 (PETSc wrongly assumed 64-bit BLAS indices there).
# GPU nodes are offline: sources must already be in the spack source cache (fetch on login/compil).
# Usage (repo root): sbatch machines/jean-zay/build_libs_gpu.slurm [--rebuild]
#   --rebuild: uninstall the GPU stack first (kokkos/kk, umpire/camp, hypre, petsc, trilinos)
#SBATCH --job-name=sma-build-gpu
#SBATCH -A trt@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/pri/ujq29su/sma-work/logs/%x-%j.out
source machines/jean-zay/env.sh
hostname; nvidia-smi -L
spack env activate $SMA_WORK/env
spack concretize > $SMA_WORK/env/concretize-build.log 2>&1 || { echo "concretize FAILED"; tail -30 $SMA_WORK/env/concretize-build.log; exit 1; }
if [ "$1" = "--rebuild" ]; then
  spack uninstall -y -a --dependents kokkos kokkos-kernels umpire camp hypre petsc trilinos || exit 1
fi
time spack install --fail-fast || exit 1
spack find -lv hypre petsc trilinos
grep -E "64BIT_BLAS" $(spack location -i petsc)/include/petscconf.h
drivers/build.sh
echo "build rc=$?"
