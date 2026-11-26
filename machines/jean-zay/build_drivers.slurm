#!/bin/bash
# Build the bench drivers against the spack env, on compil_h100 (same CPU/toolchain as H100 nodes).
# Submit from the repo root: sbatch machines/jean-zay/build_drivers.slurm [hypre petsc muelu]
#SBATCH --job-name=sma-drivers
#SBATCH --partition=compil_h100
#SBATCH -A trt@h100
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fswork/projects/rech/pri/ujq29su/sma-work/logs/%x-%j.out
source machines/jean-zay/env.sh
drivers/build.sh "$@"
echo "build rc=$?"
ls -la $SMA_WORK/bin
