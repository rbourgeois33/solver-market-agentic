#!/bin/bash
# Parameter catalogues (sma-tools catalog) from the installed libraries: the drivers initialise CUDA,
# so this runs on an H100 node. Also run at the end of build_libs_gpu.slurm.
# Usage (repo root): sbatch machines/jean-zay/catalog.slurm [--libs petsc,...]
#SBATCH --job-name=sma-catalog
#SBATCH -A trt@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/fswork/projects/rech/pri/ujq29su/sma-work/logs/%x-%j.out
source machines/jean-zay/env.sh
sma-tools catalog --launcher "srun --ntasks=1 --cpus-per-task=24 --gres=gpu:1" "$@"
echo "catalog rc=$?"
