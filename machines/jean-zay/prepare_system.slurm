#!/bin/bash
# Convert a matrix (+rhs) to the binary format and analyze it, on an H100 node (not on login).
# sbatch machines/jean-zay/prepare_system.slurm MATRIX RHS OUTDIR
#SBATCH --job-name=sma-prepare
#SBATCH -A trt@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fswork/projects/rech/pri/ujq29su/sma-work/logs/%x-%j.out
source machines/jean-zay/env.sh
hostname
python -m sma.cli convert --matrix "$1" --rhs "$2" --out "$3" || exit 1
python -m sma.cli analyze "$3/A.csr"
