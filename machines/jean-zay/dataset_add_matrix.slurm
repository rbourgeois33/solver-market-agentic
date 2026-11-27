#!/bin/bash
# One dataset matrix: convert + analyze + package (CPU work, not on login). Submitted by
# `sma-dataset add-matrix FILE --name NAME --origin TEXT ...` (arguments passed through).
#SBATCH --partition=compil_h100
#SBATCH -A trt@h100
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fswork/projects/rech/pri/ujq29su/sma-work/logs/%x-%j.out
source machines/jean-zay/env.sh
sma-dataset add-matrix "$@" --here
echo "rc=$?"
