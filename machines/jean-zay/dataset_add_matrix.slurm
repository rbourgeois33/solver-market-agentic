#!/bin/bash
# Package a converted system into the dataset (gzip of the MatrixMarket source: CPU work, not on login).
# Usage (repo root): sbatch machines/jean-zay/dataset_add_matrix.slurm SYSTEM NAME "ORIGIN" [DATASET_DIR]
#SBATCH --job-name=sma-dataset
#SBATCH --partition=compil_h100
#SBATCH -A trt@h100
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/pri/ujq29su/sma-work/logs/%x-%j.out
source machines/jean-zay/env.sh
sma-tools dataset add-matrix "$1" --name "$2" --origin "$3" ${4:+--to "$4"}
echo "rc=$?"
