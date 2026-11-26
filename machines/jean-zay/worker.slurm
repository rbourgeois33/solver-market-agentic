#!/bin/bash
# Bench worker on an H100 node. `sma-tools worker start` adds --exclusive by default (benchmarks:
# whole node, no neighbours); --no-exclusive for tests.
# Submitted by `sma-tools worker start --mode slurm`, which passes --time and the dispatcher command.
# QoS: qos_gpu_h100-dev allows up to 2 h; longer campaigns need --qos=qos_gpu_h100-t3 (20 h).
#SBATCH --job-name=sma-worker
#SBATCH -A trt@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
source machines/jean-zay/env.sh
export OMP_NUM_THREADS=24   # host-side true residual check
nvidia-smi -q -d CLOCK,PERFORMANCE,COMPUTE | head -60
# The dispatcher (python) runs in the batch step; it starts each driver as its own 1-rank
# `srun` step (mpirun is a stub on Jean Zay). Only one driver step exists at a time.
exec "$@"
