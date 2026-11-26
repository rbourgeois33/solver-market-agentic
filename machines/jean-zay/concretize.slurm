#!/bin/bash
# Render the spack env ON the compil_h100 node (module paths of that node) and concretize only.
# Submit from the repo root: sbatch machines/jean-zay/concretize.slurm
#SBATCH --job-name=sma-concretize
#SBATCH --partition=compil_h100
#SBATCH -A trt@h100
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=/lustre/fswork/projects/rech/pri/ujq29su/sma-work/logs/%x-%j.out
source machines/jean-zay/env.sh
hostname
env | grep -E "^SMA_" | sort
python -m sma.cli spack-env --machine jean-zay --cuda-arch 90 --force || exit 1
grep -n "prefix\|: /" $SMA_WORK/env/packages.yaml
spack env activate $SMA_WORK/env
rm -f $SMA_WORK/env/spack.lock
spack concretize -f > $SMA_WORK/env/concretize.log 2>&1
rc=$?; echo "concretize rc=$rc"
grep -E "Error|Cannot|Only external" $SMA_WORK/env/concretize.log
grep -E "^ - |^\[e\]" $SMA_WORK/env/concretize.log | grep -oE "(hypre|petsc|trilinos|kokkos|kokkos-kernels|cuda|openmpi|gcc|cmake)@[^~+ ]+.*target=[a-z0-9_]+" | sed -E 's/(@[^~+ ]+).*(target=[a-z0-9_]+)/\1 \2/' | sort -u
grep -E "^ - |^\[e\]" $SMA_WORK/env/concretize.log | grep -c "gcc@13.3.0"
exit $rc
