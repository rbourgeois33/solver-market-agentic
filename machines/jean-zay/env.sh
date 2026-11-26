# Source me: Jean Zay H100 environment for solver-market-agentic.
# Toolchain: gcc 13.3 + CUDA 12.8 + CUDA-aware OpenMPI 4.1.6.
module purge
module load arch/h100 gcc/13.3.0 cuda/12.8.0 openmpi/4.1.6-cuda cmake/3.31.4 python/3.11.5 2>/dev/null
export SMA_MACHINE=jean-zay
export PATH=$(cd $(dirname ${BASH_SOURCE[0]})/../.. && pwd)/bin:$PATH   # sma-tools
# Module prefixes differ between login and H100 nodes: always derive them from the loaded modules.
export SMA_GCC_PREFIX=$(dirname $(dirname $(which gcc)))
export SMA_CUDA_PREFIX=$CUDA_HOME
export SMA_MPI_PREFIX=$(dirname $(dirname $(which mpicc)))
export SMA_CMAKE_PREFIX=$(dirname $(dirname $(which cmake)))
export SMA_WORK=${SMA_WORK:-$WORK/sma-work}
# Build stage on the scratch filesystem (many small files, purged automatically).
export SMA_STAGE_DIR=${SMA_STAGE_DIR:-$SCRATCH/sma-stage}
# HOME is small on Jean Zay: keep every spack user file under SMA_WORK.
export SPACK_USER_CONFIG_PATH=$SMA_WORK/spack-user
export SPACK_USER_CACHE_PATH=$SMA_WORK/spack-user/cache
export SPACK_DISABLE_LOCAL_CONFIG=1
# The arch/h100 python/3.11.5 module lacks os.O_PATH, which spack 1.2 needs: use the system python.
export SPACK_PYTHON=/usr/bin/python3
[ -f $SMA_WORK/spack/share/spack/setup-env.sh ] && . $SMA_WORK/spack/share/spack/setup-env.sh
