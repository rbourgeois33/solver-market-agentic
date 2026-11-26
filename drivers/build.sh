#!/bin/bash
# Build the three bench drivers against the spack environment.
# Usage: drivers/build.sh [hypre|petsc|muelu ...]   (default: all). Install dir: $SMA_WORK/bin
set -u
here=$(cd $(dirname $0) && pwd)
spack env activate $SMA_WORK/env || exit 1
libs=${*:-hypre petsc muelu}
rc=0
for lib in $libs; do
  bdir=$SMA_WORK/build/driver-$lib
  extra=""
  case $lib in
    hypre) pfx="$(spack location -i hypre)" ;;
    petsc) pfx="$(spack location -i petsc)"; export PKG_CONFIG_PATH=$pfx/lib/pkgconfig:${PKG_CONFIG_PATH:-} ;;
    muelu) pfx="$(spack location -i trilinos)"
           extra="-DCMAKE_CXX_COMPILER=$(spack location -i kokkos-nvcc-wrapper)/bin/nvcc_wrapper" ;;
  esac
  echo "=== $lib (prefix $pfx)"
  cmake -S $here/$lib -B $bdir -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$pfx" \
        -DCMAKE_INSTALL_PREFIX=$SMA_WORK $extra || { rc=1; continue; }
  cmake --build $bdir -j 16 && cmake --install $bdir || rc=1
done
exit $rc
