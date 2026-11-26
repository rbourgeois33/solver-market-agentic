# Overlay: Kokkos Kernels 5.2.2 is the version bundled with Trilinos 17.3.0.
from spack_repo.builtin.packages.kokkos_kernels.package import KokkosKernels as BuiltinKokkosKernels

from spack.package import *


class KokkosKernels(BuiltinKokkosKernels):
    version("5.2.2", sha256="4b95ac27319714a393b9a9ddf3a77fb7ffe751fe215a02084dd781856814d55f")
