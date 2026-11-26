# Overlay: Kokkos 5.2.2 is the version bundled with Trilinos 17.3.0.
from spack_repo.builtin.packages.kokkos.package import Kokkos as BuiltinKokkos

from spack.package import *


class Kokkos(BuiltinKokkos):
    version("5.2.2", sha256="d6557aaef39302282a15f9c770433d1fcdf4e961dfd6d9ed726b9d0d0f546b9f")
