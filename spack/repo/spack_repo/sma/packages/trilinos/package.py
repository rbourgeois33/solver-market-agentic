# Overlay: Trilinos 17.3.0, which requires Kokkos / Kokkos Kernels 5.2.2.
from spack_repo.builtin.packages.trilinos.package import Trilinos as BuiltinTrilinos

from spack.package import *


class Trilinos(BuiltinTrilinos):
    version("17.3.0", sha256="8094740da60bf75b675ad556ed00e337c8547be0dc9603893c4894f4abf0396b")

    with when("@17.3 +kokkos"):
        depends_on("kokkos@=5.2.2")
        depends_on("kokkos-kernels@=5.2.2")
