# Overlay: newer upstream release than the builtin recipe knows.
from spack_repo.builtin.packages.hypre.package import Hypre as BuiltinHypre

from spack.package import *


class Hypre(BuiltinHypre):
    version("3.2.1", sha256="3083cf943e1dab54517acae36afad72a4ada2c72c9567e525fc9c6efedb14e0c")
