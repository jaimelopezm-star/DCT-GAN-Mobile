"""
DCT-GAN Mobile Steganography Package

Implementación del framework híbrido DCT-GAN con optimización para móviles.
Basado en Malik et al. (2025) con mejoras propuestas.
"""

__version__ = "0.1.0"
__author__ = "Tu Nombre"
__email__ = "tu.email@universidad.edu"

from . import models
from . import dct
from . import training
# from . import evaluation  # TODO: Implementar módulo evaluation
# from . import utils  # utils está fuera de src/

__all__ = [
    "models",
    "dct",
    "training",
    # "evaluation",
    # "utils",
]
