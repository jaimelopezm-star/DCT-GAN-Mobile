"""
Modelos de la arquitectura DCT-GAN
"""

from .encoder import ResNetEncoder, MobileNetV3Encoder
from .decoder import CNNDecoder, LightweightDecoder
from .discriminator import XuNetDiscriminator, EfficientXuNet
from .gan import DCTGAN

__all__ = [
    "ResNetEncoder",
    "MobileNetV3Encoder",
    "CNNDecoder",
    "LightweightDecoder",
    "XuNetDiscriminator",
    "EfficientXuNet",
    "DCTGAN",
]
