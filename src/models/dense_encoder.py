"""
DenseEncoder - Arquitectura inspirada en SteganoGAN (MIT DAI-Lab)

Diferencias clave vs nuestro ResNetEncoder:
1. Dense connections: concatena features en cada capa (no solo residual)
2. No usa residual_scale - suma directa image + output
3. Arquitectura probada que funciona (~37 dB PSNR)

Referencia: https://github.com/DAI-Lab/SteganoGAN
Paper: Zhang et al. "SteganoGAN: High Capacity Image Steganography with GANs" (2019)
"""

import torch
import torch.nn as nn


class DenseEncoder(nn.Module):
    """
    Encoder con Dense Connections (estilo SteganoGAN)
    
    Arquitectura:
    - Input: Cover (3ch) + Secret (3ch) = 6 canales
    - Conv1: 6 -> hidden
    - Conv2: hidden + 6 -> hidden (concat con input)
    - Conv3: hidden*2 + 6 -> hidden (concat con features anteriores)
    - Conv4: hidden*3 + 6 -> 3 (output)
    - Final: image + output (residual connection)
    
    La clave es que cada capa recibe TODAS las features anteriores
    concatenadas, permitiendo mejor flujo de gradientes.
    
    Args:
        data_depth: Profundidad de datos (default: 3 para RGB secret)
        hidden_size: Canales ocultos (default: 64)
    """
    
    def __init__(self, data_depth=3, hidden_size=64):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        
        # Input: cover(3) + secret(3) = 6 canales
        input_channels = 3 + data_depth
        
        # Conv1: procesa input inicial
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        # Conv2: recibe conv1 output + input original
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size + input_channels, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        # Conv3: recibe conv1 + conv2 + input original
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size * 2 + input_channels, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        # Conv4: genera residual (3 canales)
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_size * 3 + input_channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),  # Output en [-1, 1]
        )
        
    def forward(self, cover, secret):
        """
        Forward pass con dense connections
        
        Args:
            cover: Imagen cover (B, 3, H, W) en [-1, 1]
            secret: Imagen secreta (B, 3, H, W) en [-1, 1]
            
        Returns:
            stego: Imagen stego (B, 3, H, W) en [-1, 1]
        """
        # Concatenar cover y secret
        x = torch.cat([cover, secret], dim=1)  # (B, 6, H, W)
        
        # Dense forward
        x1 = self.conv1(x)                           # (B, hidden, H, W)
        x2 = self.conv2(torch.cat([x1, x], dim=1))   # (B, hidden, H, W)
        x3 = self.conv3(torch.cat([x1, x2, x], dim=1))  # (B, hidden, H, W)
        
        # Output: concatena TODAS las features
        output = self.conv4(torch.cat([x1, x2, x3, x], dim=1))  # (B, 3, H, W)
        
        # Residual connection: stego = cover + residual
        # SteganoGAN suma directamente (sin escala)
        stego = cover + output
        
        # Clamp a rango válido
        stego = torch.clamp(stego, -1.0, 1.0)
        
        return stego
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DenseEncoderLarge(nn.Module):
    """
    Versión más grande del DenseEncoder con más capas
    
    Añade una capa extra para mayor capacidad
    """
    
    def __init__(self, data_depth=3, hidden_size=64):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        
        input_channels = 3 + data_depth
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size + input_channels, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size * 2 + input_channels, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_size * 3 + input_channels, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        # Output layer
        self.conv5 = nn.Sequential(
            nn.Conv2d(hidden_size * 4 + input_channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, cover, secret):
        x = torch.cat([cover, secret], dim=1)
        
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat([x1, x], dim=1))
        x3 = self.conv3(torch.cat([x1, x2, x], dim=1))
        x4 = self.conv4(torch.cat([x1, x2, x3, x], dim=1))
        
        output = self.conv5(torch.cat([x1, x2, x3, x4, x], dim=1))
        
        stego = cover + output
        stego = torch.clamp(stego, -1.0, 1.0)
        
        return stego
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_dense_encoder(encoder_type='dense', hidden_size=64, data_depth=3):
    """
    Factory function para crear encoders densos
    
    Args:
        encoder_type: 'dense' o 'dense_large'
        hidden_size: Canales ocultos
        data_depth: Profundidad de datos (3 para RGB)
    """
    if encoder_type == 'dense':
        return DenseEncoder(data_depth=data_depth, hidden_size=hidden_size)
    elif encoder_type == 'dense_large':
        return DenseEncoderLarge(data_depth=data_depth, hidden_size=hidden_size)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# Test
if __name__ == "__main__":
    # Test DenseEncoder
    encoder = DenseEncoder(data_depth=3, hidden_size=64)
    cover = torch.randn(2, 3, 256, 256)
    secret = torch.randn(2, 3, 256, 256)
    
    stego = encoder(cover, secret)
    print(f"DenseEncoder:")
    print(f"  Input cover: {cover.shape}")
    print(f"  Input secret: {secret.shape}")
    print(f"  Output stego: {stego.shape}")
    print(f"  Parameters: {encoder.get_num_params():,}")
    
    # Test DenseEncoderLarge
    encoder_large = DenseEncoderLarge(data_depth=3, hidden_size=64)
    stego_large = encoder_large(cover, secret)
    print(f"\nDenseEncoderLarge:")
    print(f"  Output stego: {stego_large.shape}")
    print(f"  Parameters: {encoder_large.get_num_params():,}")
