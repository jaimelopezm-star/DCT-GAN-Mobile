"""
DenseDecoder - Arquitectura inspirada en SteganoGAN (MIT DAI-Lab)

Diferencias clave vs nuestro LightweightDecoder:
1. Dense connections: concatena features en cada capa
2. Más capacidad para extraer el secreto
3. Arquitectura probada que funciona

Referencia: https://github.com/DAI-Lab/SteganoGAN
Paper: Zhang et al. "SteganoGAN: High Capacity Image Steganography with GANs" (2019)
"""

import torch
import torch.nn as nn


class DenseDecoder(nn.Module):
    """
    Decoder con Dense Connections (estilo SteganoGAN)
    
    Arquitectura:
    - Input: Stego image (3 canales)
    - Conv1: 3 -> hidden
    - Conv2: hidden -> hidden (con dense connection)
    - Conv3: hidden*2 -> hidden (concat con features anteriores)
    - Conv4: hidden*3 -> 3 (output = secret recuperado)
    
    La clave es que cada capa recibe features anteriores concatenadas,
    permitiendo mejor flujo de gradientes y extracción del secreto.
    
    Args:
        data_depth: Profundidad de salida (default: 3 para RGB)
        hidden_size: Canales ocultos (default: 64)
    """
    
    def __init__(self, data_depth=3, hidden_size=64):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        
        # Conv1: procesa imagen stego
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        # Conv2: recibe conv1 output
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        # Conv3: recibe conv1 + conv2 (dense connection)
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        # Conv4: genera output final
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_size * 3, data_depth, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Output en [0, 1]
        )
        
    def forward(self, stego):
        """
        Forward pass con dense connections
        
        Args:
            stego: Imagen stego (B, 3, H, W)
            
        Returns:
            secret: Imagen secreta recuperada (B, 3, H, W) en [0, 1]
        """
        # Dense forward
        x1 = self.conv1(stego)                        # (B, hidden, H, W)
        x2 = self.conv2(x1)                           # (B, hidden, H, W)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))   # (B, hidden, H, W)
        
        # Output: concatena TODAS las features
        secret = self.conv4(torch.cat([x1, x2, x3], dim=1))  # (B, 3, H, W)
        
        return secret
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DenseDecoderLarge(nn.Module):
    """
    Versión más grande del DenseDecoder con más capas y capacidad
    
    Añade capas extra y más canales para mejor extracción
    """
    
    def __init__(self, data_depth=3, hidden_size=64):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        
        # Más capas para mayor capacidad
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_size * 3, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(hidden_size * 4, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        # Output layer
        self.conv_out = nn.Sequential(
            nn.Conv2d(hidden_size * 5, data_depth, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        
    def forward(self, stego):
        x1 = self.conv1(stego)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x1, x2, x3], dim=1))
        x5 = self.conv5(torch.cat([x1, x2, x3, x4], dim=1))
        
        secret = self.conv_out(torch.cat([x1, x2, x3, x4, x5], dim=1))
        
        return secret
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DenseDecoderWithSkip(nn.Module):
    """
    DenseDecoder con Skip Connection desde el input
    
    Añade la imagen stego original al final para mejor reconstrucción
    """
    
    def __init__(self, data_depth=3, hidden_size=64):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        # Output incluye stego original (3 canales extra)
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_size * 3 + 3, data_depth, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        
    def forward(self, stego):
        x1 = self.conv1(stego)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        
        # Incluye stego original en el output (skip connection)
        secret = self.conv4(torch.cat([x1, x2, x3, stego], dim=1))
        
        return secret
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_dense_decoder(decoder_type='dense', hidden_size=64, data_depth=3):
    """
    Factory function para crear decoders densos
    
    Args:
        decoder_type: 'dense', 'dense_large', o 'dense_skip'
        hidden_size: Canales ocultos
        data_depth: Profundidad de datos (3 para RGB)
    """
    if decoder_type == 'dense':
        return DenseDecoder(data_depth=data_depth, hidden_size=hidden_size)
    elif decoder_type == 'dense_large':
        return DenseDecoderLarge(data_depth=data_depth, hidden_size=hidden_size)
    elif decoder_type == 'dense_skip':
        return DenseDecoderWithSkip(data_depth=data_depth, hidden_size=hidden_size)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")


# Test
if __name__ == "__main__":
    stego = torch.randn(2, 3, 256, 256)
    
    # Test DenseDecoder
    decoder = DenseDecoder(data_depth=3, hidden_size=64)
    secret = decoder(stego)
    print(f"DenseDecoder:")
    print(f"  Input stego: {stego.shape}")
    print(f"  Output secret: {secret.shape}")
    print(f"  Parameters: {decoder.get_num_params():,}")
    
    # Test DenseDecoderLarge
    decoder_large = DenseDecoderLarge(data_depth=3, hidden_size=64)
    secret_large = decoder_large(stego)
    print(f"\nDenseDecoderLarge:")
    print(f"  Output secret: {secret_large.shape}")
    print(f"  Parameters: {decoder_large.get_num_params():,}")
    
    # Test DenseDecoderWithSkip
    decoder_skip = DenseDecoderWithSkip(data_depth=3, hidden_size=64)
    secret_skip = decoder_skip(stego)
    print(f"\nDenseDecoderWithSkip:")
    print(f"  Output secret: {secret_skip.shape}")
    print(f"  Parameters: {decoder_skip.get_num_params():,}")
