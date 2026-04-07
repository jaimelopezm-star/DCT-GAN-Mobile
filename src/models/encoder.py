"""
Encoder Architectures para DCT-GAN Steganography

Implementa:
1. ResNet Encoder (Paper Original - Malik et al. 2025)
2. MobileNetV3 Encoder (Propuesta 1 - Mobile-StegoNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Bloque Residual básico simplificado (sin BN para reducir params)
    
    Arquitectura:
        Conv -> ReLU -> Conv -> (+) -> ReLU
        |_______________________|
    """
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        out += identity  # Skip connection
        out = self.relu(out)
        
        return out


class ResNetEncoder(nn.Module):
    """
    Encoder basado en ResNet con 9 bloques residuales
    
    Arquitectura del paper (Tabla paper, Fig. 5):
    - Input: 6 channels (Cover 3 + Secret 3) × 256×256
    - 2 capas convolucionales de downsampling (stride 2)
    - 9 bloques residuales (64×64 feature maps)
    - 2 capas deconvolucionales de upsampling
    - Output: 3 channels (Stego Image) × 256×256
    
    Args:
        input_channels: Número de canales de entrada (default: 6)
        base_channels: Canales base para convoluciones (default: 64)
        num_residual_blocks: Número de bloques residuales (default: 9)
        use_dropout: Usar dropout en bloques residuales (default: True)
        dropout_rate: Tasa de dropout (default: 0.5)
    """
    
    def __init__(
        self,
        input_channels=6,
        base_channels=64,  # Paper Malik et al. 2025: "64×64 feature maps" (Fig. 5)
        num_residual_blocks=9
    ):
        super(ResNetEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.num_residual_blocks = num_residual_blocks
        
        # Input convolution (sin BN para reducir parámetros)
        self.conv_input = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks (9 bloques como en el paper)
        residual_blocks = []
        for _ in range(num_residual_blocks):
            residual_blocks.append(ResidualBlock(base_channels))
        self.residual_layers = nn.Sequential(*residual_blocks)
        
        # Output convolution para generar stego image
        self.conv_output = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1, bias=False)
        self.tanh = nn.Tanh()  # Output en rango [-1, 1]
        
    def forward(self, cover, secret):
        """
        Forward pass del encoder
        
        Args:
            cover: Imagen de cobertura (B, 3, 256, 256)
            secret: Imagen secreta (B, 3, 256, 256)
            
        Returns:
            stego: Imagen esteganográfica (B, 3, 256, 256)
        """
        # Concatenar cover y secret (B, 6, 256, 256)
        x = torch.cat([cover, secret], dim=1)
        
        # Procesamiento sin cambio de resolución
        x = self.conv_input(x)       # (B, 16, 256, 256)
        x = self.relu(x)
        x = self.residual_layers(x)  # (B, 16, 256, 256)
        x = self.conv_output(x)      # (B, 3, 256, 256)
        stego = self.tanh(x)
        
        return stego
    
    def get_num_params(self):
        """Retorna el número total de parámetros del modelo"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MobileNetV3Block(nn.Module):
    """
    Bloque básico de MobileNetV3 con inverted residual
    
    Usa:
    - Depthwise separable convolutions
    - Squeeze-and-Excitation (SE) blocks
    - Hard-Swish activation
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        expand_ratio,
        use_se=True,
        activation='hard_swish'
    ):
        super(MobileNetV3Block, self).__init__()
        
        hidden_dim = int(in_channels * expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels
        self.use_se = use_se
        
        layers = []
        
        # Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish() if activation == 'hard_swish' else nn.ReLU(inplace=True),
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size, stride,
                kernel_size // 2, groups=hidden_dim, bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.Hardswish() if activation == 'hard_swish' else nn.ReLU(inplace=True),
        ])
        
        # Squeeze-and-Excitation
        if use_se:
            layers.append(SEBlock(hidden_dim))
        
        # Projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Hardsigmoid(),
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class MobileNetV3Encoder(nn.Module):
    """
    Encoder basado en MobileNetV3-Small para Mobile-StegoNet
    
    Arquitectura optimizada para dispositivos móviles:
    - Depthwise separable convolutions (menos parámetros)
    - Squeeze-and-Excitation blocks (mejor representación)
    - Hard-Swish activation (eficiente en hardware)
    
    Objetivo: 60% reducción en parámetros vs ResNet (~20K params)
    
    Args:
        input_channels: Número de canales de entrada (default: 6)
        width_multiplier: Multiplicador para ajustar ancho (default: 1.0)
        use_se: Usar Squeeze-and-Excitation (default: True)
    """
    
    def __init__(
        self,
        input_channels=6,
        width_multiplier=1.0,
        use_se=True
    ):
        super(MobileNetV3Encoder, self).__init__()
        
        self.input_channels = input_channels
        
        # Configuración de bloques MobileNetV3-Small
        # [in_ch, out_ch, kernel, stride, expand_ratio, use_se]
        configs = [
            [16, 16, 3, 2, 1, True],   # 128x128
            [16, 24, 3, 2, 4.5, False], # 64x64
            [24, 24, 3, 1, 3.67, False],
            [24, 40, 5, 2, 4, True],    # 32x32
            [40, 40, 5, 1, 6, True],
            [40, 40, 5, 1, 6, True],
            [40, 48, 5, 1, 3, True],
            [48, 48, 5, 1, 3, True],
            [48, 96, 5, 2, 6, True],    # 16x16
            [96, 96, 5, 1, 6, True],
            [96, 96, 5, 1, 6, True],
        ]
        
        # Ajustar canales con width_multiplier
        def adjust_channels(channels):
            return int(channels * width_multiplier)
        
        # Primera capa convolucional
        first_ch = adjust_channels(16)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(input_channels, first_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(first_ch),
            nn.Hardswish(),
        )
        
        # Bloques MobileNetV3
        layers = []
        for in_ch, out_ch, k, s, exp, se in configs:
            in_ch = adjust_channels(in_ch)
            out_ch = adjust_channels(out_ch)
            layers.append(
                MobileNetV3Block(in_ch, out_ch, k, s, exp, se and use_se)
            )
        self.blocks = nn.Sequential(*layers)
        
        # Upsampling para recuperar tamaño original
        last_ch = adjust_channels(96)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(last_ch, 64, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(64),
            nn.Hardswish(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),       # 64x64
            nn.BatchNorm2d(32),
            nn.Hardswish(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),       # 128x128
            nn.BatchNorm2d(16),
            nn.Hardswish(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),        # 256x256
            nn.BatchNorm2d(8),
            nn.Hardswish(),
        )
        
        # Capa final
        self.conv_final = nn.Conv2d(8, 3, 3, 1, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, cover, secret):
        """
        Forward pass del encoder móvil
        
        Args:
            cover: Imagen de cobertura (B, 3, 256, 256)
            secret: Imagen secreta (B, 3, 256, 256)
            
        Returns:
            stego: Imagen esteganográfica (B, 3, 256, 256)
        """
        x = torch.cat([cover, secret], dim=1)
        
        x = self.conv_stem(x)
        x = self.blocks(x)
        x = self.upsample(x)
        stego = self.tanh(self.conv_final(x))
        
        return stego
    
    def get_num_params(self):
        """Retorna el número total de parámetros"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Función de utilidad para crear encoders
def create_encoder(config):
    """
    Factory function para crear encoders
    
    Args:
        config: Diccionario con configuración del encoder
        
    Returns:
        Instancia del encoder correspondiente
    """
    encoder_type = config.get('type', 'resnet')
    
    if encoder_type == 'resnet':
        return ResNetEncoder(
            input_channels=config.get('input_channels', 6),
            base_channels=config.get('base_channels', 16),
            num_residual_blocks=config.get('num_residual_blocks', 9)
        )
    elif encoder_type in ['mobilenetv3', 'mobilenetv3_small']:
        return MobileNetV3Encoder(
            input_channels=config.get('input_channels', 6),
            width_multiplier=config.get('width_multiplier', 1.0),
            use_se=config.get('use_se', True)
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


if __name__ == "__main__":
    # Test encoders
    print("="*60)
    print("Testing ResNet Encoder")
    print("="*60)
    
    encoder_resnet = ResNetEncoder()
    cover = torch.randn(4, 3, 256, 256)
    secret = torch.randn(4, 3, 256, 256)
    
    stego = encoder_resnet(cover, secret)
    print(f"Input shapes: Cover {cover.shape}, Secret {secret.shape}")
    print(f"Output shape: {stego.shape}")
    print(f"Parameters: {encoder_resnet.get_num_params():,}")
    
    print("\n" + "="*60)
    print("Testing MobileNetV3 Encoder")
    print("="*60)
    
    encoder_mobile = MobileNetV3Encoder()
    stego_mobile = encoder_mobile(cover, secret)
    print(f"Output shape: {stego_mobile.shape}")
    print(f"Parameters: {encoder_mobile.get_num_params():,}")
    
    reduction = (1 - encoder_mobile.get_num_params() / encoder_resnet.get_num_params()) * 100
    print(f"\nParameter reduction: {reduction:.1f}%")
