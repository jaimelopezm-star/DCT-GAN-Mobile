"""
Decoder Architectures para DCT-GAN Steganography

Implementa:
1. CNN Decoder (Paper Original - 6 capas convolucionales)
2. Lightweight Decoder (Propuesta 1 - optimizado para móviles)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNDecoder(nn.Module):
    """
    Decoder CNN simplificado para ~15K parámetros
    
    Arquitectura del paper (Fig. 4, simplificada sin BN):
    - Input: Stego Image (B, 3, 256, 256)
    - 6 capas convolucionales (3×3, stride 1, padding 1)
    - ReLU después de cada capa (excepto la última)
    - Output: Recovered Secret Image (B, 3, 256, 256)
    - Función de activación final: Sigmoid (rango [0, 1])
    
    Args:
        base_channels: Canales base (default: 16, reducido de 64)
        num_layers: Número de capas convolucionales (default: 6)
        output_channels: Canales de salida (default: 3 para RGB)
    """
    
    def __init__(
        self,
        base_channels=16,  # Reducido de 64 a 16
        num_layers=6,
        output_channels=3
    ):
        super(CNNDecoder, self).__init__()
        
        self.base_channels = base_channels
        self.num_layers = num_layers
        
        layers = []
        
        # Primera capa: 3 -> base_channels (sin BN)
        layers.extend([
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        ])
        
        # Capas intermedias: base_channels -> base_channels (sin BN)
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ])
        
        # Última capa: base_channels -> output_channels
        layers.extend([
            nn.Conv2d(base_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output en rango [0, 1]
        ])
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, stego):
        """
        Forward pass del decoder
        
        Args:
            stego: Imagen esteganográfica (B, 3, 256, 256)
            
        Returns:
            secret_recovered: Imagen secreta recuperada (B, 3, 256, 256)
        """
        secret_recovered = self.decoder(stego)
        return secret_recovered
    
    def get_num_params(self):
        """Retorna el número total de parámetros"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DepthwiseSeparableConv(nn.Module):
    """
    Convolución depthwise separable para reducir parámetros
    
    Divide conv estándar en:
    1. Depthwise: cada canal se convoluciona independientemente
    2. Pointwise: mezcla canales con conv 1×1
    
    Reduce parámetros de k²×in×out a k²×in + in×out
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LightweightDecoder(nn.Module):
    """
    Decoder ligero optimizado para dispositivos móviles
    
    Mejoras sobre CNN Decoder:
    - Depthwise separable convolutions (menos parámetros)
    - Menos canales base (32 vs 64)
    - 5 capas en vez de 6
    - Opcional: Skip connections para mejor gradiente
    
    Objetivo: ~50-60% reducción en parámetros
    
    Args:
        base_channels: Canales base (default: 32)
        num_layers: Número de capas (default: 5)
        output_channels: Canales de salida (default: 3)
        use_depthwise: Usar depthwise separable conv (default: True)
        use_skip: Usar skip connections (default: False)
    """
    
    def __init__(
        self,
        base_channels=32,
        num_layers=5,
        output_channels=3,
        use_depthwise=True,
        use_skip=False
    ):
        super(LightweightDecoder, self).__init__()
        
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.use_skip = use_skip
        
        ConvLayer = DepthwiseSeparableConv if use_depthwise else nn.Conv2d
        
        # Primera capa
        if use_depthwise:
            self.conv_first = nn.Sequential(
                DepthwiseSeparableConv(3, base_channels),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv_first = nn.Sequential(
                nn.Conv2d(3, base_channels, 3, 1, 1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            )
        
        # Capas intermedias
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers - 2):
            layer = nn.Sequential(
                DepthwiseSeparableConv(base_channels, base_channels) if use_depthwise
                else nn.Conv2d(base_channels, base_channels, 3, 1, 1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            )
            self.conv_layers.append(layer)
        
        # Última capa
        if use_depthwise:
            self.conv_final = nn.Sequential(
                DepthwiseSeparableConv(base_channels, output_channels),
                nn.Sigmoid()
            )
        else:
            self.conv_final = nn.Sequential(
                nn.Conv2d(base_channels, output_channels, 3, 1, 1),
                nn.Sigmoid()
            )
        
        # Skip connections (si se usan)
        if self.use_skip:
            self.skip_conv = nn.Conv2d(3, base_channels, 1, 1, 0)
        
    def forward(self, stego):
        """
        Forward pass del decoder ligero
        
        Args:
            stego: Imagen esteganográfica (B, 3, 256, 256)
            
        Returns:
            secret_recovered: Imagen secreta recuperada (B, 3, 256, 256)
        """
        identity = stego
        
        x = self.conv_first(stego)
        
        # Skip connection opcional
        if self.use_skip:
            skip = self.skip_conv(identity)
        
        # Capas intermedias
        for layer in self.conv_layers:
            x = layer(x)
        
        # Añadir skip si está habilitado
        if self.use_skip:
            x = x + skip
        
        secret_recovered = self.conv_final(x)
        return secret_recovered
    
    def get_num_params(self):
        """Retorna el número total de parámetros"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionBlock(nn.Module):
    """
    Bloque de atención simple para mejorar recuperación
    Opcional para versiones futuras
    """
    
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        att = self.attention(x)
        return x * att


class EnhancedLightweightDecoder(nn.Module):
    """
    Versión mejorada del Lightweight Decoder con atención
    
    Para experimentos futuros si se necesita mejor calidad
    sin sacrificar demasiados parámetros.
    """
    
    def __init__(
        self,
        base_channels=32,
        num_layers=5,
        output_channels=3,
        use_attention=True
    ):
        super(EnhancedLightweightDecoder, self).__init__()
        
        self.use_attention = use_attention
        
        # Arquitectura similar a LightweightDecoder
        self.conv_first = nn.Sequential(
            DepthwiseSeparableConv(3, base_channels),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Capas con atención intermedia
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers - 2):
            layer = nn.Sequential(
                DepthwiseSeparableConv(base_channels, base_channels),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            )
            self.conv_layers.append(layer)
            
            # Añadir atención cada 2 capas
            if use_attention and (i + 1) % 2 == 0:
                self.conv_layers.append(AttentionBlock(base_channels))
        
        self.conv_final = nn.Sequential(
            DepthwiseSeparableConv(base_channels, output_channels),
            nn.Sigmoid()
        )
        
    def forward(self, stego):
        x = self.conv_first(stego)
        
        for layer in self.conv_layers:
            x = layer(x)
        
        secret_recovered = self.conv_final(x)
        return secret_recovered
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Factory function
def create_decoder(config):
    """
    Factory function para crear decoders
    
    Args:
        config: Diccionario con configuración del decoder
        
    Returns:
        Instancia del decoder correspondiente
    """
    decoder_type = config.get('type', 'cnn')
    
    if decoder_type == 'cnn':
        return CNNDecoder(
            base_channels=config.get('base_channels', 10),  # Optimizado a 10 para ~4K params
            num_layers=config.get('num_layers', 6),
            output_channels=config.get('output_channels', 3)
        )
    elif decoder_type == 'lightweight_cnn':
        return LightweightDecoder(
            base_channels=config.get('base_channels', 32),
            num_layers=config.get('num_layers', 5),
            output_channels=config.get('output_channels', 3),
            use_depthwise=config.get('use_depthwise', True),
            use_skip=config.get('use_skip', False)
        )
    elif decoder_type == 'enhanced_lightweight':
        return EnhancedLightweightDecoder(
            base_channels=config.get('base_channels', 32),
            num_layers=config.get('num_layers', 5),
            output_channels=config.get('output_channels', 3),
            use_attention=config.get('use_attention', True)
        )
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")


if __name__ == "__main__":
    # Test decoders
    print("="*60)
    print("Testing CNN Decoder")
    print("="*60)
    
    decoder_cnn = CNNDecoder()
    stego = torch.randn(4, 3, 256, 256)
    
    secret_recovered = decoder_cnn(stego)
    print(f"Input shape: {stego.shape}")
    print(f"Output shape: {secret_recovered.shape}")
    print(f"Parameters: {decoder_cnn.get_num_params():,}")
    
    print("\n" + "="*60)
    print("Testing Lightweight Decoder")
    print("="*60)
    
    decoder_light = LightweightDecoder()
    secret_light = decoder_light(stego)
    print(f"Output shape: {secret_light.shape}")
    print(f"Parameters: {decoder_light.get_num_params():,}")
    
    reduction = (1 - decoder_light.get_num_params() / decoder_cnn.get_num_params()) * 100
    print(f"\nParameter reduction: {reduction:.1f}%")
    
    print("\n" + "="*60)
    print("Testing Enhanced Lightweight Decoder")
    print("="*60)
    
    decoder_enhanced = EnhancedLightweightDecoder()
    secret_enhanced = decoder_enhanced(stego)
    print(f"Output shape: {secret_enhanced.shape}")
    print(f"Parameters: {decoder_enhanced.get_num_params():,}")
