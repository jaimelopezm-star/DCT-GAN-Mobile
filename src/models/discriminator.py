"""
Discriminator Architectures para DCT-GAN Steganography

Implementa:
1. XuNet Discriminator Modificado (Paper Original)
2. Efficient XuNet (Propuesta 1 - optimizado para móviles)

El discriminador actúa como estegoanalizador para hacer
el sistema más robusto contra detección.

Paper: "A Hybrid Steganography Framework Using DCT and GAN"
       Malik et al., Scientific Reports 2025
       DOI: 10.1038/s41598-025-01054-7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SRMFilter(nn.Module):
    """
    Spatial Rich Model (SRM) filter para preprocesamiento
    
    Filtros de alto paso diseñados para detectar modificaciones
    sutiles en imágenes (usado en estegoanálisis).
    
    Basado en: Fridrich & Kodovsky (2012)
    "Rich Models for Steganalysis of Digital Images"
    """
    
    def __init__(self):
        super(SRMFilter, self).__init__()
        
        # Kernel KV del paper (5×5) para 3 canales
        # Este es un ejemplo simplificado del filtro SRM
        kernel = self._get_srm_kernel()

        # Repetir para cada canal RGB.
        # _get_srm_kernel retorna (1, 1, 5, 5), y Conv2d espera (out_ch, in_ch/groups, kH, kW).
        kernel = np.repeat(kernel, repeats=3, axis=0)  # (3, 1, 5, 5)
        kernel = torch.FloatTensor(kernel)

        if kernel.ndim != 4:
            raise ValueError(f"SRM kernel debe ser 4D, recibido shape={tuple(kernel.shape)}")
        
        # Usar Conv2d - stride=1 por defecto
        self.conv = nn.Conv2d(3, 3, kernel_size=5, padding=2, groups=3, bias=False)
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False
        
    def _get_srm_kernel(self):
        """
        Genera un kernel SRM básico 5×5
        
        En el paper usan el kernel KV transpuesto para 3 canales.
        Aquí usamos una aproximación.
        """
        kernel = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ]) / 12.0
        
        return kernel.reshape(1, 1, 5, 5)
    
    def forward(self, x):
        """
        Aplica filtro SRM
        
        Args:
            x: Imagen (B, 3, H, W)
            
        Returns:
            Imagen filtrada (B, 3, H, W)
        """
        # Convolución depthwise (cada canal independiente)
        return self.conv(x)


class XuNetDiscriminator(nn.Module):
    """
    Discriminador basado en XuNet Modificado (Paper Exact)
    
    Arquitectura del paper:
    - Primera capa: kernel 5×5 adaptado a 3 canales RGB
    - 5 capas convolucionales con BatchNorm + Leaky ReLU
    - Spectral Normalization para estabilidad WGAN
    - Global Average Pooling
    - Fully Connected para clasificación
    
    Para WGAN: NO usar Sigmoid al final (raw logits)
    
    Args:
        input_channels: Canales de entrada (default: 3 para RGB)
        base_channels: Canales base (default: 64)
        num_conv_layers: Número de capas conv (default: 5)
        use_srm: Usar filtro SRM inicial (default: False)
        use_spectral_norm: Usar normalización espectral (default: True)
    """
    
    def __init__(
        self,
        input_channels=3,
        base_channels=64,  # Paper uses 64 channels
        num_conv_layers=5,
        use_srm=False,
        use_spectral_norm=True  # Para estabilidad WGAN
    ):
        super(XuNetDiscriminator, self).__init__()
        
        self.use_srm = use_srm
        
        # Filtro SRM para preprocesamiento (opcional)
        if use_srm:
            self.srm_filter = SRMFilter()
        
        # Función para aplicar spectral norm si está habilitada
        def maybe_spectral_norm(module):
            if use_spectral_norm:
                return nn.utils.spectral_norm(module)
            return module
        
        # Primera capa especial: kernel 5×5 (Paper: KV transpuesto)
        self.conv1 = maybe_spectral_norm(
            nn.Conv2d(input_channels, base_channels, kernel_size=5, stride=1, padding=2, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        # Capas convolucionales intermedias con BN
        self.conv_layers = nn.ModuleList()
        in_ch = base_channels
        
        for i in range(num_conv_layers - 1):
            out_ch = min(base_channels * (2 ** (i + 1)), 512)  # Max 512 canales
            
            layer = nn.Sequential(
                maybe_spectral_norm(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
                ),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.conv_layers.append(layer)
            in_ch = out_ch
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Clasificador - NO Sigmoid para WGAN (output raw logits)
        self.fc = nn.Linear(in_ch, 1)
        
    def forward(self, x):
        """
        Forward pass del discriminador
        
        Args:
            x: Imagen (B, 3, 256, 256)
            
        Returns:
            logits: Raw logits (B, 1) - sin Sigmoid para WGAN
                    Valores positivos = más probable cover
                    Valores negativos = más probable stego
        """
        # Preprocesamiento opcional con SRM
        if self.use_srm:
            x = self.srm_filter(x)
        
        # Primera capa con BN
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        
        # Capas convolucionales
        for layer in self.conv_layers:
            x = layer(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Clasificación (raw logits para WGAN)
        logits = self.fc(x)
        
        return logits
    
    def get_num_params(self):
        """Retorna el número total de parámetros"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EfficientXuNet(nn.Module):
    """
    Versión eficiente del discriminador XuNet para Mobile-StegoNet
    
    Optimizaciones:
    - Menos capas convolucionales (4 en vez de 5)
    - Menos canales base (32 en vez de 64)
    - Depthwise separable convolutions opcionales
    - Arquitectura más ligera manteniendo efectividad
    
    Objetivo: ~50% reducción en parámetros
    
    Args:
        input_channels: Canales de entrada (default: 3)
        base_channels: Canales base (default: 32)
        num_conv_layers: Número de capas (default: 4)
        use_srm: Usar filtro SRM (default: True)
        use_depthwise: Usar depthwise separable (default: False)
    """
    
    def __init__(
        self,
        input_channels=3,
        base_channels=32,
        num_conv_layers=4,
        use_srm=True,
        use_depthwise=False
    ):
        super(EfficientXuNet, self).__init__()
        
        self.use_srm = use_srm
        
        if use_srm:
            self.srm_filter = SRMFilter()
        
        # Primera capa
        self.conv1 = nn.Conv2d(input_channels, base_channels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        # Capas convolucionales
        self.conv_layers = nn.ModuleList()
        in_ch = base_channels
        
        for i in range(num_conv_layers - 1):
            out_ch = min(base_channels * (2 ** (i + 1)), 256)  # Máximo 256 en vez de 512
            
            if use_depthwise and i > 0:
                # Usar depthwise separable después de la primera capa
                layer = nn.Sequential(
                    # Depthwise
                    nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, groups=in_ch),
                    # Pointwise
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            
            self.conv_layers.append(layer)
            in_ch = out_ch
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Clasificador más simple
        self.fc = nn.Sequential(
            nn.Linear(in_ch, 128),  # Reducido de 256
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),  # Menos dropout
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass del discriminador eficiente
        
        Args:
            x: Imagen (B, 3, 256, 256)
            
        Returns:
            prob: Probabilidad de que sea stego (B, 1)
        """
        if self.use_srm:
            x = self.srm_filter(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        
        for layer in self.conv_layers:
            x = layer(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        prob = self.fc(x)
        
        return prob
    
    def get_num_params(self):
        """Retorna el número total de parámetros"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SpectralNormDiscriminator(nn.Module):
    """
    Discriminador con Spectral Normalization para entrenamiento estable
    
    Opcional para experimentos futuros.
    Spectral Norm ayuda a estabilizar el entrenamiento de GANs.
    
    Basado en: Miyato et al. (2018)
    "Spectral Normalization for Generative Adversarial Networks"
    """
    
    def __init__(
        self,
        input_channels=3,
        base_channels=64,
        num_conv_layers=5
    ):
        super(SpectralNormDiscriminator, self).__init__()
        
        # Aplicar spectral norm a todas las capas conv
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(input_channels, base_channels, 5, 1, 2)
        )
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        self.conv_layers = nn.ModuleList()
        in_ch = base_channels
        
        for i in range(num_conv_layers - 1):
            out_ch = min(base_channels * (2 ** (i + 1)), 512)
            
            layer = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 2, 1)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.conv_layers.append(layer)
            in_ch = out_ch
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(in_ch, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(256, 1)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        
        for layer in self.conv_layers:
            x = layer(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        return self.fc(x)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Factory function
def create_discriminator(config):
    """
    Factory function para crear discriminadores
    
    Args:
        config: Diccionario con configuración del discriminador
        
    Returns:
        Instancia del discriminador correspondiente
    """
    disc_type = config.get('type', 'xunet_modified')
    
    if disc_type == 'xunet_modified':
        return XuNetDiscriminator(
            input_channels=config.get('input_channels', 3),
            base_channels=config.get('base_channels', 64),  # Paper: 64 channels
            num_conv_layers=config.get('num_conv_layers', 5),
            use_srm=config.get('use_srm', False),
            use_spectral_norm=config.get('use_spectral_norm', True)  # Estabilidad WGAN
        )
    elif disc_type == 'efficient_xunet':
        return EfficientXuNet(
            input_channels=config.get('input_channels', 3),
            base_channels=config.get('base_channels', 32),
            num_conv_layers=config.get('num_conv_layers', 4),
            use_srm=config.get('use_srm', True),
            use_depthwise=config.get('use_depthwise', False)
        )
    elif disc_type == 'spectral_norm':
        return SpectralNormDiscriminator(
            input_channels=config.get('input_channels', 3),
            base_channels=config.get('base_channels', 64),
            num_conv_layers=config.get('num_conv_layers', 5)
        )
    else:
        raise ValueError(f"Unknown discriminator type: {disc_type}")


if __name__ == "__main__":
    # Test discriminators
    print("="*60)
    print("Testing XuNet Discriminator (without SRM)")
    print("="*60)
    
    disc_xunet = XuNetDiscriminator(use_srm=False)
    images = torch.randn(4, 3, 256, 256)
    
    probs = disc_xunet(images)
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {probs.shape}")
    print(f"Output range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
    print(f"Parameters: {disc_xunet.get_num_params():,}")
    
    print("\n" + "="*60)
    print("Testing Efficient XuNet")
    print("="*60)
    
    disc_efficient = EfficientXuNet(use_srm=False)
    probs_eff = disc_efficient(images)
    print(f"Output shape: {probs_eff.shape}")
    print(f"Parameters: {disc_efficient.get_num_params():,}")
    
    reduction = (1 - disc_efficient.get_num_params() / disc_xunet.get_num_params()) * 100
    print(f"\nParameter reduction: {reduction:.1f}%")
    
    print("\n" + "="*60)
    print("Testing Spectral Norm Discriminator")
    print("="*60)
    
    disc_sn = SpectralNormDiscriminator()
    probs_sn = disc_sn(images)
    print(f"Output shape: {probs_sn.shape}")
    print(f"Parameters: {disc_sn.get_num_params():,}")
