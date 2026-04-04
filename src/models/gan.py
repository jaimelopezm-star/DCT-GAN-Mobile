"""
DCT-GAN Complete Model

Integra Encoder + Decoder + Discriminator en una arquitectura completa.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Permitir imports tanto relativos como absolutos
try:
    from .encoder import create_encoder
    from .decoder import create_decoder
    from .discriminator import create_discriminator
except ImportError:
    # Cuando se ejecuta como script directo
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.encoder import create_encoder
    from models.decoder import create_decoder
    from models.discriminator import create_discriminator


class DCTGAN(nn.Module):
    """
    Modelo completo DCT-GAN para esteganografía
    
    Componentes:
    1. Encoder: Oculta imagen secreta en imagen de cobertura (genera stego)
    2. Decoder: Extrae imagen secreta de imagen stego
    3. Discriminator: Distingue entre cover y stego (entrenamiento adversarial)
    
    Flujo:
        Cover + Secret -> Encoder -> Stego
        Stego -> Decoder -> Recovered Secret
        Stego vs Cover -> Discriminator -> Real/Fake probability
    
    Args:
        encoder_config: Configuración del encoder
        decoder_config: Configuración del decoder  
        discriminator_config: Configuración del discriminator
    """
    
    def __init__(self, encoder_config, decoder_config, discriminator_config):
        super(DCTGAN, self).__init__()
        
        self.encoder = create_encoder(encoder_config)
        self.decoder = create_decoder(decoder_config)
        self.discriminator = create_discriminator(discriminator_config)
        
    def forward(self, cover, secret, mode='full'):
        """
        Forward pass del modelo completo
        
        Args:
            cover: Imagen de cobertura (B, 3, H, W)
            secret: Imagen secreta (B, 3, H, W)
            mode: Modo de ejecución
                - 'full': Ejecuta encoder + decoder (training)
                - 'encode': Solo encoder (embedding)
                - 'decode': Solo decoder (extraction)
                - 'discriminate': Solo discriminator
        
        Returns:
            Dependiendo del modo:
            - 'full': (stego, recovered_secret)
            - 'encode': stego
            - 'decode': recovered_secret
            - 'discriminate': probability
        """
        if mode == 'full':
            # Modo de entrenamiento completo
            stego = self.encoder(cover, secret)
            recovered_secret = self.decoder(stego)
            return stego, recovered_secret
            
        elif mode == 'encode':
            # Solo embedding
            stego = self.encoder(cover, secret)
            return stego
            
        elif mode == 'decode':
            # Solo extraction (cover es realmente stego aquí)
            recovered_secret = self.decoder(cover)
            return recovered_secret
            
        elif mode == 'discriminate':
            # Solo discriminación
            prob = self.discriminator(cover)
            return prob
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def embed(self, cover, secret):
        """
        Oculta imagen secreta en imagen de cobertura
        
        Args:
            cover: Imagen de cobertura (B, 3, H, W)
            secret: Imagen secreta (B, 3, H, W)
            
        Returns:
            stego: Imagen esteganográfica (B, 3, H, W)
        """
        return self.encoder(cover, secret)
    
    def extract(self, stego):
        """
        Extrae imagen secreta de imagen esteganográfica
        
        Args:
            stego: Imagen esteganográfica (B, 3, H, W)
            
        Returns:
            recovered_secret: Imagen secreta recuperada (B, 3, H, W)
        """
        return self.decoder(stego)
    
    def discriminate(self, image):
        """
        Determina si una imagen es cover o stego
        
        Args:
            image: Imagen a analizar (B, 3, H, W)
            
        Returns:
            prob: Probabilidad de ser stego (B, 1)
        """
        return self.discriminator(image)
    
    def get_num_params(self):
        """
        Retorna número de parámetros de cada componente
        
        Returns:
            dict con conteos de parámetros
        """
        return {
            'encoder': self.encoder.get_num_params(),
            'decoder': self.decoder.get_num_params(),
            'discriminator': self.discriminator.get_num_params(),
            'total': (
                self.encoder.get_num_params() + 
                self.decoder.get_num_params() + 
                self.discriminator.get_num_params()
            )
        }
    
    def get_generator_params(self):
        """
        Retorna parámetros del generador (encoder + decoder)
        
        Útil para optimizar el generador durante el entrenamiento
        """
        return list(self.encoder.parameters()) + list(self.decoder.parameters())
    
    def get_discriminator_params(self):
        """
        Retorna parámetros del discriminador
        
        Útil para optimizar el discriminator durante el entrenamiento
        """
        return list(self.discriminator.parameters())


def create_dct_gan_from_config(config):
    """
    Crea modelo DCT-GAN desde archivo de configuración
    
    Args:
        config: Diccionario o objeto de configuración con secciones:
                - model.encoder
                - model.decoder
                - model.discriminator
    
    Returns:
        Instancia de DCTGAN
    """
    try:
        encoder_config = config['model']['encoder']
        decoder_config = config['model']['decoder']
        discriminator_config = config['model']['discriminator']
    except (KeyError, TypeError):
        raise ValueError(
            "Config debe tener estructura: config['model']['encoder/decoder/discriminator']"
        )
    
    return DCTGAN(encoder_config, decoder_config, discriminator_config)


def load_pretrained_model(checkpoint_path, device='cuda'):
    """
    Carga modelo preentrenado desde checkpoint
    
    Args:
        checkpoint_path: Ruta al archivo .pth del checkpoint
        device: Dispositivo donde cargar el modelo
        
    Returns:
        Tupla (model, checkpoint_data) donde checkpoint_data contiene
        información adicional como epoch, métricas, etc.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruir configuración del modelo
    encoder_config = checkpoint.get('encoder_config', {})
    decoder_config = checkpoint.get('decoder_config', {})
    discriminator_config = checkpoint.get('discriminator_config', {})
    
    # Crear modelo
    model = DCTGAN(encoder_config, decoder_config, discriminator_config)
    
    # Cargar pesos
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def save_model(
    model,
    save_path,
    epoch=None,
    optimizer_g=None,
    optimizer_d=None,
    metrics=None,
    config=None
):
    """
    Guarda modelo y estado de entrenamiento
    
    Args:
        model: Instancia de DCTGAN
        save_path: Ruta donde guardar
        epoch: Época actual (opcional)
        optimizer_g: Optimizador del generador (opcional)
        optimizer_d: Optimizador del discriminador (opcional)
        metrics: Diccionario con métricas (opcional)
        config: Configuración del modelo (opcional)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'encoder_config': config.get('model', {}).get('encoder', {}) if config else {},
        'decoder_config': config.get('model', {}).get('decoder', {}) if config else {},
        'discriminator_config': config.get('model', {}).get('discriminator', {}) if config else {},
    }
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if optimizer_g is not None:
        checkpoint['optimizer_g_state_dict'] = optimizer_g.state_dict()
    
    if optimizer_d is not None:
        checkpoint['optimizer_d_state_dict'] = optimizer_d.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    # Test del modelo completo
    print("="*60)
    print("Testing DCT-GAN Complete Model")
    print("="*60)
    
    # Configuración de ejemplo (paper original - optimizada para ~50K params)
    encoder_config = {
        'type': 'resnet',
        'input_channels': 6,
        'base_channels': 10,  # Optimizado: 16 -> 10 (~17K params)
        'num_residual_blocks': 9
    }
    
    decoder_config = {
        'type': 'cnn',
        'base_channels': 10,  # Optimizado: 16 -> 10 (~4K params)
        'num_layers': 6,
        'output_channels': 3
    }
    
    discriminator_config = {
        'type': 'xunet_modified',
        'input_channels': 3,
        'base_channels': 4,  # Optimizado: 8 -> 4 (~26K params)
        'num_conv_layers': 5,
        'use_srm': False  # Deshabilitar por bug en PyTorch 2.10
    }
    
    # Crear modelo
    model = DCTGAN(encoder_config, decoder_config, discriminator_config)
    
    # Test con datos dummy
    batch_size = 4
    cover = torch.randn(batch_size, 3, 256, 256)
    secret = torch.randn(batch_size, 3, 256, 256)
    
    print(f"\nInput shapes:")
    print(f"  Cover: {cover.shape}")
    print(f"  Secret: {secret.shape}")
    
    # Modo full (entrenamiento)
    print(f"\nMode: full")
    stego, recovered = model(cover, secret, mode='full')
    print(f"  Stego: {stego.shape}")
    print(f"  Recovered: {recovered.shape}")
    
    # Modo encode
    print(f"\nMode: encode")
    stego_only = model(cover, secret, mode='encode')
    print(f"  Stego: {stego_only.shape}")
    
    # Modo decode
    print(f"\nMode: decode")
    recovered_only = model(stego, secret, mode='decode')
    print(f"  Recovered: {recovered_only.shape}")
    
    # Modo discriminate
    print(f"\nMode: discriminate")
    prob_cover = model(cover, secret, mode='discriminate')
    prob_stego = model(stego, secret, mode='discriminate')
    print(f"  Prob(Cover is stego): {prob_cover.mean().item():.4f}")
    print(f"  Prob(Stego is stego): {prob_stego.mean().item():.4f}")
    
    # Parámetros
    print(f"\nModel Parameters:")
    params = model.get_num_params()
    for name, count in params.items():
        print(f"  {name}: {count:,}")
    
    print("\n" + "="*60)
    print("Testing Mobile-StegoNet")
    print("="*60)
    
    # Configuración móvil
    mobile_encoder_config = {
        'type': 'mobilenetv3_small',
        'input_channels': 6,
        'width_multiplier': 1.0,
        'use_se': True
    }
    
    mobile_decoder_config = {
        'type': 'lightweight_cnn',
        'base_channels': 32,
        'num_layers': 5,
        'output_channels': 3,
        'use_depthwise': True
    }
    
    mobile_disc_config = {
        'type': 'efficient_xunet',
        'input_channels': 3,
        'base_channels': 32,
        'num_conv_layers': 4,
        'use_srm': False  # Deshabilitar por bug en PyTorch 2.10
    }
    
    mobile_model = DCTGAN(mobile_encoder_config, mobile_decoder_config, mobile_disc_config)
    
    stego_mobile, recovered_mobile = mobile_model(cover, secret, mode='full')
    print(f"\nOutput shapes:")
    print(f"  Stego: {stego_mobile.shape}")
    print(f"  Recovered: {recovered_mobile.shape}")
    
    print(f"\nMobile Model Parameters:")
    mobile_params = mobile_model.get_num_params()
    for name, count in mobile_params.items():
        print(f"  {name}: {count:,}")
    
    # Comparación
    print(f"\nParameter Reduction vs Base Model:")
    for name in ['encoder', 'decoder', 'discriminator', 'total']:
        if name in params and name in mobile_params:
            reduction = (1 - mobile_params[name] / params[name]) * 100
            print(f"  {name}: {reduction:.1f}%")
