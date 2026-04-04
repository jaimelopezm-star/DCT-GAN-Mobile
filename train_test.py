"""
Script de prueba rápida - 2 épocas para validar implementación
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import random
import yaml
import logging
from pathlib import Path

from src.models.gan import DCTGAN
from src.training.losses import HybridLoss, WassersteinGANLoss
from utils.trainer import Trainer


class SteganographyDataset(Dataset):
    """
    Dataset personalizado para esteganografía
    Devuelve pares (cover_image, secret_image) en lugar de (image, label)
    """
    
    def __init__(self, image_folder_dataset):
        """
        Args:
            image_folder_dataset: ImageFolder dataset
        """
        self.dataset = image_folder_dataset
        self.images = [(img, label) for img, label in image_folder_dataset]
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Cover: imagen en el índice actual
        cover, _ = self.images[idx]
        
        # Secret: imagen aleatoria diferente
        secret_idx = random.randint(0, len(self.images) - 1)
        while secret_idx == idx:  #  Evitar que cover y secret sean la misma imagen
            secret_idx = random.randint(0, len(self.images) - 1)
        secret, _ = self.images[secret_idx]
        
        return cover, secret

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TrainTest')


def load_config(config_path='configs/test_config.yaml'):
    """Carga configuración desde YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_data_loaders(config):
    """
    Crea DataLoaders para train/val con pares de imágenes (cover, secret)
    """
    data_config = config.get('data', {})
    
    # Transformaciones
    transform = transforms.Compose([
        transforms.Resize((data_config.get('image_size', 256), data_config.get('image_size', 256))),
        transforms.ToTensor()
    ])
    
    # ImageFolder datasets base
    train_imagefolder = ImageFolder(
        root='data/imagenet2012/splits/train',
        transform=transform
    )
    
    val_imagefolder = ImageFolder(
        root='data/imagenet2012/splits/val',
        transform=transform
    )
    
    # Wrap en SteganographyDataset para generar pares (cover, secret)
    train_dataset = SteganographyDataset(train_imagefolder)
    val_dataset = SteganographyDataset(val_imagefolder)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.get('batch_size', 16),
        shuffle=True,
        num_workers=data_config.get('num_workers', 2),
        pin_memory=data_config.get('pin_memory', True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.get('batch_size', 16),
        shuffle=False,
        num_workers=data_config.get('num_workers', 2),
        pin_memory=data_config.get('pin_memory', True)
    )
    
    return train_loader, val_loader


def main():
    """
    Test rápido: 2 épocas para validar que todo funciona
    """
    logger.info("🔥 Iniciando test de entrenamiento (2 épocas)")
    
    # Configuración
    config = load_config('configs/test_config.yaml')
    logger.info(f"✅ Configuración cargada: {config.get('project_name', 'test')}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"📱 Device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Data loaders
    logger.info("📦 Cargando datos...")
    train_loader, val_loader = get_data_loaders(config)
    logger.info(f"   Train: {len(train_loader.dataset)} imágenes ({len(train_loader)} batches)")
    logger.info(f"   Val: {len(val_loader.dataset)} imágenes ({len(val_loader)} batches)")
    
    # Modelo
    logger.info("🏗️  Construyendo modelo...")
    
    # Extraer configuraciones individuales
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    encoder_config = model_config.get('encoder', {})
    decoder_config = model_config.get('decoder', {})
    discriminator_config = model_config.get('discriminator', {})
    
    model = DCTGAN(encoder_config, decoder_config, discriminator_config)
    
    # Crear funciones de pérdida
    loss_weights = training_config.get('loss_weights', {})
    model.hybrid_loss = HybridLoss(
        alpha=loss_weights.get('alpha', 0.3),
        beta=loss_weights.get('beta', 15.0),
        gamma=loss_weights.get('gamma', 0.03),
        use_wgan=True
    )
    model.wgan_loss = WassersteinGANLoss()
    
    logger.info("✅ Funciones de pérdida creadas:")
    logger.info(f"   - MSE weight (α): {loss_weights.get('alpha', 0.3)}")
    logger.info(f"   - BCE weight (β): {loss_weights.get('beta', 15.0)}")
    logger.info(f"   - Adversarial weight (γ): {loss_weights.get('gamma', 0.03)}")
    
    # Contar parámetros manualmente
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    disc_params = sum(p.numel() for p in model.discriminator.parameters())
    
    logger.info(f"   Parámetros totales: {total_params:,}")
    logger.info(f"   - Encoder: {encoder_params:,}")
    logger.info(f"   - Decoder: {decoder_params:,}")
    logger.info(f"   - Discriminator: {disc_params:,}")
    
    # Optimizadores
    opt_g_config = training_config.get('optimizer_G', {})
    opt_d_config = training_config.get('optimizer_D', {})
    
    optimizer_g = optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=opt_g_config.get('lr', 0.001),
        betas=tuple(opt_g_config.get('betas', [0.9, 0.999])),
        weight_decay=opt_g_config.get('weight_decay', 0.005)
    )
    
    optimizer_d = optim.Adam(
        model.discriminator.parameters(),
        lr=opt_d_config.get('lr', 0.001),
        betas=tuple(opt_d_config.get('betas', [0.9, 0.999])),
        weight_decay=opt_d_config.get('weight_decay', 0.005)
    )
    
    # Schedulers
    scheduler_g = optim.lr_scheduler.StepLR(
        optimizer_g,
        step_size=training_config.get('lr_decay_step', 30),
        gamma=training_config.get('lr_decay_gamma', 0.5)
    )
    
    scheduler_d = optim.lr_scheduler.StepLR(
        optimizer_d,
        step_size=training_config.get('lr_decay_step', 30),
        gamma=training_config.get('lr_decay_gamma', 0.5)
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        optimizers={'generator': optimizer_g, 'discriminator': optimizer_d},
        schedulers={'generator': scheduler_g, 'discriminator': scheduler_d},
        config=config,
        device=device
    )
    
    # Test rápido: 2 épocas
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST: 2 épocas para validar implementación")
    logger.info("="*60 + "\n")
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,  # Solo 2 épocas de prueba
        checkpoint_dir='checkpoints/test',
        val_every=1
    )
    
    logger.info("\n" + "="*60)
    logger.info("✅ TEST COMPLETADO - Código funcionando correctamente!")
    logger.info("="*60)
    logger.info("\n📋 Próximos pasos:")
    logger.info("1. ✅ Código validado (trainer.py funciona)")
    logger.info("2. ⏳ Subir a GitHub")
    logger.info("3. ⏳ Entrenar 100 épocas en Colab GPU")
    logger.info("4. ⏳ Validar métricas vs paper")
    

if __name__ == '__main__':
    main()
