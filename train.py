"""
Train Script - Script Principal de Entrenamiento

Entrena el modelo DCT-GAN completo siguiendo la configuración del paper.

Uso:
    python train.py --config configs/base_config.yaml
    python train.py --config configs/mobile_config.yaml --resume checkpoints/checkpoint_epoch_050.pth
    
Paper: Malik et al. (2025)
Target: PSNR 58.27 dB, SSIM 0.942
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.gan import DCTGAN
from training.trainer import DCTGANTrainer
from data.bossbase_dataset import BOSSBaseDataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image


class SteganographyDataset(Dataset):
    """
    Dataset de pares (cover, secret) para esteganografía
    
    Carga imágenes reales y crea pares aleatorios para entrenamiento.
    Compatible con ImageNet, BOSSBase, y otros datasets.
    
    Args:
        image_dataset: Dataset de imágenes (PyTorch ImageFolder)
        transform: Transformaciones a aplicar (opcional)
        mode: 'train', 'val', o 'test'
    """
    
    def __init__(self, image_dataset, transform=None, mode='train'):
        self.images = image_dataset
        self.transform = transform
        self.mode = mode
        
        print(f"[{mode.upper()}] Dataset initialized with {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Retorna un par de imágenes (cover, secret)
        
        Cover: imagen en posición idx
        Secret: imagen aleatoria del dataset
        """
        # Cover image
        cover, _ = self.images[idx]
        
        # Secret image (aleatorio)
        import random
        secret_idx = random.randint(0, len(self.images) - 1)
        secret, _ = self.images[secret_idx]
        
        return {
            'cover': cover,
            'secret': secret
        }


class SteganoDataset(Dataset):
    """
    Dataset SINTÉTICO para testing rápido
    ⚠️  NO USAR PARA REPLICAR PAPER - solo para testing de código
    
    Args:
        image_size: Tamaño de las imágenes (256x256)
        num_samples: Número de pares a generar
        mode: 'train' o 'val'
    """
    
    def __init__(self, 
                 image_size: int = 256,
                 num_samples: int = 10000,
                 mode: str = 'train'):
        
        self.image_size = image_size
        self.num_samples = num_samples
        self.mode = mode
        
        print(f"[{mode.upper()}] Initializing dataset with {num_samples} SYNTHETIC pairs")
        print("⚠️  WARNING: Using synthetic data. NOT SUITABLE for paper replication.")
        print("   For real training, use: --dataset imagenet")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Genera pares sintéticos aleatorios"""
        cover = torch.rand(3, self.image_size, self.image_size)
        secret = torch.rand(3, self.image_size, self.image_size)
        
        return {
            'cover': cover,
            'secret': secret
        }


def load_config(config_path: Path) -> dict:
    """
    Carga configuración desde archivo YAML
    
    Args:
        config_path: Path al archivo de configuración
        
    Returns:
        Diccionario de configuración
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict, dataset_type: str = 'synthetic', dataset_path: str = None) -> tuple:
    """
    Crea DataLoaders de entrenamiento y validación
    
    Args:
        config: Diccionario de configuración
        dataset_type: 'synthetic', 'imagenet', o 'bossbase'
        dataset_path: Path al dataset real (ImageNet o BOSSBase)
        
    Returns:
        (train_loader, val_loader)
    """
    from torchvision import datasets, transforms
    
    data_config = config.get('data', {})
    
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 4)
    pin_memory = data_config.get('pin_memory', True)
    image_size = data_config.get('image_size', 256)
    
    if dataset_type == 'bossbase':
        # ============================================
        # BOSSBASE DATASET (Real grayscale images)
        # ============================================
        print("\n📊 Usando dataset BOSSBase (imágenes reales)")
        print(f"   Path: {dataset_path}")
        
        if not dataset_path:
            print("❌ Error: --dataset-path requerido para BOSSBase")
            print("   Uso: --dataset bossbase --dataset-path /workspace/BOSSbase_prepared")
            raise ValueError("dataset_path requerido para BOSSBase")
        
        # Verificar dataset
        success, message = BOSSBaseDataset.verify_dataset(dataset_path)
        if not success:
            print(f"❌ Error: {message}")
            print("\n   Ejecuta primero:")
            print("   python prepare_bossbase.py --source /workspace/BOSSbase --output /workspace/BOSSbase_prepared")
            raise ValueError("BOSSBase dataset no preparado")
        
        print(f"✅ {message}")
        
        # Cargar splits
        train_dataset = BOSSBaseDataset(dataset_path, split='train')
        val_dataset = BOSSBaseDataset(dataset_path, split='val')
        
        print(f"✅ BOSSBase cargado:")
        print(f"   - Train: {len(train_dataset)} pares")
        print(f"   - Val: {len(val_dataset)} pares")
    
    elif dataset_type == 'imagenet' and dataset_path:
        # ============================================
        # DATASET REAL (ImageNet)
        # ============================================
        print("\n📊 Usando dataset REAL (ImageNet)")
        print(f"   Path: {dataset_path}")
        
        # Transformaciones para ImageNet
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            # Normalizar entre [0, 1] (ya que ToTensor hace esto)
        ])
        
        # Cargar splits
        train_dir = Path(dataset_path) / "train"
        val_dir = Path(dataset_path) / "val"
        
        if not train_dir.exists() or not val_dir.exists():
            print(f"\n❌ Error: No se encontraron splits en {dataset_path}")
            print("   Ejecuta: python scripts/prepare_dataset.py")
            print("\n⚠️  Usando datos sintéticos como fallback...")
            use_real_data = False
        else:
            # Cargar ImageNet usando ImageFolder
            train_imagenet = datasets.ImageFolder(root=str(train_dir), transform=transform)
            val_imagenet = datasets.ImageFolder(root=str(val_dir), transform=transform)
            
            # Crear pares para esteganografía
            train_dataset = SteganographyDataset(train_imagenet, mode='train')
            val_dataset = SteganographyDataset(val_imagenet, mode='val')
            
            print(f"✅ ImageNet cargado:")
            print(f"   - Train: {len(train_dataset)} pares")
            print(f"   - Val: {len(val_dataset)} pares")
    
    else:
        # ============================================
        # DATASET SINTÉTICO (solo para testing)
        # ============================================
        print("\n📊 Usando dataset SINTÉTICO (testing only)")
        print("   ⚠️  NO adecuado para replicar métricas del paper")
        
        train_size = data_config.get('train_size', 40000)
        val_size = data_config.get('val_size', 5000)
        
        train_dataset = SteganoDataset(
            image_size=image_size,
            num_samples=train_size,
            mode='train'
        )
        
        val_dataset = SteganoDataset(
            image_size=image_size,
            num_samples=val_size,
            mode='val'
        )
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Evitar último batch incompleto
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"\n📦 DataLoaders created:")
    print(f"  - Train batches: {len(train_loader)} (batch_size={batch_size})")
    print(f"  - Val batches: {len(val_loader)} (batch_size={batch_size})")
    
    return train_loader, val_loader


def create_model(config: dict) -> DCTGAN:
    """
    Crea el modelo DCT-GAN desde configuración
    
    Args:
        config: Diccionario de configuración
        
    Returns:
        Modelo DCTGAN
    """
    model_config = config.get('model', {})
    
    encoder_config = model_config.get('encoder', {})
    decoder_config = model_config.get('decoder', {})
    discriminator_config = model_config.get('discriminator', {})
    
    model = DCTGAN(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        discriminator_config=discriminator_config
    )
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    disc_params = sum(p.numel() for p in model.discriminator.parameters())
    
    print(f"\nModel created:")
    print(f"  - Encoder params: {encoder_params:,}")
    print(f"  - Decoder params: {decoder_params:,}")
    print(f"  - Discriminator params: {disc_params:,}")
    print(f"  - Total params: {total_params:,}")
    print(f"  - Target: 49,950 (deviation: {((total_params - 49950) / 49950 * 100):+.1f}%)")
    
    return model


def main():
    """
    Función principal de entrenamiento
    """
    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Train DCT-GAN Steganography Model')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu/cuda). Auto-detect if not specified')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save logs')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'imagenet', 'bossbase'],
                       help='Dataset to use (synthetic/imagenet/bossbase)')
    parser.add_argument('--dataset-path', type=str, default='data/imagenet2012/splits',
                       help='Path to ImageNet splits (if using --dataset imagenet)')
    
    args = parser.parse_args()
    
    # Banner
    print("="*70)
    print("DCT-GAN STEGANOGRAPHY TRAINING")
    print("Paper: Malik et al. (2025)")
    print("Target: PSNR 58.27 dB, SSIM 0.942")
    print("="*70)
    
    # Cargar configuración
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print(f"   Available configs:")
        configs_dir = Path('configs')
        if configs_dir.exists():
            for cfg in configs_dir.glob('*.yaml'):
                print(f"     - {cfg}")
        return
    
    print(f"\n1. Loading config from: {config_path}")
    config = load_config(config_path)
    print(f"   Project: {config.get('project', {}).get('name', 'N/A')}")
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n2. Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Crear directorios
    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n3. Directories:")
    print(f"   Checkpoints: {checkpoint_dir}")
    print(f"   Logs: {log_dir}")
    
    # Crear modelo
    print(f"\n4. Creating model...")
    model = create_model(config)
    
    # Crear DataLoaders
    print(f"\n5. Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        config, 
        dataset_type=args.dataset,
        dataset_path=args.dataset_path if args.dataset in ['imagenet', 'bossbase'] else None
    )
    
    # Crear trainer
    print(f"\n6. Initializing trainer...")
    trainer = DCTGANTrainer(
        model=model,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    
    # Resumir desde checkpoint si se especifica
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\n7. Resuming from checkpoint: {resume_path}")
            trainer.load_checkpoint(resume_path)
        else:
            print(f"\n⚠️  Checkpoint not found: {resume_path}")
            print("   Starting from scratch...")
    
    # Configuración de entrenamiento
    num_epochs = config.get('training', {}).get('num_epochs', 100)
    
    # Get update strategy from config
    update_strategy = config.get('training', {}).get('update_strategy', {})
    disc_updates = update_strategy.get('discriminator_updates_per_batch', 5)
    gen_updates = update_strategy.get('generator_updates_per_batch', 1)
    
    # Get actual loss weights from config
    loss_config = config.get('loss', {})
    alpha = loss_config.get('alpha', 0.3)
    beta = loss_config.get('beta', 15.0)
    gamma = loss_config.get('gamma', 0.03)
    disc_type = config.get('training', {}).get('optimizer', {}).get('discriminator', {}).get('type', 'sgd').upper()
    
    print(f"\n{'='*70}")
    print(f"STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {config.get('data', {}).get('batch_size', 32)}")
    print(f"Strategy: {disc_updates}:{gen_updates} (Discriminator:Generator)")
    print(f"Optimizer G: Adam lr={config.get('training', {}).get('optimizer', {}).get('generator', {}).get('lr', 1e-3)}")
    print(f"Optimizer D: {disc_type} lr={config.get('training', {}).get('optimizer', {}).get('discriminator', {}).get('lr', 1e-3)}")
    print(f"Loss weights: α={alpha}, β={beta}, γ={gamma}")
    print(f"{'='*70}\n")
    
    # Entrenar
    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            save_frequency=10,
            early_stopping_patience=50  # Aumentado de 20 a 50 para dar más margen
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint(
            epoch=trainer.current_epoch,
            metrics={'interrupted': True},
            filename='checkpoint_interrupted.pth'
        )
        print("Checkpoint saved. You can resume later with --resume")
    
    print(f"\n{'='*70}")
    print("TRAINING FINISHED")
    print(f"{'='*70}")
    print(f"Best PSNR: {trainer.best_psnr:.2f} dB (target: 58.27 dB)")
    print(f"Best model: {checkpoint_dir / 'best_model.pth'}")
    print(f"Logs: {log_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
