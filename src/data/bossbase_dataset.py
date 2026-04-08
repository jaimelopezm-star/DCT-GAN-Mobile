"""
BOSSBase Dataset Loader para DCT-GAN
Carga imágenes del dataset BOSSBase (10,000 imágenes grayscale 512×512)
Convertidas a RGB 256×256 para entrenamiento
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import random


class BOSSBaseDataset(Dataset):
    """
    Dataset loader para BOSSBase
    
    BOSSBase: 10,000 imágenes grayscale naturales (512×512)
    Preparadas: Convertidas a RGB 256×256 en splits train/val/test
    
    Args:
        root_dir: Path al directorio preparado (con subdirs train/val/test)
        split: 'train', 'val', o 'test'
        transform: Transformaciones a aplicar (opcional)
    
    Usage:
        train_dataset = BOSSBaseDataset('/workspace/BOSSbase_prepared', split='train')
        val_dataset = BOSSBaseDataset('/workspace/BOSSbase_prepared', split='val')
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Inicializa el dataset BOSSBase
        
        Args:
            root_dir: Path al directorio preparado
            split: 'train', 'val', o 'test'
            transform: Transformaciones (si None, usa default)
        """
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.transform = transform or self._default_transform()
        
        # Verificar que existe el directorio
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"BOSSBase directory not found: {self.root_dir}\n"
                f"Run: python prepare_bossbase.py --source /workspace/BOSSbase --output {Path(root_dir)}"
            )
        
        # Listar todas las imágenes PNG
        self.images = sorted(list(self.root_dir.glob('*.png')))
        
        if len(self.images) == 0:
            raise ValueError(
                f"No PNG images found in {self.root_dir}\n"
                f"Expected structure: {root_dir}/train/*.png, {root_dir}/val/*.png, etc."
            )
        
        print(f"[{split.upper()}] BOSSBase loaded: {len(self.images)} images from {self.root_dir}")
    
    def _default_transform(self):
        """
        Transformaciones por defecto para BOSSBase
        Ya están en 256×256 RGB, solo necesitamos ToTensor y normalizar
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                               std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Retorna un par (cover, secret) para esteganografía
        
        Cover: imagen en posición idx
        Secret: imagen aleatoria del dataset
        
        Returns:
            dict con keys 'cover' y 'secret' (tensors 3×256×256)
        """
        # Cargar cover image
        cover_path = self.images[idx]
        cover = Image.open(cover_path).convert('RGB')
        
        # Cargar secret image (aleatorio)
        secret_idx = random.randint(0, len(self.images) - 1)
        secret_path = self.images[secret_idx]
        secret = Image.open(secret_path).convert('RGB')
        
        # Aplicar transformaciones
        if self.transform:
            cover = self.transform(cover)
            secret = self.transform(secret)
        
        return {
            'cover': cover,
            'secret': secret
        }
    
    def get_image_path(self, idx):
        """Retorna el path de la imagen en posición idx"""
        return self.images[idx]
    
    @staticmethod
    def verify_dataset(root_dir):
        """
        Verifica que el dataset BOSSBase esté correctamente preparado
        
        Args:
            root_dir: Path al directorio preparado
            
        Returns:
            (bool, str): (success, message)
        """
        root_path = Path(root_dir)
        
        # Verificar estructura
        if not root_path.exists():
            return False, f"Directory not found: {root_path}"
        
        # Verificar splits
        required_splits = ['train', 'val', 'test']
        for split in required_splits:
            split_dir = root_path / split
            if not split_dir.exists():
                return False, f"Split directory not found: {split_dir}"
            
            # Contar imágenes
            images = list(split_dir.glob('*.png'))
            if len(images) == 0:
                return False, f"No images found in: {split_dir}"
            
            print(f"  ✓ {split}: {len(images)} images")
        
        return True, "BOSSBase dataset verified successfully"


if __name__ == "__main__":
    # Test del dataset
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python bossbase_dataset.py /path/to/BOSSbase_prepared")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    print("="*70)
    print("TESTING BOSSBASE DATASET LOADER")
    print("="*70)
    
    # Verificar dataset
    print("\n1. Verifying dataset structure...")
    success, message = BOSSBaseDataset.verify_dataset(dataset_path)
    
    if not success:
        print(f"❌ {message}")
        sys.exit(1)
    
    print(f"✅ {message}")
    
    # Cargar train split
    print("\n2. Loading train split...")
    try:
        train_dataset = BOSSBaseDataset(dataset_path, split='train')
        print(f"✅ Train dataset loaded: {len(train_dataset)} pairs")
    except Exception as e:
        print(f"❌ Error loading train dataset: {e}")
        sys.exit(1)
    
    # Cargar val split
    print("\n3. Loading val split...")
    try:
        val_dataset = BOSSBaseDataset(dataset_path, split='val')
        print(f"✅ Val dataset loaded: {len(val_dataset)} pairs")
    except Exception as e:
        print(f"❌ Error loading val dataset: {e}")
        sys.exit(1)
    
    # Test __getitem__
    print("\n4. Testing data loading...")
    try:
        sample = train_dataset[0]
        cover = sample['cover']
        secret = sample['secret']
        
        print(f"✅ Sample loaded successfully:")
        print(f"   - Cover shape: {cover.shape}")
        print(f"   - Secret shape: {secret.shape}")
        print(f"   - Cover range: [{cover.min():.3f}, {cover.max():.3f}]")
        print(f"   - Secret range: [{secret.min():.3f}, {secret.max():.3f}]")
        
        # Verificar que son tensors
        assert isinstance(cover, torch.Tensor), "Cover should be a tensor"
        assert isinstance(secret, torch.Tensor), "Secret should be a tensor"
        
        # Verificar shape
        assert cover.shape == (3, 256, 256), f"Expected (3, 256, 256), got {cover.shape}"
        assert secret.shape == (3, 256, 256), f"Expected (3, 256, 256), got {secret.shape}"
        
        print("✅ All checks passed!")
        
    except Exception as e:
        print(f"❌ Error testing data loading: {e}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("BOSSBASE DATASET LOADER TEST: SUCCESS")
    print("="*70)
    print("\n🚀 Ready to train with:")
    print(f"   python train.py --config configs/bossbase_config.yaml --dataset bossbase --dataset-path {dataset_path}")
