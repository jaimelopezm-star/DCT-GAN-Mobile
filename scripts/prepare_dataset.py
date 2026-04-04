"""
Dataset Preparation Script

Crea splits de train/val/test desde ImageNet descargado.
Split del paper: 40,000 train / 5,000 val / 5,000 test

Uso:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --data-dir data/imagenet2012
"""

import argparse
import random
import shutil
from pathlib import Path
from tqdm import tqdm


def create_splits(source_dir, output_dir, train_size=40000, val_size=5000, test_size=5000, seed=42):
    """
    Crea splits train/val/test desde directorio de imágenes
    
    Args:
        source_dir: Directorio con todas las imágenes
        output_dir: Directorio para guardar splits
        train_size: Número de imágenes para entrenamiento
        val_size: Número de imágenes para validación
        test_size: Número de imágenes para test
        seed: Semilla aleatoria para reproducibilidad
    """
    print("="*70)
    print("CREANDO SPLITS DE DATASET")
    print("="*70)
    
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Encontrar todas las imágenes
    print(f"\n📁 Buscando imágenes en: {source_dir}")
    images = list(source_dir.glob("*.JPEG")) + list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
    
    print(f"✅ Encontradas {len(images)} imágenes")
    
    # Verificar que hay suficientes imágenes
    total_needed = train_size + val_size + test_size
    if len(images) < total_needed:
        print(f"\n⚠️  ADVERTENCIA: Solo hay {len(images)} imágenes disponibles")
        print(f"   Se necesitan {total_needed} para el split completo")
        print(f"   Ajustando tamaños de split proporcionalmente...")
        
        # Ajustar proporcionalmente
        ratio = len(images) / total_needed
        train_size = int(train_size * ratio)
        val_size = int(val_size * ratio)
        test_size = len(images) - train_size - val_size
        
        print(f"   Nuevo split: {train_size} train / {val_size} val / {test_size} test")
    
    # Shuffle con seed fijo para reproducibilidad
    random.seed(seed)
    random.shuffle(images)
    
    # Crear splits
    splits = {
        'train': images[:train_size],
        'val': images[train_size:train_size+val_size],
        'test': images[train_size+val_size:train_size+val_size+test_size]
    }
    
    # Crear directorios y copiar/mover imágenes
    for split_name, split_images in splits.items():
        # Para PyTorch ImageFolder necesitamos subdirectorio 'all'
        split_dir = output_dir / split_name / 'all'
        split_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📦 Creando split '{split_name}': {len(split_images)} imágenes")
        print(f"   Directorio: {split_dir}")
        
        for img in tqdm(split_images, desc=f"Copiando {split_name}"):
            target = split_dir / img.name
            if not target.exists():
                shutil.copy2(img, target)
        
        print(f"✅ Split '{split_name}' creado: {len(list(split_dir.glob('*')))} imágenes")
    
    # Resumen
    print("\n" + "="*70)
    print("✅ SPLITS CREADOS EXITOSAMENTE")
    print("="*70)
    print(f"\nEstructura creada en: {output_dir}")
    print(f"  ├── train/all/  ({len(splits['train'])} imágenes)")
    print(f"  ├── val/all/    ({len(splits['val'])} imágenes)")
    print(f"  └── test/all/   ({len(splits['test'])} imágenes)")
    print(f"\nTotal: {sum(len(s) for s in splits.values())} imágenes")
    
    # Verificar
    print("\n" + "="*70)
    print("VERIFICANDO SPLITS")
    print("="*70)
    
    for split_name in ['train', 'val', 'test']:
        split_dir = output_dir / split_name / 'all'
        count = len(list(split_dir.glob('*')))
        expected = len(splits[split_name])
        
        if count == expected:
            print(f"✅ {split_name}: {count}/{expected} imágenes")
        else:
            print(f"⚠️  {split_name}: {count}/{expected} imágenes (falta {expected-count})")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Prepara splits de dataset para entrenamiento'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/imagenet2012',
        help='Directorio raíz del dataset'
    )
    parser.add_argument(
        '--train-size',
        type=int,
        default=40000,
        help='Tamaño del conjunto de entrenamiento (default: 40000)'
    )
    parser.add_argument(
        '--val-size',
        type=int,
        default=5000,
        help='Tamaño del conjunto de validación (default: 5000)'
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=5000,
        help='Tamaño del conjunto de test (default: 5000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla aleatoria para reproducibilidad (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Directorio source (todas las imágenes organizadas)
    source_dir = Path(args.data_dir) / "organized" / "all"
    
    if not source_dir.exists():
        print(f"❌ Error: No se encontró {source_dir}")
        print("\nPrimero ejecuta:")
        print("  python scripts/download_imagenet.py --method kaggle")
        return
    
    # Directorio output (splits)
    output_dir = Path(args.data_dir) / "splits"
    
    # Crear splits
    success = create_splits(
        source_dir=source_dir,
        output_dir=output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed
    )
    
    if success:
        print("\n" + "="*70)
        print("🎉 DATASET LISTO PARA ENTRENAMIENTO")
        print("="*70)
        print("\nPróximo paso:")
        print("  python train.py --config configs/base_config.yaml")


if __name__ == "__main__":
    main()
