#!/usr/bin/env python3
"""
Script para preparar BOSSBase dataset para entrenamiento DCT-GAN
Convierte imágenes PGM a PNG, redimensiona a 256x256, y hace split train/val/test
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

def prepare_bossbase(source_dir, output_dir, split_ratios=(0.8, 0.1, 0.1)):
    """
    Prepara BOSSBase para entrenamiento
    
    Args:
        source_dir: Directorio con imágenes PGM
        output_dir: Directorio de salida
        split_ratios: (train, val, test) ratios
    """
    print("=" * 70)
    print("PREPARANDO BOSSBASE DATASET")
    print("=" * 70)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Verificar que existe el directorio fuente
    if not source_path.exists():
        print(f"❌ ERROR: Directorio fuente no existe: {source_path}")
        return
    
    # Crear subdirectorios
    print(f"\n1. Creando estructura de directorios en: {output_path}")
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {split}/")
    
    # Listar todas las imágenes
    print(f"\n2. Buscando imágenes PGM en: {source_path}")
    images = sorted(list(source_path.glob('*.pgm')))
    total = len(images)
    
    if total == 0:
        print(f"❌ ERROR: No se encontraron imágenes .pgm en {source_path}")
        print(f"   Contenido del directorio:")
        for item in source_path.iterdir():
            print(f"   - {item.name}")
        return
    
    print(f"   ✓ Total imágenes encontradas: {total}")
    
    # Calcular splits
    train_end = int(total * split_ratios[0])
    val_end = train_end + int(total * split_ratios[1])
    
    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }
    
    print(f"\n3. Splits:")
    print(f"   - Train: {len(splits['train'])} imágenes ({split_ratios[0]*100:.0f}%)")
    print(f"   - Val:   {len(splits['val'])} imágenes ({split_ratios[1]*100:.0f}%)")
    print(f"   - Test:  {len(splits['test'])} imágenes ({split_ratios[2]*100:.0f}%)")
    
    # Convertir y guardar
    print(f"\n4. Procesando imágenes...")
    print(f"   - Convirtiendo PGM → PNG")
    print(f"   - Grayscale → RGB (replicar canales)")
    print(f"   - Redimensionando 512x512 → 256x256")
    
    total_processed = 0
    
    for split_name, split_images in splits.items():
        print(f"\n   [{split_name.upper()}]")
        
        with tqdm(total=len(split_images), desc=f"   Procesando", 
                  unit="img", ncols=80) as pbar:
            
            for i, img_path in enumerate(split_images):
                try:
                    # Leer imagen grayscale
                    img = Image.open(img_path)
                    
                    # Convertir a RGB (replicar canal grayscale)
                    img_rgb = img.convert('RGB')
                    
                    # Redimensionar a 256x256 (modelo espera 256x256)
                    img_resized = img_rgb.resize((256, 256), Image.LANCZOS)
                    
                    # Guardar como PNG
                    output_file = output_path / split_name / f"{img_path.stem}.png"
                    img_resized.save(output_file, 'PNG')
                    
                    total_processed += 1
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\n   ⚠️  Error procesando {img_path.name}: {e}")
                    continue
    
    print(f"\n" + "=" * 70)
    print(f"✅ DATASET PREPARADO EXITOSAMENTE")
    print(f"=" * 70)
    print(f"Ubicación: {output_path.absolute()}")
    print(f"Total procesadas: {total_processed}/{total} imágenes")
    print(f"\nEstructura:")
    for split in ['train', 'val', 'test']:
        count = len(list((output_path / split).glob('*.png')))
        print(f"  - {split}/: {count} imágenes")
    print(f"\n🚀 Listo para entrenar con:")
    print(f"   python train.py --config configs/bossbase_config.yaml --dataset bossbase")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preparar BOSSBase dataset')
    parser.add_argument('--source', type=str, default='/workspace/BOSSbase',
                        help='Directorio con imágenes PGM originales')
    parser.add_argument('--output', type=str, default='/workspace/BOSSbase_prepared',
                        help='Directorio de salida para dataset preparado')
    parser.add_argument('--split', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Ratios train/val/test (ej: 0.8 0.1 0.1)')
    
    args = parser.parse_args()
    
    # Validar ratios
    if sum(args.split) != 1.0:
        print(f"❌ ERROR: Los ratios deben sumar 1.0 (actual: {sum(args.split)})")
        exit(1)
    
    prepare_bossbase(
        source_dir=args.source,
        output_dir=args.output,
        split_ratios=tuple(args.split)
    )
