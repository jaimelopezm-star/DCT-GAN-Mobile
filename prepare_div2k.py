#!/usr/bin/env python3
"""
Preparación de dataset DIV2K para DCT-GAN con estructura ImageFolder

Convierte DIV2K descargado a estructura compatible:
  DIV2K_prepared/
    train/
      images/
        0001.png
        0002.png
        ...
    val/
      images/
        0001.png
        ...

Uso:
    python prepare_div2k.py
    
    O con rutas personalizadas:
    python prepare_div2k.py --train-dir /path/to/DIV2K_train_HR \
                            --val-dir /path/to/DIV2K_valid_HR \
                            --output-dir /path/to/DIV2K_prepared
"""

import os
import argparse
import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def prepare_split(source_dir, output_dir, split_name, img_size=256, max_images=None):
    """
    Prepara un split del dataset
    
    Args:
        source_dir: Directorio con imágenes originales
        output_dir: Directorio de salida
        split_name: 'train' o 'val'
        img_size: Tamaño objetivo (256x256)
        max_images: Límite de imágenes (None = todas)
    """
    source_path = Path(source_dir)
    
    # Crear directorio de salida con subdirectorio "images" (para ImageFolder)
    output_path = Path(output_dir) / split_name / "images"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Obtener todas las imágenes
    images = sorted(list(source_path.glob('*.png')))
    
    if max_images:
        images = images[:max_images]
    
    print(f"\n🔄 Procesando {split_name}: {len(images)} imágenes")
    
    processed = 0
    errors = 0
    
    for img_path in tqdm(images, desc=f"  {split_name}"):
        try:
            # Cargar imagen
            img = Image.open(img_path).convert('RGB')
            
            # Redimensionar manteniendo aspect ratio
            img.thumbnail((img_size, img_size), Image.LANCZOS)
            
            # Crop central a tamaño exacto
            width, height = img.size
            left = (width - img_size) // 2
            top = (height - img_size) // 2
            img_cropped = img.crop((left, top, left + img_size, top + img_size))
            
            # Guardar con nombre original
            output_file = output_path / img_path.name
            img_cropped.save(output_file, 'PNG', optimize=True)
            
            processed += 1
            
        except Exception as e:
            print(f"\n❌ Error procesando {img_path.name}: {e}")
            errors += 1
    
    print(f"   ✅ Procesadas: {processed}")
    if errors > 0:
        print(f"   ⚠️  Errores: {errors}")
    
    return processed, errors


def verify_structure(output_dir):
    """Verifica la estructura del dataset preparado"""
    output_path = Path(output_dir)
    
    print("\n🔍 Verificando estructura del dataset...")
    
    for split in ['train', 'val']:
        split_path = output_path / split / "images"
        
        if not split_path.exists():
            print(f"   ❌ {split}/images/ no existe")
            return False
        
        images = list(split_path.glob('*.png'))
        print(f"   ✅ {split}/images/: {len(images)} imágenes")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Preparar DIV2K para DCT-GAN')
    parser.add_argument('--train-dir', type=str, 
                       default='/workspace/DIV2K_download/DIV2K_train_HR',
                       help='Directorio con imágenes de training')
    parser.add_argument('--val-dir', type=str, 
                       default='/workspace/DIV2K_download/DIV2K_valid_HR',
                       help='Directorio con imágenes de validación')
    parser.add_argument('--output-dir', type=str, 
                       default='/workspace/DIV2K_prepared',
                       help='Directorio de salida')
    parser.add_argument('--img-size', type=int, default=256,
                       help='Tamaño de imagen (default: 256)')
    parser.add_argument('--max-train', type=int, default=None,
                       help='Máximo de imágenes de train (default: todas)')
    parser.add_argument('--max-val', type=int, default=None,
                       help='Máximo de imágenes de val (default: todas)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PREPARACIÓN DE DIV2K DATASET")
    print("=" * 60)
    print(f"\n📂 Configuración:")
    print(f"   Training dir: {args.train_dir}")
    print(f"   Validation dir: {args.val_dir}")
    print(f"   Output dir: {args.output_dir}")
    print(f"   Image size: {args.img_size}x{args.img_size}")
    
    # Verificar directorios de entrada
    if not Path(args.train_dir).exists():
        print(f"\n❌ Error: {args.train_dir} no existe")
        print("\n💡 Ejecuta primero:")
        print("   bash download_div2k.sh")
        return
    
    if not Path(args.val_dir).exists():
        print(f"\n❌ Error: {args.val_dir} no existe")
        print("\n💡 Ejecuta primero:")
        print("   bash download_div2k.sh")
        return
    
    # Procesar training set
    train_processed, train_errors = prepare_split(
        args.train_dir, 
        args.output_dir, 
        'train',
        args.img_size,
        args.max_train
    )
    
    # Procesar validation set
    val_processed, val_errors = prepare_split(
        args.val_dir, 
        args.output_dir, 
        'val',
        args.img_size,
        args.max_val
    )
    
    # Verificar estructura
    if not verify_structure(args.output_dir):
        print("\n❌ Error en la estructura del dataset")
        return
    
    # Resumen
    print("\n" + "=" * 60)
    print("✅ PREPARACIÓN COMPLETADA")
    print("=" * 60)
    print(f"\n📊 Resumen:")
    print(f"   Train: {train_processed} imágenes")
    print(f"   Val: {val_processed} imágenes")
    print(f"   Total: {train_processed + val_processed} imágenes")
    
    if (train_errors + val_errors) > 0:
        print(f"\n⚠️  Errores totales: {train_errors + val_errors}")
    
    print(f"\n📁 Dataset preparado en: {args.output_dir}")
    print("\n📝 Siguiente paso:")
    print(f"   python train.py --config configs/exp16_real_dataset.yaml \\")
    print(f"                   --dataset imagenet \\")
    print(f"                   --dataset-path {args.output_dir}")
    print()


if __name__ == '__main__':
    main()
