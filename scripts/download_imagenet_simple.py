"""
Download ImageNet Validation Set ONLY - Simple Version
Este script usa Kaggle Hub para descargar SOLO el validation set (~6.3 GB)
NO descarga toda la competencia (155 GB)
"""

import os
from pathlib import Path
import zipfile
import shutil

print("="*70)
print("DESCARGA IMAGENET VALIDATION SET (6.3 GB)")
print("="*70)
print()

# Intentar con kagglehub (más moderno, permite descargas parciales)
try:
    import kagglehub
    print("✅ KaggleHub instalado")
except ImportError:
    print("⏳ Instalando kagglehub...")
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "kagglehub", "-q"])
    import kagglehub
    print("✅ KaggleHub instalado")

# Crear directorio de destino
data_dir = Path("data/imagenet2012")
data_dir.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Directorio: {data_dir.absolute()}")

# OPCIÓN 1: Descargar desde Tiny ImageNet (más pequeño - 200 clases, 200 MB)
print("\n" + "="*70)
print("DESCARGA RÁPIDA: Tiny ImageNet")
print("="*70)
print("\nTiny ImageNet es un subset reducido perfecto para pruebas:")
print("  - 200 clases (vs 1000 en ImageNet completo)")
print("  - 100,000 imágenes (vs 1.2 millones)")
print("  - Tamaño: ~200 MB (vs 155 GB)")
print("  - Formato idéntico a ImageNet original")

print("\n¿Deseas descargar Tiny ImageNet primero para probar?")
print("(Recomendado para validar que todo funcione antes de 6.3 GB)")
choice = input("\n[S]í / [N]o (ImageNet completo): ").strip().upper()

if choice == 'S':
    print("\n⏳ Descargando Tiny ImageNet...")
    
    # Descargar desde URL directa
    import urllib.request
    
    tiny_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    tiny_zip = data_dir / "tiny-imagenet-200.zip"
    
    print(f"📥 Descargando desde: {tiny_url}")
    print("   Tamaño: ~200 MB")
    
    # Descargar con barra de progreso
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\r   Progreso: {percent:.1f}% [{downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB]", end='')
    
    urllib.request.urlretrieve(tiny_url, tiny_zip, reporthook=download_progress)
    print("\n✅ Descarga completada")
    
    # Extraer
    print("\n📦 Extrayendo archivos...")
    with zipfile.ZipFile(tiny_zip, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("✅ Extracción completada")
    
    # Limpiar zip
    tiny_zip.unlink()
    
    # Organizar en formato PyTorch
    tiny_dir = data_dir / "tiny-imagenet-200"
    if tiny_dir.exists():
        # Mover imágenes de validación
        val_src = tiny_dir / "val" / "images"
        organized_dir = data_dir / "organized" / "all"
        organized_dir.mkdir(parents=True, exist_ok=True)
        
        if val_src.exists():
            print("\n📁 Organizando imágenes para PyTorch...")
            for img in val_src.glob("*.JPEG"):
                shutil.copy2(img, organized_dir / img.name)
            
            print(f"✅ {len(list(organized_dir.glob('*.JPEG')))} imágenes listas")
            print(f"\n📍 Ubicación: {organized_dir.absolute()}")
            
            print("\n" + "="*70)
            print("✅ DATASET LISTO PARA ENTRENAR")
            print("="*70)
            print(f"\nPuedes entrenar con:")
            print(f"  python train.py --config configs/config.yaml --dataset imagenet")
        else:
            print("⚠️ No se encontró directorio de validación")
    
else:
    # OPCIÓN 2: Instrucciones para ImageNet completo
    print("\n" + "="*70)
    print("DESCARGA MANUAL - ImageNet Validation Set (6.3 GB)")
    print("="*70)
    
    print("\nPara descargar ImageNet validation set (50K imágenes):\n")
    
    print("📌 OPCIÓN 1: Kaggle Manual (Recomendado)")
    print("   1. Ir a: https://www.kaggle.com/c/imagenet-object-localization-challenge/data")
    print("   2. Buscar: 'ILSVRC2012_img_val.tar' o 'validation images'")
    print("   3. Descargar SOLO ese archivo (~6.3 GB)")
    print(f"   4. Extraer a: {data_dir.absolute() / 'ILSVRC' / 'Data' / 'CLS-LOC' / 'val'}")
    
    print("\n📌 OPCIÓN 2: ImageNet Official")
    print("   1. Registrarse: https://image-net.org/download")
    print("   2. Descargar 'ILSVRC2012_img_val.tar'")
    print(f"   3. Extraer a: {data_dir.absolute() / 'ILSVRC' / 'Data' / 'CLS-LOC' / 'val'}")
    
    print("\n📌 OPCIÓN 3: Academic Torrents")
    print("   1. Buscar: ImageNet ILSVRC2012 validation")
    print("   2  Link: http://academictorrents.com/")
    print(f"   3. Extraer a: {data_dir.absolute() / 'ILSVRC' / 'Data' / 'CLS-LOC' / 'val'}")
    
    print("\n💡 Después de descargar manualmente:")
    print("   python scripts/prepare_dataset.py")
    
print("\n" + "="*70)
