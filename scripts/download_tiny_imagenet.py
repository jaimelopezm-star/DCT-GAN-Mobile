"""
Descarga Tiny ImageNet - Dataset pequeño para pruebas (200 MB)
Funciona igual que ImageNet pero es 30x más pequeño
"""

import urllib.request
import zipfile
import shutil
from pathlib import Path

print("="*70)
print("DESCARGANDO TINY IMAGENET (200 MB)")
print("="*70)

# Crear directorio
data_dir = Path("data/imagenet2012")
data_dir.mkdir(parents=True, exist_ok=True)

# URL de Tiny ImageNet
tiny_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
tiny_zip = data_dir / "tiny-imagenet-200.zip"

print(f"\n📥 Descargando desde Stanford...")
print(f"   URL: {tiny_url}")
print(f"   Tamaño: ~200 MB\n")

# Descargar con progreso
def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(downloaded * 100 / total_size, 100)
    mb_downloaded = downloaded / 1024 / 1024
    mb_total = total_size / 1024 / 1024
    print(f"\r   Progreso: {percent:5.1f}% [{mb_downloaded:6.1f} MB / {mb_total:6.1f} MB]", end='', flush=True)

try:
    urllib.request.urlretrieve(tiny_url, tiny_zip, reporthook=show_progress)
    print("\n\n✅ Descarga completada!")
except Exception as e:
    print(f"\n\n❌ Error: {e}")
    exit(1)

# Extraer
print("\n📦 Extrayendo archivos...")
with zipfile.ZipFile(tiny_zip, 'r') as zip_ref:
    zip_ref.extractall(data_dir)
print("✅ Extracción completada")

# Limpiar
tiny_zip.unlink()
print("🗑️  Archivo ZIP eliminado")

# Organizar para PyTorch
tiny_dir = data_dir / "tiny-imagenet-200"
val_src = tiny_dir / "val" / "images"
organized_dir = data_dir / "organized" / "all"
organized_dir.mkdir(parents=True, exist_ok=True)

if val_src.exists():
    print("\n📁 Organizando imágenes para PyTorch...")
    count = 0
    for img in val_src.glob("*.JPEG"):
        shutil.copy2(img, organized_dir / img.name)
        count += 1
    
    print(f"✅ {count} imágenes organizadas")
    print(f"\n📍 Ubicación: {organized_dir.absolute()}")
    
    print("\n" + "="*70)
    print("✅ DATASET LISTO PARA ENTRENAR")
    print("="*70)
    print(f"\nEjecuta:")
    print(f"  python train.py --config configs/config.yaml --dataset imagenet")
else:
    print(f"\n⚠️ No se encontró: {val_src}")
