"""
Download ImageNet Script - Descarga ImageNet 2012 Validation Set

Este script descarga el validation set de ImageNet 2012 usado en el paper.
Dataset: 50,000 imágenes RGB de 1,000 categorías
Tamaño: ~6.3 GB

Opciones de descarga:
1. Kaggle API (recomendado - rápido)
2. Manual (si no tienes cuenta Kaggle)

Uso:
    python scripts/download_imagenet.py --method kaggle
    python scripts/download_imagenet.py --method manual
"""

import argparse
import os
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import subprocess


def download_with_progress(url, output_path):
    """Descarga con barra de progreso"""
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_via_kaggle(data_dir):
    """
    Descarga ImageNet desde Kaggle usando Kaggle API
    
    Requisitos:
    1. Cuenta gratuita en Kaggle (https://www.kaggle.com)
    2. API token en ~/.kaggle/kaggle.json
    
    Args:
        data_dir: Directorio donde guardar los datos
    """
    print("="*70)
    print("DESCARGA IMAGENET VÍA KAGGLE API")
    print("="*70)
    
    # Verificar si kaggle está instalado
    try:
        import kaggle
        print("✅ Kaggle API instalada")
    except ImportError:
        print("⚠️  Kaggle API no instalada. Instalando...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
        import kaggle
        print("✅ Kaggle API instalada exitosamente")
    
    # Verificar credenciales
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_config.exists():
        print("\n❌ ERROR: No se encontró kaggle.json")
        print("\nPasos para configurar Kaggle API:")
        print("1. Ir a https://www.kaggle.com/account")
        print("2. Scroll down a 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Guardar kaggle.json en:", kaggle_config.parent)
        print("\nEn Windows PowerShell:")
        print(f"  mkdir {kaggle_config.parent}")
        print(f"  move Downloads\\kaggle.json {kaggle_config}")
        return False
    
    print(f"✅ Credenciales encontradas: {kaggle_config}")
    
    # Crear directorio
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Directorio de descarga: {data_dir}")
    
    # Descargar validation set de ImageNet
    print("\n📥 Descargando ImageNet 2012 validation set (~6.3 GB)...")
    print("⏳ Esto puede tomar 30-60 minutos dependiendo de tu conexión...")
    
    try:
        # Descargar usando Kaggle API
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        # Dataset: imagenet-object-localization-challenge
        # Archivo: ILSVRC/Data/CLS-LOC/val (validation images)
        print("\nDescargando dataset...")
        api.competition_download_files(
            'imagenet-object-localization-challenge',
            path=str(data_dir)
        )
        
        print("\n✅ Descarga completada!")
        
        # Extraer archivos
        print("\n📦 Extrayendo archivos...")
        zip_file = data_dir / "imagenet-object-localization-challenge.zip"
        
        if zip_file.exists():
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("✅ Extracción completada")
            
            # Limpiar archivo zip
            print("\n🗑️  Eliminando archivo zip...")
            zip_file.unlink()
        
        # Buscar el directorio de validación
        val_dir = data_dir / "ILSVRC" / "Data" / "CLS-LOC" / "val"
        if val_dir.exists():
            print(f"\n✅ Imágenes de validación en: {val_dir}")
            print(f"   Total imágenes: {len(list(val_dir.glob('*.JPEG')))}")
        else:
            print(f"\n⚠️  Buscando directorio de validación...")
            # Buscar recursivamente
            for path in data_dir.rglob("val"):
                if path.is_dir():
                    print(f"   Encontrado: {path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error durante la descarga: {e}")
        print("\nIntenta:")
        print("1. Verificar que tienes cuenta en Kaggle")
        print("2. Aceptar las reglas de la competencia:")
        print("   https://www.kaggle.com/c/imagenet-object-localization-challenge")
        print("3. Verificar tu API token")
        return False


def download_manual(data_dir):
    """
    Instrucciones para descarga manual de ImageNet
    
    Args:
        data_dir: Directorio donde guardar los datos
    """
    print("="*70)
    print("DESCARGA MANUAL DE IMAGENET")
    print("="*70)
    
    print("\nImageNet requiere registro académico para descarga directa.")
    print("Opciones disponibles:\n")
    
    print("📌 OPCIÓN 1: Kaggle (Recomendado)")
    print("  1. Crear cuenta gratuita: https://www.kaggle.com")
    print("  2. Aceptar reglas: https://www.kaggle.com/c/imagenet-object-localization-challenge")
    print("  3. Ejecutar: python scripts/download_imagenet.py --method kaggle\n")
    
    print("📌 OPCIÓN 2: ImageNet Official")
    print("  1. Registrarse: https://image-net.org/signup")
    print("  2. Request access al validation set")
    print("  3. Descargar ILSVRC2012_img_val.tar (~6.3 GB)")
    print(f"  4. Extraer a: {data_dir}/ILSVRC/Data/CLS-LOC/val/\n")
    
    print("📌 OPCIÓN 3: Academic Torrents")
    print("  1. Instalar cliente torrent")
    print("  2. Descargar: http://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5")
    print(f"  3. Extraer a: {data_dir}/ILSVRC/Data/CLS-LOC/val/\n")
    
    print("📌 OPCIÓN 4: Subset Reducido (Testing)")
    print("  Si solo quieres probar rápidamente:")
    print("  - Tiny ImageNet: https://www.kaggle.com/c/tiny-imagenet")
    print("  - Subset 1K imágenes (crear manualmente)\n")
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Directorio destino: {data_dir}")
    print(f"\n💡 Después de descargar manualmente, las imágenes deben estar en:")
    print(f"   {data_dir}/ILSVRC/Data/CLS-LOC/val/*.JPEG")


def organize_imagenet(data_dir):
    """
    Organiza ImageNet en estructura compatible con PyTorch ImageFolder
    
    ImageNet validation viene en formato plano (50K imágenes en un directorio).
    PyTorch ImageFolder requiere subdirectorios por clase.
    
    Args:
        data_dir: Directorio con imágenes de ImageNet
    """
    print("="*70)
    print("ORGANIZANDO IMAGENET PARA PYTORCH")
    print("="*70)
    
    data_dir = Path(data_dir)
    val_dir = data_dir / "ILSVRC" / "Data" / "CLS-LOC" / "val"
    
    if not val_dir.exists():
        print(f"❌ No se encontró: {val_dir}")
        return False
    
    # Contar imágenes
    images = list(val_dir.glob("*.JPEG"))
    print(f"\n📊 Encontradas {len(images)} imágenes")
    
    if len(images) == 0:
        print("❌ No hay imágenes .JPEG en el directorio")
        return False
    
    # Para esteganografía no necesitamos labels específicos
    # Solo necesitamos que estén en subdirectorios (PyTorch ImageFolder requirement)
    
    # Opción 1: Crear un solo directorio 'all' con todas las imágenes
    organized_dir = data_dir / "organized" / "all"
    organized_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Organizando en: {organized_dir}")
    print("⏳ Creando symlinks...")
    
    for img in tqdm(images[:50000], desc="Organizando"):  # Primeras 50K
        # Crear symlink en lugar de copiar (ahorra espacio)
        target = organized_dir / img.name
        if not target.exists():
            try:
                # En Windows requiere privilegios de admin para symlinks
                # Alternativa: copiar archivo
                import shutil
                shutil.copy2(img, target)
            except:
                # Si falla, copiar archivo
                import shutil
                shutil.copy2(img, target)
    
    print(f"\n✅ Imágenes organizadas en: {organized_dir}")
    print(f"   Total: {len(list(organized_dir.glob('*.JPEG')))} imágenes")
    
    return True


def verify_dataset(data_dir):
    """Verifica que el dataset esté correctamente descargado"""
    print("\n" + "="*70)
    print("VERIFICANDO DATASET")
    print("="*70)
    
    data_dir = Path(data_dir)
    
    # Buscar imágenes
    organized_dir = data_dir / "organized" / "all"
    if organized_dir.exists():
        images = list(organized_dir.glob("*.JPEG"))
        print(f"\n✅ Encontradas {len(images)} imágenes en {organized_dir}")
        
        if len(images) >= 50000:
            print("✅ Dataset completo (50,000 imágenes)")
            return True
        elif len(images) >= 10000:
            print(f"⚠️  Dataset parcial ({len(images)} imágenes)")
            print("   Suficiente para testing, pero no para replicar paper exactamente")
            return True
        else:
            print(f"❌ Dataset incompleto ({len(images)} imágenes)")
            return False
    else:
        print(f"❌ No se encontró directorio organizado: {organized_dir}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Descarga ImageNet 2012 validation set para DCT-GAN'
    )
    parser.add_argument(
        '--method',
        choices=['kaggle', 'manual'],
        default='kaggle',
        help='Método de descarga (default: kaggle)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/imagenet2012',
        help='Directorio para guardar datos (default: data/imagenet2012)'
    )
    parser.add_argument(
        '--organize',
        action='store_true',
        help='Solo organizar imágenes existentes (sin descargar)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Solo verificar dataset existente'
    )
    
    args = parser.parse_args()
    
    if args.verify:
        success = verify_dataset(args.data_dir)
        sys.exit(0 if success else 1)
    
    if args.organize:
        success = organize_imagenet(args.data_dir)
        sys.exit(0 if success else 1)
    
    # Descarga
    if args.method == 'kaggle':
        success = download_via_kaggle(args.data_dir)
    else:
        download_manual(args.data_dir)
        success = False  # Manual no descarga automáticamente
    
    if success:
        # Organizar dataset
        organize_imagenet(args.data_dir)
        
        # Verificar
        verify_dataset(args.data_dir)
        
        print("\n" + "="*70)
        print("✅ IMAGENET LISTO PARA USAR")
        print("="*70)
        print("\nPróximo paso:")
        print("  python train.py --config configs/base_config.yaml")
        print("\nDataset ubicado en:")
        print(f"  {Path(args.data_dir).absolute()}/organized/all/")
    else:
        print("\n" + "="*70)
        print("⚠️  DESCARGA MANUAL REQUERIDA")
        print("="*70)
        print("\nSigue las instrucciones arriba para descargar ImageNet manualmente")


if __name__ == "__main__":
    main()
