"""
Quick Setup Script - Descarga + Testing en Paralelo

Este script:
1. 🔄 Inicia descarga de ImageNet en BACKGROUND
2. ✅ Ejecuta testing rápido con datos sintéticos (validar código)
3. 📊 Muestra progreso de ambos procesos
4. 📦 Prepara splits cuando descarga termine

Uso:
    python quick_setup.py
    python quick_setup.py --skip-download  (solo testing)
    python quick_setup.py --skip-test      (solo descarga)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
import threading
import os


class Colors:
    """Colores para output en consola"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Imprime header colorido"""
    print("\n" + "="*70)
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.ENDC}")
    print("="*70 + "\n")


def print_success(text):
    """Imprime mensaje de éxito"""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")


def print_warning(text):
    """Imprime warning"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")


def print_error(text):
    """Imprime error"""
    print(f"{Colors.RED}❌ {text}{Colors.ENDC}")


def print_info(text):
    """Imprime info"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.ENDC}")


def check_kaggle_setup():
    """Verifica si Kaggle está configurado"""
    print_info("Verificando configuración de Kaggle...")
    
    # Verificar si kaggle está instalado
    try:
        import kaggle
        print_success("Kaggle API instalada")
    except ImportError:
        print_warning("Kaggle API no instalada. Instalando...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle", "-q"], check=True)
        print_success("Kaggle API instalada exitosamente")
    
    # Verificar credenciales
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_config.exists():
        print_success(f"Credenciales encontradas: {kaggle_config}")
        return True
    else:
        print_error("No se encontró kaggle.json")
        print("\nPasos para configurar Kaggle:")
        print("1. Ir a https://www.kaggle.com/account")
        print("2. Scroll a 'API' → 'Create New API Token'")
        print("3. Guardar kaggle.json en:", kaggle_config.parent)
        print("\nEn PowerShell:")
        print(f"  mkdir {kaggle_config.parent}")
        print(f"  move Downloads\\kaggle.json {kaggle_config}")
        return False


def download_imagenet_background():
    """Inicia descarga de ImageNet en background"""
    print_header("🔽 INICIANDO DESCARGA DE IMAGENET (BACKGROUND)")
    
    print_info("Descarga ejecutándose en segundo plano...")
    print_info("Tamaño: ~6.3 GB")
    print_info("Tiempo estimado: 1-2 horas")
    print_info("Puede continuar trabajando mientras se descarga\n")
    
    # Crear archivo de log para la descarga
    log_file = Path("logs") / "download_imagenet.log"
    log_file.parent.mkdir(exist_ok=True)
    
    # Ejecutar en background
    if sys.platform == "win32":
        # Windows: usar subprocess con CREATE_NO_WINDOW
        process = subprocess.Popen(
            [sys.executable, "scripts/download_imagenet.py", "--method", "kaggle"],
            stdout=open(log_file, 'w'),
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
    else:
        # Unix: usar nohup
        process = subprocess.Popen(
            [sys.executable, "scripts/download_imagenet.py", "--method", "kaggle"],
            stdout=open(log_file, 'w'),
            stderr=subprocess.STDOUT
        )
    
    print_success(f"Descarga iniciada (PID: {process.pid})")
    print_info(f"Ver progreso: Get-Content {log_file} -Tail 20 -Wait")
    
    return process


def run_quick_test():
    """Ejecuta test rápido con datos sintéticos"""
    print_header("🧪 TESTING RÁPIDO - Validación de Código")
    
    print_info("Ejecutando 2 epochs con datos sintéticos...")
    print_info("Esto valida que todo el pipeline funciona correctamente")
    print_info("Tiempo estimado: 5-10 minutos\n")
    
    # Crear config temporal para testing rápido
    import yaml
    
    config_path = Path("configs/base_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Modificar para testing rápido
    config['data']['train_size'] = 100  # Solo 100 samples
    config['data']['val_size'] = 20
    config['training']['num_epochs'] = 2  # Solo 2 epochs
    
    # Guardar config temporal
    test_config_path = Path("configs/test_config.yaml")
    with open(test_config_path, 'w') as f:
        yaml.dump(config, f)
    
    print_info("Config de testing creado: configs/test_config.yaml")
    print_info("  - Train samples: 100 (sintéticos)")
    print_info("  - Val samples: 20 (sintéticos)")
    print_info("  - Epochs: 2")
    print_info("  - Objetivo: Validar pipeline completo\n")
    
    # Ejecutar training
    try:
        subprocess.run(
            [
                sys.executable, "train.py",
                "--config", str(test_config_path),
                "--dataset", "synthetic",
                "--checkpoint_dir", "checkpoints_test",
                "--log_dir", "logs_test"
            ],
            check=True
        )
        
        print_success("\n✅ TEST COMPLETADO EXITOSAMENTE!")
        print_success("El código funciona correctamente")
        print_info("Ahora puedes esperar a que termine la descarga de ImageNet")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print_error(f"\n❌ Test falló con código {e.returncode}")
        print_info("Revisa los logs para más detalles")
        return False


def monitor_download(process):
    """Monitorea el progreso de la descarga"""
    log_file = Path("logs") / "download_imagenet.log"
    
    print_header("📊 MONITOREO DE DESCARGA")
    print_info(f"PID del proceso: {process.pid}")
    print_info(f"Log file: {log_file}")
    print_info("Presiona Ctrl+C para dejar de monitorear (la descarga continuará)\n")
    
    try:
        # Leer log en tiempo real
        with open(log_file, 'r') as f:
            # Ir al final
            f.seek(0, 2)
            
            while process.poll() is None:
                line = f.readline()
                if line:
                    print(line.strip())
                else:
                    time.sleep(1)
        
        # Proceso terminó
        if process.returncode == 0:
            print_success("\n✅ DESCARGA COMPLETADA!")
            return True
        else:
            print_error(f"\n❌ Descarga falló (código {process.returncode})")
            return False
            
    except KeyboardInterrupt:
        print_warning("\n⚠️  Monitoreo detenido (descarga continúa en background)")
        print_info(f"Ver progreso: Get-Content {log_file} -Tail 20 -Wait")
        return None


def prepare_splits_if_ready():
    """Prepara splits si ImageNet está descargado"""
    imagenet_dir = Path("data/imagenet2012/organized/all")
    
    if imagenet_dir.exists():
        images = list(imagenet_dir.glob("*.JPEG"))
        
        if len(images) >= 50000:
            print_header("📦 PREPARANDO SPLITS")
            
            print_info("ImageNet descargado completamente")
            print_info(f"Encontradas {len(images)} imágenes")
            print_info("Creando splits 40K/5K/5K...\n")
            
            try:
                subprocess.run(
                    [sys.executable, "scripts/prepare_dataset.py"],
                    check=True
                )
                
                print_success("\n✅ SPLITS CREADOS!")
                print_info("Dataset listo en: data/imagenet2012/splits/")
                return True
                
            except subprocess.CalledProcessError as e:
                print_error(f"\n❌ Error creando splits (código {e.returncode})")
                return False
        else:
            print_warning(f"ImageNet incompleto: {len(images)}/50000 imágenes")
            return False
    else:
        print_warning("ImageNet no encontrado en el directorio esperado")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Quick Setup - Descarga + Testing en Paralelo'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Saltar descarga de ImageNet (solo testing)'
    )
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Saltar testing rápido (solo descarga)'
    )
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Monitorear descarga en tiempo real después del test'
    )
    
    args = parser.parse_args()
    
    print_header("🚀 DCT-GAN QUICK SETUP")
    print("Este script configura todo automáticamente:\n")
    print("1. 🔄 Descarga ImageNet en background (1-2 horas)")
    print("2. ✅ Valida código con testing rápido (5-10 min)")
    print("3. 📦 Prepara splits cuando descarga termine")
    print()
    
    # Variables de estado
    download_process = None
    test_passed = False
    download_completed = False
    
    # PASO 1: Iniciar descarga en background
    if not args.skip_download:
        # Verificar Kaggle
        if not check_kaggle_setup():
            print_error("\nConfigura Kaggle primero y vuelve a ejecutar")
            print_info("O usa: python quick_setup.py --skip-download (solo testing)")
            sys.exit(1)
        
        # Iniciar descarga
        download_process = download_imagenet_background()
        
        # Pequeña pausa para verificar que inició
        time.sleep(2)
        if download_process.poll() is not None:
            print_error("La descarga falló al iniciar")
            print_info("Ejecuta manualmente: python scripts/download_imagenet.py --method kaggle")
            sys.exit(1)
    else:
        print_warning("Descarga saltada (--skip-download)")
    
    # PASO 2: Testing rápido
    if not args.skip_test:
        print_info("\n⏳ Esperando 3 segundos antes de iniciar testing...")
        time.sleep(3)
        
        test_passed = run_quick_test()
        
        if not test_passed:
            print_error("\n⚠️  El testing falló")
            print_info("Revisa los errores antes de continuar")
            
            if download_process:
                print_info(f"\nDescarga continúa en background (PID: {download_process.pid})")
            
            sys.exit(1)
    else:
        print_warning("Testing saltado (--skip-test)")
    
    # PASO 3: Monitorear descarga (opcional)
    if args.monitor and download_process:
        download_completed = monitor_download(download_process)
    
    # PASO 4: Preparar splits si descarga terminó
    if download_completed or download_process is None:
        prepare_splits_if_ready()
    
    # RESUMEN FINAL
    print_header("📋 RESUMEN")
    
    if test_passed:
        print_success("Testing: PASADO ✅")
        print_info("  → El código funciona correctamente")
    
    if download_process and download_process.poll() is None:
        print_info(f"Descarga: EN PROGRESO (PID: {download_process.pid})")
        print_info("  → Monitorear: Get-Content logs/download_imagenet.log -Tail 20 -Wait")
        print_info("  → Cuando termine, ejecuta: python scripts/prepare_dataset.py")
    elif download_completed:
        print_success("Descarga: COMPLETADA ✅")
        
        # Verificar si splits están listos
        if (Path("data/imagenet2012/splits/train/all").exists()):
            print_success("Splits: LISTOS ✅")
            print_info("\n🎉 TODO LISTO PARA ENTRENAMIENTO!")
            print_info("Ejecuta: python train.py --config configs/base_config.yaml --dataset imagenet")
        else:
            print_warning("Splits: PENDIENTE")
            print_info("Ejecuta: python scripts/prepare_dataset.py")
    
    print("\n" + "="*70)
    print(f"{Colors.BOLD}{Colors.GREEN}✨ Quick Setup Completado!{Colors.ENDC}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
