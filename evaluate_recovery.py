"""
Script para evaluar la calidad de recuperación del secreto

Métricas a evaluar:
1. PSNR entre secret_original y secret_recuperado
2. SSIM entre secret_original y secret_recuperado
3. Visualización de ejemplos
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.dctgan import DCTGAN
from src.data.dataset import StegoDataset
from src.training.metrics import calculate_ssim, calculate_rmse
from src.training.losses import calculate_psnr
import yaml
from pathlib import Path
import numpy as np


def evaluate_secret_recovery(checkpoint_path: str, config_path: str, num_samples: int = 50):
    """
    Evalúa qué tan bien el decoder recupera el secreto
    """
    print("="*60)
    print("EVALUACIÓN DE RECUPERACIÓN DEL SECRETO")
    print("="*60)
    
    # Cargar config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Crear modelo
    model = DCTGAN(config['model'])
    
    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Checkpoint cargado: {checkpoint_path}")
    
    # Crear dataset de prueba
    dataset = StegoDataset(
        dataset_type='synthetic',
        split='test',
        size=num_samples,
        image_size=config['data'].get('image_size', 256)
    )
    
    # Métricas
    psnr_cover_stego = []  # PSNR entre cover y stego (calidad visual)
    psnr_secret_recovered = []  # PSNR entre secret original y recuperado
    ssim_cover_stego = []
    ssim_secret_recovered = []
    
    print(f"\nEvaluando {num_samples} muestras...")
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            cover = sample['cover'].unsqueeze(0).to(device)
            secret = sample['secret'].unsqueeze(0).to(device)
            
            # Forward pass
            stego, recovered_secret = model(cover, secret, mode='full')
            
            # Calcular métricas cover vs stego
            psnr_cs = calculate_psnr(cover, stego).item()
            ssim_cs = calculate_ssim(cover, stego).item()
            psnr_cover_stego.append(psnr_cs)
            ssim_cover_stego.append(ssim_cs)
            
            # Calcular métricas secret original vs recuperado
            psnr_sr = calculate_psnr(secret, recovered_secret).item()
            ssim_sr = calculate_ssim(secret, recovered_secret).item()
            psnr_secret_recovered.append(psnr_sr)
            ssim_secret_recovered.append(ssim_sr)
            
            if (i + 1) % 10 == 0:
                print(f"  Procesadas {i+1}/{num_samples} muestras")
    
    # Calcular promedios
    avg_psnr_cs = np.mean(psnr_cover_stego)
    avg_ssim_cs = np.mean(ssim_cover_stego)
    avg_psnr_sr = np.mean(psnr_secret_recovered)
    avg_ssim_sr = np.mean(ssim_secret_recovered)
    
    print("\n" + "="*60)
    print("RESULTADOS")
    print("="*60)
    
    print("\n📊 CALIDAD VISUAL (Cover vs Stego):")
    print(f"   PSNR: {avg_psnr_cs:.2f} dB (target: 58.27 dB)")
    print(f"   SSIM: {avg_ssim_cs:.4f} (target: 0.942)")
    
    print("\n🔓 RECUPERACIÓN DEL SECRETO (Original vs Recuperado):")
    print(f"   PSNR: {avg_psnr_sr:.2f} dB")
    print(f"   SSIM: {avg_ssim_sr:.4f}")
    
    print("\n" + "="*60)
    print("EVALUACIÓN:")
    print("="*60)
    
    # Evaluar resultados
    visual_ok = avg_psnr_cs >= 40 and avg_ssim_cs >= 0.95
    recovery_ok = avg_psnr_sr >= 20 and avg_ssim_sr >= 0.8
    
    if visual_ok:
        print("✅ Calidad visual: EXCELENTE")
    else:
        print("⚠️  Calidad visual: NECESITA MEJORA")
    
    if recovery_ok:
        print("✅ Recuperación del secreto: EXITOSA")
    else:
        print("❌ Recuperación del secreto: FALLIDA")
        print("   → El residual_scale puede ser demasiado pequeño")
        print("   → El decoder no puede extraer suficiente información")
    
    return {
        'psnr_cover_stego': avg_psnr_cs,
        'ssim_cover_stego': avg_ssim_cs,
        'psnr_secret_recovered': avg_psnr_sr,
        'ssim_secret_recovered': avg_ssim_sr
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--config', type=str, default='configs/high_psnr.yaml')
    parser.add_argument('--samples', type=int, default=50)
    args = parser.parse_args()
    
    results = evaluate_secret_recovery(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_samples=args.samples
    )
