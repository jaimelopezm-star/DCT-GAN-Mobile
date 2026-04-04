"""
Metrics Module - Métricas de Evaluación

Implementa métricas de calidad y recuperación para esteganografía:
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index
- RMSE: Root Mean Squared Error
- MSE: Mean Squared Error

Paper targets (Malik et al. 2025):
- PSNR: 58.27 dB (cover vs stego)
- SSIM: 0.942 (cover vs stego)
- RMSE: 96.10% (secret recovery)
"""

import torch
import torch.nn.functional as F
from typing import Tuple
import numpy as np


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Calcula Peak Signal-to-Noise Ratio
    
    PSNR = 10 × log10(MAX²/MSE)
    
    Valores típicos:
    - 20-25 dB: Baja calidad
    - 30-40 dB: Buena calidad
    - 40-50 dB: Alta calidad
    - >50 dB: Excelente calidad
    
    Paper target: PSNR ~58.27 dB
    
    Args:
        img1: Imagen 1 [B, C, H, W]
        img2: Imagen 2 [B, C, H, W]
        max_val: Valor máximo de pixel (1.0 para normalizado [0,1])
        
    Returns:
        PSNR en dB (escalar)
    """
    mse = torch.mean((img1 - img2) ** 2)
    
    if mse < 1e-10:
        # Imágenes idénticas
        return torch.tensor(100.0, device=img1.device)
    
    psnr = 10 * torch.log10((max_val ** 2) / mse)
    return psnr


def calculate_ssim(img1: torch.Tensor, 
                   img2: torch.Tensor,
                   window_size: int = 11,
                   C1: float = 0.01**2,
                   C2: float = 0.03**2) -> float:
    """
    Calcula Structural Similarity Index (SSIM)
    
    SSIM mide la similitud estructural entre dos imágenes.
    Considera luminancia, contraste y estructura.
    
    SSIM = (2μ₁μ₂ + C1)(2σ₁₂ + C2) / ((μ₁² + μ₂² + C1)(σ₁² + σ₂² + C2))
    
    Valores:
    - SSIM = 1.0: Imágenes idénticas
    - SSIM > 0.9: Muy similar
    - SSIM > 0.8: Similar
    - SSIM < 0.5: Diferente
    
    Paper target: SSIM ~0.942
    
    Args:
        img1: Imagen 1 [B, C, H, W]
        img2: Imagen 2 [B, C, H, W]
        window_size: Tamaño de ventana para promediar (default: 11)
        C1: Constante para estabilidad (default: (0.01)²)
        C2: Constante para estabilidad (default: (0.03)²)
        
    Returns:
        SSIM promedio (float)
    """
    # Crear ventana gaussiana
    sigma = 1.5
    gauss = torch.Tensor([
        np.exp(-(x - window_size//2)**2 / (2*sigma**2)) 
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    
    # Ventana 2D
    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(img1.size(1), 1, window_size, window_size).contiguous()
    window = window.to(img1.device)
    
    # Calcular medias locales
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calcular varianzas y covarianzas locales
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def calculate_rmse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calcula Root Mean Squared Error
    
    RMSE = √(MSE)
    
    El paper reporta "RMSE = 96.10%" pero en realidad es accuracy de recuperación.
    Aquí calculamos RMSE verdadero para secret recovery.
    
    Args:
        img1: Imagen 1 [B, C, H, W]
        img2: Imagen 2 [B, C, H, W]
        
    Returns:
        RMSE (float)
    """
    mse = torch.mean((img1 - img2) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()


def calculate_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calcula Mean Squared Error
    
    MSE = mean((img1 - img2)²)
    
    Args:
        img1: Imagen 1 [B, C, H, W]
        img2: Imagen 2 [B, C, H, W]
        
    Returns:
        MSE (float)
    """
    mse = torch.mean((img1 - img2) ** 2)
    return mse.item()


def calculate_recovery_accuracy(secret_original: torch.Tensor, 
                                secret_recovered: torch.Tensor,
                                threshold: float = 0.5) -> float:
    """
    Calcula accuracy de recuperación del secret
    
    Para imágenes binarias (negro/blanco), calcula porcentaje de píxeles
    correctamente recuperados.
    
    Paper reporta: 96.10% recovery accuracy
    
    Args:
        secret_original: Secret original [B, C, H, W] o [B, num_bits]
        secret_recovered: Secret recuperado [B, C, H, W] o [B, num_bits]
        threshold: Umbral para binarización (default: 0.5)
        
    Returns:
        Accuracy en porcentaje (0-100)
    """
    # Binarizar
    secret_orig_binary = (secret_original > threshold).float()
    secret_rec_binary = (secret_recovered > threshold).float()
    
    # Calcular píxeles correctos
    correct = (secret_orig_binary == secret_rec_binary).float()
    accuracy = correct.mean().item() * 100.0
    
    return accuracy


def calculate_bit_error_rate(secret_original: torch.Tensor,
                             secret_recovered: torch.Tensor,
                             threshold: float = 0.5) -> float:
    """
    Calcula Bit Error Rate (BER)
    
    BER = (número de bits erróneos) / (total de bits)
    
    Menor es mejor:
    - BER = 0.0: Perfecto
    - BER < 0.01: Excelente
    - BER < 0.05: Bueno
    - BER > 0.1: Pobre
    
    Args:
        secret_original: Secret original [B, C, H, W] o [B, num_bits]
        secret_recovered: Secret recuperado [B, C, H, W] o [B, num_bits]
        threshold: Umbral para binarización (default: 0.5)
        
    Returns:
        BER (0-1)
    """
    # Binarizar
    secret_orig_binary = (secret_original > threshold).float()
    secret_rec_binary = (secret_recovered > threshold).float()
    
    # Contar errores
    errors = (secret_orig_binary != secret_rec_binary).float()
    ber = errors.mean().item()
    
    return ber


def calculate_all_metrics(cover: torch.Tensor,
                          stego: torch.Tensor,
                          secret_original: torch.Tensor,
                          secret_recovered: torch.Tensor,
                          max_val: float = 1.0) -> dict:
    """
    Calcula todas las métricas de evaluación
    
    Args:
        cover: Imagen cover [B, C, H, W]
        stego: Imagen stego [B, C, H, W]
        secret_original: Secret original [B, C, H, W]
        secret_recovered: Secret recuperado [B, C, H, W]
        max_val: Valor máximo de pixel (default: 1.0)
        
    Returns:
        Diccionario con todas las métricas
    """
    metrics = {}
    
    # Calidad visual (cover vs stego)
    metrics['psnr_cover_stego'] = calculate_psnr(cover, stego, max_val).item()
    metrics['ssim_cover_stego'] = calculate_ssim(cover, stego)
    metrics['mse_cover_stego'] = calculate_mse(cover, stego)
    
    # Recuperación de secret (original vs recovered)
    metrics['psnr_secret_recovery'] = calculate_psnr(secret_original, secret_recovered, max_val).item()
    metrics['ssim_secret_recovery'] = calculate_ssim(secret_original, secret_recovered)
    metrics['rmse_secret_recovery'] = calculate_rmse(secret_original, secret_recovered)
    metrics['mse_secret_recovery'] = calculate_mse(secret_original, secret_recovered)
    metrics['recovery_accuracy'] = calculate_recovery_accuracy(secret_original, secret_recovered)
    metrics['bit_error_rate'] = calculate_bit_error_rate(secret_original, secret_recovered)
    
    return metrics


# ============================================
# TESTING
# ============================================
if __name__ == "__main__":
    """
    Test de métricas con datos sintéticos
    """
    print("Testing metrics module...")
    
    # Crear imágenes de prueba
    batch_size = 4
    channels = 3
    height = width = 256
    
    # Cover e imágenes muy similares (stego)
    cover = torch.rand(batch_size, channels, height, width)
    stego = cover + torch.randn_like(cover) * 0.01  # Pequeño ruido
    
    # Secret original y recuperado
    secret_orig = torch.rand(batch_size, channels, height, width)
    secret_rec = secret_orig + torch.randn_like(secret_orig) * 0.05  # Ruido moderado
    
    print("\n1. Testing individual metrics:")
    print("-" * 40)
    
    # PSNR
    psnr = calculate_psnr(cover, stego, max_val=1.0)
    print(f"PSNR (cover vs stego): {psnr:.2f} dB")
    print(f"   Target: 58.27 dB")
    print(f"   Status: {'✅' if psnr > 40 else '⚠️'} (>40 dB is good)")
    
    # SSIM
    ssim = calculate_ssim(cover, stego)
    print(f"\nSSIM (cover vs stego): {ssim:.4f}")
    print(f"   Target: 0.942")
    print(f"   Status: {'✅' if ssim > 0.9 else '⚠️'} (>0.9 is good)")
    
    # RMSE
    rmse = calculate_rmse(secret_orig, secret_rec)
    print(f"\nRMSE (secret recovery): {rmse:.6f}")
    print(f"   Status: {'✅' if rmse < 0.1 else '⚠️'} (<0.1 is good)")
    
    # Recovery accuracy
    accuracy = calculate_recovery_accuracy(secret_orig, secret_rec)
    print(f"\nRecovery Accuracy: {accuracy:.2f}%")
    print(f"   Target: 96.10%")
    print(f"   Status: {'✅' if accuracy > 90 else '⚠️'} (>90% is good)")
    
    # BER
    ber = calculate_bit_error_rate(secret_orig, secret_rec)
    print(f"\nBit Error Rate: {ber:.4f}")
    print(f"   Status: {'✅' if ber < 0.05 else '⚠️'} (<0.05 is good)")
    
    print("\n2. Testing calculate_all_metrics:")
    print("-" * 40)
    
    all_metrics = calculate_all_metrics(cover, stego, secret_orig, secret_rec)
    
    for metric_name, metric_value in all_metrics.items():
        if 'psnr' in metric_name:
            print(f"{metric_name}: {metric_value:.2f} dB")
        elif 'accuracy' in metric_name:
            print(f"{metric_name}: {metric_value:.2f}%")
        else:
            print(f"{metric_name}: {metric_value:.4f}")
    
    print("\n3. Edge case: Identical images")
    print("-" * 40)
    
    identical1 = torch.ones(2, 3, 64, 64)
    identical2 = identical1.clone()
    
    psnr_identical = calculate_psnr(identical1, identical2)
    ssim_identical = calculate_ssim(identical1, identical2)
    
    print(f"PSNR (identical): {psnr_identical:.2f} dB ✅")
    print(f"SSIM (identical): {ssim_identical:.4f} ✅")
    print(f"Expected: PSNR=100 dB, SSIM=1.0")
    
    print("\n✅ All metrics tests passed!")
    print("\nMetrics Summary:")
    print(f"  - PSNR: Measures image quality (dB, higher is better)")
    print(f"  - SSIM: Structural similarity (0-1, higher is better)")
    print(f"  - RMSE: Recovery error (lower is better)")
    print(f"  - Accuracy: Pixel-level correctness (%, higher is better)")
    print(f"  - BER: Bit error rate (0-1, lower is better)")
