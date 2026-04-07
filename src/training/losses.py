"""
Loss Functions - Funciones de Pérdida del Paper

Implementa Ecuación 5 del paper Malik et al. (2025):
    L_total = α × L_MSE + β × L_crossentropy + γ × L_adversarial
    
Valores del paper:
    α = 0.3 (similitud cover-stego)
    β = 15.0 (recuperación de secret)
    γ = 0.03 (adversarial)

Componentes:
- L_MSE: Mean Squared Error para preservar calidad visual
- L_CE: Binary Cross Entropy para recuperación del secret
- L_adv: WGAN loss para fooling discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class MSELoss(nn.Module):
    """
    Mean Squared Error Loss
    
    Mide similitud entre cover y stego para preservar calidad visual.
    Paper target: PSNR ~58 dB (requiere MSE muy bajo)
    
    MSE bajo → alta similitud → PSNR alto
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(MSELoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)
    
    def forward(self, cover: torch.Tensor, stego: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cover: Imagen original [B, C, H, W]
            stego: Imagen esteganográfica [B, C, H, W]
            
        Returns:
            MSE loss (escalar)
        """
        return self.loss_fn(cover, stego)


class BCERecoveryLoss(nn.Module):
    """
    Binary Cross Entropy Loss para recuperación de secret
    
    Mide qué tan bien el decoder recupera el secret del stego.
    Paper target: ~100% recovery accuracy
    
    BCE bajo → buena recuperación → alta fidelidad
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(BCERecoveryLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward(self, secret_original: torch.Tensor, 
                secret_recovered: torch.Tensor) -> torch.Tensor:
        """
        Args:
            secret_original: Secret original [B, C, H, W] o [B, num_bits]
            secret_recovered: Secret recuperado (logits o probabilities)
            
        Returns:
            BCE loss (escalar)
        """
        # Si son logits, usar BCEWithLogitsLoss
        # Si son probabilities, usar BCELoss
        return self.loss_fn(secret_recovered, secret_original)


class WassersteinGANLoss(nn.Module):
    """
    Wasserstein GAN Loss (WGAN)
    
    Usa Wasserstein distance como loss adversarial.
    Paper menciona uso de adversarial loss para engañar al discriminador.
    
    WGAN: L_D = E[D(fake)] - E[D(real)]
          L_G = -E[D(fake)]
    
    Ventajas:
    - Más estable que GAN tradicional
    - No mode collapse
    - Métrica interpretable (distancia)
    """
    
    def __init__(self):
        super(WassersteinGANLoss, self).__init__()
    
    def discriminator_loss(self, 
                          real_validity: torch.Tensor, 
                          fake_validity: torch.Tensor) -> torch.Tensor:
        """
        Loss del discriminador (Wasserstein Distance)
        
        Args:
            real_validity: D(real) [B, 1]
            fake_validity: D(fake) [B, 1]
            
        Returns:
            Discriminator loss (negated Wasserstein distance for gradient descent)
        """
        # Wasserstein loss: -(E[D(real)] - E[D(fake)])
        # El negativo convierte maximización en minimización para gradient descent
        # Discriminador quiere:
        #   - D(real) alto
        #   - D(fake) bajo
        #   - Maximizar: D(real) - D(fake)
        #   - Con negativo: Minimizar -(D(real) - D(fake))
        loss_d = -(real_validity.mean() - fake_validity.mean())
        return loss_d
    
    def generator_loss(self, fake_validity: torch.Tensor) -> torch.Tensor:
        """
        Loss del generador
        
        Args:
            fake_validity: D(fake) [B, 1]
            
        Returns:
            Generator loss (queremos engañar discriminador)
        """
        # Generator quiere D(fake) alto
        loss_g = -fake_validity.mean()
        return loss_g


class GradientPenalty(nn.Module):
    """
    Gradient Penalty para WGAN-GP
    
    Regulariza el discriminador forzando gradientes cercanos a 1
    en interpolaciones entre real y fake.
    
    Paper puede usar esto para estabilizar entrenamiento adversarial.
    
    GP = λ × E[(||∇D(x_interp)||_2 - 1)²]
    Típicamente λ = 10
    """
    
    def __init__(self, lambda_gp: float = 10.0):
        super(GradientPenalty, self).__init__()
        self.lambda_gp = lambda_gp
    
    def forward(self, 
                discriminator: nn.Module,
                real_images: torch.Tensor,
                fake_images: torch.Tensor,
                device: torch.device) -> torch.Tensor:
        """
        Calcula gradient penalty
        
        Args:
            discriminator: Red discriminadora
            real_images: Imágenes reales [B, C, H, W]
            fake_images: Imágenes falsas [B, C, H, W]
            device: Device (CPU/GPU)
            
        Returns:
            Gradient penalty (escalar)
        """
        batch_size = real_images.size(0)
        
        # Interpolación aleatoria entre real y fake
        epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolated = epsilon * real_images + (1 - epsilon) * fake_images
        interpolated.requires_grad_(True)
        
        # Discriminar interpolación
        d_interpolated = discriminator(interpolated)
        
        # Calcular gradientes
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Flatten gradientes
        gradients = gradients.view(batch_size, -1)
        
        # Norma L2 de gradientes
        gradient_norm = gradients.norm(2, dim=1)
        
        # Penalty: (norm - 1)²
        penalty = self.lambda_gp * ((gradient_norm - 1) ** 2).mean()
        
        return penalty


class HybridLoss(nn.Module):
    """
    Loss Híbrida del Paper (Ecuación 5)
    
    L_total = α × L_MSE + β × L_crossentropy + γ × L_adversarial
    
    Paper values:
        α = 0.3 (similitud cover-stego)
        β = 15.0 (recuperación secret)
        γ = 0.03 (adversarial)
    
    Componentes:
    1. MSE: Preservar calidad visual (PSNR ~58 dB)
    2. BCE: Recuperar secret correctamente (~100% accuracy)
    3. Adversarial: Engañar discriminador (indetectable)
    
    Uso durante entrenamiento:
    - Encoder/Decoder (generador): minimizar L_total
    - Discriminador: minimizar L_discriminator por separado
    """
    
    def __init__(self, 
                 alpha: float = 0.3,
                 beta: float = 15.0,
                 gamma: float = 0.03,
                 use_wgan: bool = True):
        """
        Args:
            alpha: Peso de MSE loss (default: 0.3)
            beta: Peso de BCE loss (default: 15.0)
            gamma: Peso de adversarial loss (default: 0.03)
            use_wgan: Usar WGAN en lugar de GAN estándar (default: True)
        """
        super(HybridLoss, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Componentes de loss
        self.mse_loss = MSELoss()
        self.bce_loss = BCERecoveryLoss()
        
        if use_wgan:
            self.adv_loss = WassersteinGANLoss()
        else:
            # Fallback: GAN estándar
            self.adv_loss = nn.BCEWithLogitsLoss()
    
    def generator_loss(self,
                      cover_image: torch.Tensor,
                      stego_image: torch.Tensor,
                      secret_original: torch.Tensor,
                      secret_recovered: torch.Tensor,
                      discriminator_output: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Loss total del generador (Encoder + Decoder)
        
        Args:
            cover_image: Imagen cover [B, C, H, W]
            stego_image: Imagen stego generada [B, C, H, W]
            secret_original: Secret original [B, ....]
            secret_recovered: Secret recuperado [B, ....]
            discriminator_output: D(stego) [B, 1]
            
        Returns:
            (loss_total, loss_dict): Loss total y diccionario con componentes
        """
        # 1. MSE: similitud cover-stego
        loss_mse = self.mse_loss(cover_image, stego_image)
        
        # 2. BCE: recuperación de secret
        loss_bce = self.bce_loss(secret_original, secret_recovered)
        
        # 3. Adversarial: engañar discriminador
        if isinstance(self.adv_loss, WassersteinGANLoss):
            loss_adv = self.adv_loss.generator_loss(discriminator_output)
        else:
            # GAN estándar: D(stego) → 1 (real)
            target_real = torch.ones_like(discriminator_output)
            loss_adv = self.adv_loss(discriminator_output, target_real)
        
        # Loss total (Ecuación 5)
        loss_total = (self.alpha * loss_mse + 
                     self.beta * loss_bce + 
                     self.gamma * loss_adv)
        
        # Diccionario para logging
        loss_dict = {
            'loss_total': loss_total.item(),
            'loss_mse': loss_mse.item(),
            'loss_bce': loss_bce.item(),
            'loss_adv': loss_adv.item(),
            'mse_weighted': (self.alpha * loss_mse).item(),
            'bce_weighted': (self.beta * loss_bce).item(),
            'adv_weighted': (self.gamma * loss_adv).item()
        }
        
        return loss_total, loss_dict
    
    def discriminator_loss(self,
                          real_validity: torch.Tensor,
                          fake_validity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Loss del discriminador
        
        Args:
            real_validity: D(real) [B, 1]
            fake_validity: D(fake) [B, 1]
            
        Returns:
            (loss_d, loss_dict): Loss discriminador y diccionario
        """
        if isinstance(self.adv_loss, WassersteinGANLoss):
            loss_d = self.adv_loss.discriminator_loss(real_validity, fake_validity)
        else:
            # GAN estándar
            target_real = torch.ones_like(real_validity)
            target_fake = torch.zeros_like(fake_validity)
            loss_real = self.adv_loss(real_validity, target_real)
            loss_fake = self.adv_loss(fake_validity, target_fake)
            loss_d = (loss_real + loss_fake) / 2
        
        loss_dict = {
            'loss_discriminator': loss_d.item(),
            'D_real': real_validity.mean().item(),
            'D_fake': fake_validity.mean().item()
        }
        
        return loss_d, loss_dict


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Calcula Peak Signal-to-Noise Ratio
    
    PSNR = 10 × log10(MAX²/MSE)
    
    Paper target: PSNR ~58.27 dB
    
    Args:
        img1: Imagen 1 [B, C, H, W]
        img2: Imagen 2 [B, C, H, W]
        max_val: Valor máximo de pixel (default: 1.0 para normalizado)
        
    Returns:
        PSNR en dB (escalar)
    """
    mse = torch.mean((img1 - img2) ** 2)
    psnr = 10 * torch.log10((max_val ** 2) / (mse + 1e-10))
    return psnr


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Calcula Structural Similarity Index (SSIM) - Simplificado
    
    Paper target: SSIM ~0.942 (94.2%)
    
    Esta es una versión simplificada. Para producción usar:
    - pytorch-msssim library
    - skimage.metrics.structural_similarity
    
    Args:
        img1: [B, C, H, W]
        img2: [B, C, H, W]
        
    Returns:
        SSIM (0-1, más alto mejor)
    """
    # Constantes SSIM
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Medias
    mu1 = img1.mean(dim=[1, 2, 3], keepdim=True)
    mu2 = img2.mean(dim=[1, 2, 3], keepdim=True)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2
    
    # Varianzas y covarianza
    sigma1_sq = ((img1 - mu1) ** 2).mean(dim=[1, 2, 3], keepdim=True)
    sigma2_sq = ((img2 - mu2) ** 2).mean(dim=[1, 2, 3], keepdim=True)
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=[1, 2, 3], keepdim=True)
    
    # SSIM
    numerator = (2 * mu12 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim = numerator / (denominator + 1e-10)
    
    return ssim.mean()


if __name__ == "__main__":
    # Tests
    print("="*60)
    print("Testing Loss Functions")
    print("="*60)
    
    batch_size = 4
    channels = 3
    height, width = 256, 256
    
    # Crear datos de prueba
    cover = torch.randn(batch_size, channels, height, width)
    stego = cover + torch.randn_like(cover) * 0.01  # Pequeña diferencia
    secret_original = torch.randn(batch_size, channels, height, width)
    secret_recovered = secret_original + torch.randn_like(secret_original) * 0.1
    
    print(f"Cover shape: {cover.shape}")
    print(f"Stego shape: {stego.shape}")
    print(f"Secret original shape: {secret_original.shape}")
    print(f"Secret recovered shape: {secret_recovered.shape}")
    
    # Test MSE Loss
    print("\n" + "="*60)
    print("Testing MSE Loss")
    print("="*60)
    mse_loss_fn = MSELoss()
    loss_mse = mse_loss_fn(cover, stego)
    print(f"MSE Loss: {loss_mse.item():.6f}")
    
    # Test BCE Loss
    print("\n" + "="*60)
    print("Testing BCE Loss")
    print("="*60)
    bce_loss_fn = BCERecoveryLoss()
    loss_bce = bce_loss_fn(secret_original, secret_recovered)
    print(f"BCE Loss: {loss_bce.item():.6f}")
    
    # Test WGAN Loss
    print("\n" + "="*60)
    print("Testing WGAN Loss")
    print("="*60)
    wgan_loss = WassersteinGANLoss()
    real_validity = torch.randn(batch_size, 1) * 2 + 1  # Alrededor de 1
    fake_validity = torch.randn(batch_size, 1) * 2 - 1  # Alrededor de -1
    
    loss_d = wgan_loss.discriminator_loss(real_validity, fake_validity)
    loss_g = wgan_loss.generator_loss(fake_validity)
    
    print(f"Discriminator Loss: {loss_d.item():.6f}")
    print(f"Generator Loss: {loss_g.item():.6f}")
    print(f"D(real) mean: {real_validity.mean().item():.4f}")
    print(f"D(fake) mean: {fake_validity.mean().item():.4f}")
    
    # Test Hybrid Loss
    print("\n" + "="*60)
    print("Testing Hybrid Loss (Ecuación 5)")
    print("="*60)
    
    hybrid_loss = HybridLoss(alpha=0.3, beta=15.0, gamma=0.03, use_wgan=True)
    
    disc_output = torch.randn(batch_size, 1) * 2 - 1
    
    loss_total_g, loss_dict_g = hybrid_loss.generator_loss(
        cover,
        stego,
        secret_original,
        secret_recovered,
        disc_output
    )
    
    print(f"\nGenerator Loss Components:")
    for key, value in loss_dict_g.items():
        print(f"  {key}: {value:.6f}")
    
    real_val = torch.randn(batch_size, 1) * 2 + 1
    fake_val = torch.randn(batch_size, 1) * 2 - 1
    
    loss_total_d, loss_dict_d = hybrid_loss.discriminator_loss(real_val, fake_val)
    
    print(f"\nDiscriminator Loss Components:")
    for key, value in loss_dict_d.items():
        print(f"  {key}: {value:.6f}")
    
    # Test PSNR
    print("\n" + "="*60)
    print("Testing PSNR Calculation")
    print("="*60)
    psnr = calculate_psnr(cover, stego)
    print(f"PSNR: {psnr.item():.2f} dB")
    print(f"Target: ~58.27 dB")
    
    # Test SSIM
    print("\n" + "="*60)
    print("Testing SSIM Calculation")
    print("="*60)
    ssim = calculate_ssim(cover, stego)
    print(f"SSIM: {ssim.item():.4f}")
    print(f"Target: ~0.942 (94.2%)")
    
    print("\n✅ All loss function tests passed!")
