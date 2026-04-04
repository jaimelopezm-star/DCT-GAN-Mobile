"""
Trainer para DCT-GAN Mobile
Implementa estrategia 4:1 (4 actualizaciones generator por 1 discriminator)
"""
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging


class Trainer:
    """
    Orquestador de entrenamiento para DCT-GAN
    Implementa ratio 4:1 generator/discriminator del paper
    """
    
    def __init__(self, model, optimizers, schedulers, config, device='cuda'):
        """
        Args:
            model: DCT_GAN instance
            optimizers: dict con 'generator' y 'discriminator'
            schedulers: dict con 'generator' y 'discriminator'
            config: configuración de entrenamiento
            device: 'cuda' o 'cpu'
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Optimizadores
        self.optimizer_g = optimizers['generator']
        self.optimizer_d = optimizers['discriminator']
        
        # Schedulers
        self.scheduler_g = schedulers['generator']
        self.scheduler_d = schedulers['discriminator']
        
        # Pérdidas del modelo
        self.hybrid_loss = model.hybrid_loss
        self.wgan_loss = model.wgan_loss
        
        # Métricas
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        
        # Logger
        self.logger = logging.getLogger('Trainer')
        
    def update_generator(self, cover, secret):
        """
        Actualización del generador (encoder + decoder)
        
        Args:
            cover: imágenes cover [B, 3, H, W]
            secret: imágenes secret [B, 3, H, W]
            
        Returns:
            dict con pérdidas del generador
        """
        self.optimizer_g.zero_grad()
        
        # Forward pass
        stego, recovered = self.model(cover, secret, mode='full')
        
        # Discriminador evalúa stego
        d_stego = self.model.discriminator(stego)
        
        # Pérdida híbrida del paper (Equation 5)
        loss_hybrid, loss_dict = self.hybrid_loss.generator_loss(
            cover_image=cover,
            stego_image=stego,
            secret_original=secret,
            secret_recovered=recovered,
            discriminator_output=d_stego
        )
        
        # Backward y optimización
        loss_hybrid.backward()
        self.optimizer_g.step()
        
        # Métricas
        with torch.no_grad():
            psnr = self.calculate_psnr(cover, stego)
            ssim = self.calculate_ssim(cover, stego)
        
        # Agregar métricas al diccionario de pérdidas
        loss_dict['psnr'] = psnr
        loss_dict['ssim'] = ssim
        
        return loss_dict
    
    def update_discriminator(self, cover, secret):
        """
        Actualización del discriminador con WGAN-GP
        
        Args:
            cover: imágenes cover [B, 3, H, W]
            secret: imágenes secret [B, 3, H, W]
            
        Returns:
            dict con pérdidas del discriminador
        """
        self.optimizer_d.zero_grad()
        
        # Generamos stego (sin gradientes para generator)
        with torch.no_grad():
            stego, _ = self.model(cover, secret, mode='full')
        
        # Discriminador evalúa real y fake
        d_real = self.model.discriminator(cover)
        d_fake = self.model.discriminator(stego)
        
        # WGAN-GP loss
        loss_d = self.wgan_loss.discriminator_loss(d_real, d_fake)
        
        # Backward y optimización
        loss_d.backward()
        self.optimizer_d.step()
        
        return {
            'loss_d': loss_d.item(),
            'd_real': d_real.mean().item(),
            'd_fake': d_fake.mean().item()
        }
    
    def train_epoch(self, epoch, train_loader):
        """
        Entrena una época completa usando estrategia 4:1
        
        Args:
            epoch: número de época actual
            train_loader: DataLoader de entrenamiento
            
        Returns:
            dict con métricas promedio de la época
        """
        self.model.train()
        
        metrics = {
            'loss_g': [],
            'loss_d': [],
            'psnr': [],
            'ssim': []
        }
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (cover, secret) in enumerate(pbar):
            cover = cover.to(self.device)
            secret = secret.to(self.device)
            
            # Estrategia 4:1 del paper
            # 4 actualizaciones generator por cada 1 discriminator
            for _ in range(4):
                g_metrics = self.update_generator(cover, secret)
                metrics['loss_g'].append(g_metrics['loss_total'])
                metrics['psnr'].append(g_metrics['psnr'])
                metrics['ssim'].append(g_metrics['ssim'])
            
            # 1 actualización discriminator
            d_metrics = self.update_discriminator(cover, secret)
            metrics['loss_d'].append(d_metrics['loss_d'])
            
            # Actualizar progress bar cada 10 batches
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'L_G': f"{g_metrics['loss_total']:.4f}",
                    'L_D': f"{d_metrics['loss_d']:.4f}",
                    'PSNR': f"{g_metrics['psnr']:.2f}",
                    'SSIM': f"{g_metrics['ssim']:.4f}"
                })
        
        # Promedios de época
        return {
            'loss_g': np.mean(metrics['loss_g']),
            'loss_d': np.mean(metrics['loss_d']),
            'psnr': np.mean(metrics['psnr']),
            'ssim': np.mean(metrics['ssim'])
        }
    
    @torch.no_grad()
    def validate(self, val_loader):
        """
        Valida el modelo en conjunto de validación
        
        Args:
            val_loader: DataLoader de validación
            
        Returns:
            dict con métricas de validación
        """
        self.model.eval()
        
        psnr_list = []
        ssim_list = []
        recovery_psnr_list = []
        
        for cover, secret in tqdm(val_loader, desc='Validating'):
            cover = cover.to(self.device)
            secret = secret.to(self.device)
            
            # Forward pass
            stego, recovered = self.model(cover, secret, mode='full')
            
            # Métricas de ocultación (cover vs stego)
            psnr_list.append(self.calculate_psnr(cover, stego))
            ssim_list.append(self.calculate_ssim(cover, stego))
            
            # Métricas de recuperación (secret vs recovered)
            recovery_psnr_list.append(self.calculate_psnr(secret, recovered))
        
        return {
            'psnr': np.mean(psnr_list),
            'ssim': np.mean(ssim_list),
            'recovery_psnr': np.mean(recovery_psnr_list)
        }
    
    def calculate_psnr(self, img1, img2):
        """
        Calcula PSNR entre dos imágenes
        
        Args:
            img1, img2: tensores [B, C, H, W] en rango [0, 1]
            
        Returns:
            PSNR promedio del batch
        """
        mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.mean().item()
    
    def calculate_ssim(self, img1, img2, window_size=11):
        """
        Calcula SSIM entre dos imágenes (versión simplificada)
        
        Args:
            img1, img2: tensores [B, C, H, W]
            window_size: tamaño de ventana
            
        Returns:
            SSIM promedio
        """
        # Constantes SSIM
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        
        # Promedios
        mu1 = torch.mean(img1, dim=[2, 3], keepdim=True)
        mu2 = torch.mean(img2, dim=[2, 3], keepdim=True)
        
        # Varianzas
        sigma1_sq = torch.var(img1, dim=[2, 3], keepdim=True)
        sigma2_sq = torch.var(img2, dim=[2, 3], keepdim=True)
        sigma12 = torch.mean((img1 - mu1) * (img2 - mu2), dim=[2, 3], keepdim=True)
        
        # SSIM
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim.mean().item()
    
    def save_checkpoint(self, epoch, metrics, checkpoint_dir):
        """
        Guarda checkpoint del modelo
        
        Args:
            epoch: época actual
            metrics: métricas de validación
            checkpoint_dir: directorio para guardar
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'metrics': metrics,
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim
        }
        
        # Guardar último checkpoint
        torch.save(checkpoint, checkpoint_dir / 'checkpoint_latest.pth')
        
        # Guardar mejor checkpoint si mejora
        if metrics['psnr'] > self.best_psnr:
            self.best_psnr = metrics['psnr']
            torch.save(checkpoint, checkpoint_dir / 'checkpoint_best_psnr.pth')
            self.logger.info(f"✅ Nuevo mejor PSNR: {self.best_psnr:.2f} dB")
        
        if metrics['ssim'] > self.best_ssim:
            self.best_ssim = metrics['ssim']
            torch.save(checkpoint, checkpoint_dir / 'checkpoint_best_ssim.pth')
            self.logger.info(f"✅ Nuevo mejor SSIM: {self.best_ssim:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Carga checkpoint para continuar entrenamiento
        
        Args:
            checkpoint_path: ruta al archivo .pth
            
        Returns:
            epoch desde donde continuar
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        
        self.best_psnr = checkpoint['best_psnr']
        self.best_ssim = checkpoint['best_ssim']
        
        self.logger.info(f"✅ Checkpoint cargado desde época {checkpoint['epoch']}")
        
        return checkpoint['epoch']
    
    def train(self, train_loader, val_loader, num_epochs, checkpoint_dir='checkpoints', val_every=1):
        """
        Loop principal de entrenamiento
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            num_epochs: número de épocas a entrenar
            checkpoint_dir: directorio para checkpoints
            val_every: validar cada N épocas
        """
        self.logger.info(f"🚀 Iniciando entrenamiento por {num_epochs} épocas")
        self.logger.info(f"📊 Device: {self.device}")
        self.logger.info(f"📦 Train batches: {len(train_loader)}")
        self.logger.info(f"📦 Val batches: {len(val_loader)}")
        
        for epoch in range(1, num_epochs + 1):
            # Entrenar época
            train_metrics = self.train_epoch(epoch, train_loader)
            
            self.logger.info(
                f"\nÉpoca {epoch}/{num_epochs} - "
                f"L_G: {train_metrics['loss_g']:.4f}, "
                f"L_D: {train_metrics['loss_d']:.4f}, "
                f"PSNR: {train_metrics['psnr']:.2f} dB, "
                f"SSIM: {train_metrics['ssim']:.4f}"
            )
            
            # Validar cada val_every épocas
            if epoch % val_every == 0:
                val_metrics = self.validate(val_loader)
                
                self.logger.info(
                    f"📊 Validación - "
                    f"PSNR: {val_metrics['psnr']:.2f} dB, "
                    f"SSIM: {val_metrics['ssim']:.4f}, "
                    f"Recovery PSNR: {val_metrics['recovery_psnr']:.2f} dB"
                )
                
                # Guardar checkpoint
                self.save_checkpoint(epoch, val_metrics, checkpoint_dir)
            
            # Actualizar learning rates
            self.scheduler_g.step()
            self.scheduler_d.step()
        
        self.logger.info(f"\n✅ Entrenamiento completado!")
        self.logger.info(f"🏆 Mejor PSNR: {self.best_psnr:.2f} dB")
        self.logger.info(f"🏆 Mejor SSIM: {self.best_ssim:.4f}")
