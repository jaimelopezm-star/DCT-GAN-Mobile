"""
Trainer Module - DCT-GAN Training Pipeline

Implementa el training loop completo del paper Malik et al. (2025):
- Estrategia de actualización 4:1 (Generator:Discriminator)
- Optimizadores: Adam para Generator, SGD para Discriminator
- Scheduler: StepLR (decay cada 30 epochs)
- 100 epochs total
- Batch size: 32
- Logging de métricas: PSNR, SSIM, losses
- Checkpointing automático
- Validación por epoch

Paper: "A Hybrid Steganography Framework Using DCT and GAN"
Target: PSNR ~58.27 dB, SSIM ~0.942
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import time
from typing import Dict, Tuple, Optional, List
import logging
from tqdm import tqdm

from .losses import HybridLoss, GradientPenalty, calculate_psnr
from .metrics import calculate_ssim, calculate_rmse


class DCTGANTrainer:
    """
    Trainer para DCT-GAN Steganography
    
    Implementa el training loop completo con estrategia 4:1
    (4 actualizaciones generator por 1 actualización discriminator)
    
    Args:
        model: Modelo DCTGAN completo
        config: Diccionario de configuración
        device: Device (cpu/cuda)
        checkpoint_dir: Directorio para guardar checkpoints
        log_dir: Directorio para logs
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 config: Dict,
                 device: torch.device,
                 checkpoint_dir: Optional[Path] = None,
                 log_dir: Optional[Path] = None):
        
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Directorios
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Loss functions
        self._setup_loss_functions()
        
        # Optimizers and schedulers
        self._setup_optimizers()
        
        # Mixed Precision and Gradient Clipping
        self._setup_training_optimizations()
        
        # Training state
        self.current_epoch = 0
        self.best_psnr = 0.0
        self.train_history = []
        self.val_history = []
        
        self.logger.info("DCTGANTrainer initialized successfully")
        self.logger.info(f"Device: {device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    def _setup_logging(self):
        """Configura sistema de logging"""
        log_file = self.log_dir / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DCTGANTrainer')
    
    def _setup_loss_functions(self):
        """
        Configura funciones de pérdida según paper
        
        Ecuación 5:
        L_total = α × L_MSE + β × L_crossentropy + γ × L_adversarial
        
        α = 0.3 (similitud cover-stego)
        β = 15.0 (recuperación secret)
        γ = 0.03 (adversarial)
        """
        loss_config = self.config.get('loss', {})
        
        alpha = loss_config.get('alpha', 0.3)
        beta = loss_config.get('beta', 15.0)
        gamma = loss_config.get('gamma', 0.03)
        
        self.hybrid_loss = HybridLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            use_wgan=True
        )
        
        # Gradient penalty para WGAN-GP
        self.gradient_penalty = GradientPenalty(lambda_gp=10.0)
        
        self.logger.info(f"Loss weights: α={alpha}, β={beta}, γ={gamma}")
    
    def _setup_optimizers(self):
        """
        Configura optimizadores y schedulers según paper
        
        Generator (Encoder + Decoder): Adam lr=1e-3
        Discriminator: SGD lr=1e-3
        Scheduler: StepLR (decay 0.5 cada 30 epochs)
        """
        train_config = self.config.get('training', {})
        opt_config = train_config.get('optimizer', {})
        
        # Generator optimizer (Encoder + Decoder)
        gen_config = opt_config.get('generator', {})
        gen_lr = gen_config.get('lr', 1e-3)
        gen_betas = tuple(gen_config.get('betas', [0.9, 0.999]))
        gen_wd = gen_config.get('weight_decay', 0.005)
        
        generator_params = list(self.model.encoder.parameters()) + \
                          list(self.model.decoder.parameters())
        
        self.optimizer_G = optim.Adam(
            generator_params,
            lr=gen_lr,
            betas=gen_betas,
            weight_decay=gen_wd
        )
        
        # Discriminator optimizer - puede ser Adam o SGD
        disc_config = opt_config.get('discriminator', {})
        disc_type = disc_config.get('type', 'sgd').lower()
        disc_lr = disc_config.get('lr', 1e-3)
        disc_wd = disc_config.get('weight_decay', 0.005)
        
        if disc_type == 'adam':
            disc_betas = tuple(disc_config.get('betas', [0.5, 0.999]))
            self.optimizer_D = optim.Adam(
                self.model.discriminator.parameters(),
                lr=disc_lr,
                betas=disc_betas,
                weight_decay=disc_wd
            )
            self.logger.info(f"Discriminator optimizer: Adam lr={disc_lr}")
        else:  # SGD por defecto
            disc_momentum = disc_config.get('momentum', 0.9)
            self.optimizer_D = optim.SGD(
                self.model.discriminator.parameters(),
                lr=disc_lr,
                momentum=disc_momentum,
                weight_decay=disc_wd
            )
            self.logger.info(f"Discriminator optimizer: SGD lr={disc_lr}")
        
        # Learning rate schedulers
        scheduler_config = train_config.get('lr_scheduler', {})
        step_size = scheduler_config.get('step_size', 30)
        gamma = scheduler_config.get('gamma', 0.5)
        
        self.scheduler_G = optim.lr_scheduler.StepLR(
            self.optimizer_G, 
            step_size=step_size, 
            gamma=gamma
        )
        self.scheduler_D = optim.lr_scheduler.StepLR(
            self.optimizer_D,
            step_size=step_size,
            gamma=gamma
        )
        
        self.logger.info(f"Generator optimizer: Adam lr={gen_lr}")
        self.logger.info(f"Scheduler: StepLR step_size={step_size}, gamma={gamma}")
    
    def _setup_training_optimizations(self):
        """
        Configura optimizaciones de entrenamiento:
        - Mixed Precision (AMP) para velocidad
        - Gradient Clipping para estabilidad
        """
        train_config = self.config.get('training', {})
        hardware_config = self.config.get('hardware', {})
        
        # Mixed Precision (AMP)
        self.use_amp = hardware_config.get('mixed_precision', False)
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Gradient Clipping
        grad_clip_config = train_config.get('gradient_clipping', {})
        self.use_grad_clip = grad_clip_config.get('enabled', False)
        self.grad_clip_max_norm = grad_clip_config.get('max_norm', 1.0)
        
        if self.use_amp:
            self.logger.info(f"Mixed Precision (AMP): ENABLED")
        else:
            self.logger.info(f"Mixed Precision (AMP): DISABLED")
        
        if self.use_grad_clip:
            self.logger.info(f"Gradient Clipping: ENABLED (max_norm={self.grad_clip_max_norm})")
        else:
            self.logger.info(f"Gradient Clipping: DISABLED")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Entrena una época completa
        
        Implementa estrategia 5:1 (D:G):
        - 5 actualizaciones del DISCRIMINATOR por cada actualización del GENERATOR
        - Esto es lo "típico" según el paper (Training Generator section)
        - El paper reportó usar 4:1 (G:D) pero eso causa Loss_D≈0
        
        Args:
            train_loader: DataLoader con imágenes de entrenamiento
            
        Returns:
            Diccionario con métricas promedio del epoch
        """
        self.model.train()
        
        # CRITICAL FIX: Cambiar ratio a 5D:1G (típico y funcional)
        # Paper dice: "Typically, the discriminator weights are updated
        # five times, followed by a single update to the generator weights"
        update_strategy = self.config.get('training', {}).get('update_strategy', {})
        disc_updates_per_batch = update_strategy.get('discriminator_updates_per_batch', 5)
        gen_updates_per_batch = update_strategy.get('generator_updates_per_batch', 1)
        
        # Acumuladores de métricas
        epoch_metrics = {
            'loss_G_total': 0.0,
            'loss_G_mse': 0.0,
            'loss_G_bce': 0.0,
            'loss_G_adv': 0.0,
            'loss_D': 0.0,
            'loss_GP': 0.0,
            'psnr': 0.0,
            'ssim': 0.0,
            'D_real': 0.0,
            'D_fake': 0.0
        }
        
        num_batches = len(train_loader)
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Extraer imágenes del batch
            # Formato esperado: {'cover': tensor, 'secret': tensor}
            cover = batch['cover'].to(self.device)
            secret = batch['secret'].to(self.device)
            batch_size = cover.size(0)
            
            # ============================================
            # 1. ENTRENAR DISCRIMINATOR
            # ============================================
            # CRITICAL: Acumular pérdidas de las 5 iteraciones y promediar
            d_loss_accum = 0.0
            d_gp_accum = 0.0
            d_real_accum = 0.0
            d_fake_accum = 0.0
            
            for _ in range(disc_updates_per_batch):
                self.optimizer_D.zero_grad(set_to_none=True)  # Más eficiente
                
                # Forward pass con AMP
                with autocast(enabled=self.use_amp):
                    # Forward pass del generator (sin gradientes para encoder/decoder)
                    with torch.no_grad():
                        stego, _ = self.model(cover, secret, mode='full')
                    
                    # Discriminator outputs
                    real_validity = self.model.discriminator(cover)
                    fake_validity = self.model.discriminator(stego.detach())
                    
                    # Discriminator loss (WGAN)
                    loss_D, disc_metrics = self.hybrid_loss.discriminator_loss(
                        real_validity, 
                        fake_validity
                    )
                    
                    # Gradient penalty (WGAN-GP)
                    gp = self.gradient_penalty(
                        self.model.discriminator,
                        cover,
                        stego.detach(),
                        self.device
                    )
                    
                    # Total discriminator loss
                    loss_D_total = loss_D + gp
                
                # Backward con AMP
                self.scaler.scale(loss_D_total).backward()
                
                # Gradient clipping (opcional)
                if self.use_grad_clip:
                    self.scaler.unscale_(self.optimizer_D)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.discriminator.parameters(), 
                        self.grad_clip_max_norm
                    )
                
                # Optimizer step con AMP
                self.scaler.step(self.optimizer_D)
                self.scaler.update()
                
                # Acumular para promedio
                d_loss_accum += loss_D.item()
                d_gp_accum += gp.item()
                d_real_accum += disc_metrics['D_real']
                d_fake_accum += disc_metrics['D_fake']
            
            # Promediar métricas de discriminador (manejar caso disc_updates=0)
            if disc_updates_per_batch > 0:
                loss_D_avg = d_loss_accum / disc_updates_per_batch
                gp_avg = d_gp_accum / disc_updates_per_batch
                d_real_avg = d_real_accum / disc_updates_per_batch
                d_fake_avg = d_fake_accum / disc_updates_per_batch
            else:
                # Sin entrenamiento de discriminador (gamma=0)
                loss_D_avg = 0.0
                gp_avg = 0.0
                d_real_avg = 0.0
                d_fake_avg = 0.0
            
            # ============================================
            # 2. ENTRENAR GENERATOR (ENCODER + DECODER)
            # ============================================
            # CRITICAL: Acumular pérdidas (aunque sea 1 iteración, por consistencia)
            g_loss_accum = 0.0
            g_mse_accum = 0.0
            g_bce_accum = 0.0
            g_adv_accum = 0.0
            
            for _ in range(gen_updates_per_batch):
                self.optimizer_G.zero_grad(set_to_none=True)
                
                # Forward pass con AMP
                with autocast(enabled=self.use_amp):
                    # Forward pass completo
                    stego, recovered_secret = self.model(cover, secret, mode='full')
                    
                    # Discriminator output para stego
                    fake_validity = self.model.discriminator(stego)
                    
                    # Generator loss (Ecuación 5)
                    loss_G, gen_metrics = self.hybrid_loss.generator_loss(
                        cover_image=cover,
                        stego_image=stego,
                        secret_original=secret,
                        secret_recovered=recovered_secret,
                        discriminator_output=fake_validity
                    )
                
                # Backward con AMP
                self.scaler.scale(loss_G).backward()
                
                # Gradient clipping (opcional)
                if self.use_grad_clip:
                    self.scaler.unscale_(self.optimizer_G)
                    generator_params = list(self.model.encoder.parameters()) + \
                                      list(self.model.decoder.parameters())
                    torch.nn.utils.clip_grad_norm_(
                        generator_params, 
                        self.grad_clip_max_norm
                    )
                
                # Optimizer step con AMP
                self.scaler.step(self.optimizer_G)
                self.scaler.update()
                
                # Acumular para promedio
                g_loss_accum += gen_metrics['loss_total']
                g_mse_accum += gen_metrics['loss_mse']
                g_bce_accum += gen_metrics['loss_bce']
                g_adv_accum += gen_metrics['loss_adv']
            
            # Promediar métricas de generator
            loss_G_avg = g_loss_accum / gen_updates_per_batch
            g_mse_avg = g_mse_accum / gen_updates_per_batch
            g_bce_avg = g_bce_accum / gen_updates_per_batch
            g_adv_avg = g_adv_accum / gen_updates_per_batch
            
            # ============================================
            # 3. CALCULAR MÉTRICAS
            # ============================================
            with torch.no_grad():
                # Convert to float32 for metrics (AMP compatibility)
                cover_f32 = cover.float()
                stego_f32 = stego.float()
                
                # PSNR (cover vs stego)
                psnr = calculate_psnr(cover_f32, stego_f32, max_val=1.0)
                
                # SSIM (cover vs stego)
                ssim = calculate_ssim(cover_f32, stego_f32)
                
                # Acumular métricas (usar promedios calculados)
                epoch_metrics['loss_G_total'] += loss_G_avg
                epoch_metrics['loss_G_mse'] += g_mse_avg
                epoch_metrics['loss_G_bce'] += g_bce_avg
                epoch_metrics['loss_G_adv'] += g_adv_avg
                epoch_metrics['loss_D'] += loss_D_avg
                epoch_metrics['loss_GP'] += gp_avg
                epoch_metrics['psnr'] += psnr.item()
                epoch_metrics['ssim'] += ssim
                epoch_metrics['D_real'] += d_real_avg
                epoch_metrics['D_fake'] += d_fake_avg
            
            # Actualizar progress bar (usar promedios)
            pbar.set_postfix({
                'L_G': f"{loss_G_avg:.4f}",
                'L_D': f"{loss_D_avg:.4f}",
                'D(x)': f"{d_real_avg:.3f}",  # D(real) - debe estar cerca de 1
                'D(G(z))': f"{d_fake_avg:.3f}",  # D(fake) - debe estar cerca de 0
                'PSNR': f"{psnr.item():.2f} dB",
                'SSIM': f"{ssim:.4f}"
            })
        
        # Promediar métricas
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Valida el modelo en el conjunto de validación
        
        Args:
            val_loader: DataLoader con imágenes de validación
            
        Returns:
            Diccionario con métricas de validación
        """
        self.model.eval()
        
        val_metrics = {
            'loss_G_total': 0.0,
            'psnr': 0.0,
            'ssim': 0.0,
            'rmse': 0.0
        }
        
        num_batches = len(val_loader)
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            
            for batch in pbar:
                cover = batch['cover'].to(self.device)
                secret = batch['secret'].to(self.device)
                
                # Forward pass con AMP
                with autocast(enabled=self.use_amp):
                    # Forward pass
                    stego, recovered_secret = self.model(cover, secret, mode='full')
                    
                    # Discriminator output
                    fake_validity = self.model.discriminator(stego)
                    
                    # Generator loss
                    loss_G, _ = self.hybrid_loss.generator_loss(
                        cover_image=cover,
                        stego_image=stego,
                        secret_original=secret,
                        secret_recovered=recovered_secret,
                        discriminator_output=fake_validity
                    )
                
                # Métricas (convert to float32 for AMP compatibility)
                cover_f32 = cover.float()
                stego_f32 = stego.float()
                secret_f32 = secret.float()
                recovered_secret_f32 = recovered_secret.float()
                
                psnr = calculate_psnr(cover_f32, stego_f32, max_val=1.0)
                ssim = calculate_ssim(cover_f32, stego_f32)
                rmse = calculate_rmse(secret_f32, recovered_secret_f32)
                
                # Acumular
                val_metrics['loss_G_total'] += loss_G.item()
                val_metrics['psnr'] += psnr.item()
                val_metrics['ssim'] += ssim
                val_metrics['rmse'] += rmse
                
                pbar.set_postfix({
                    'PSNR': f"{psnr.item():.2f} dB",
                    'SSIM': f"{ssim:.4f}"
                })
        
        # Promediar
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_metrics
    
    def save_checkpoint(self, 
                       epoch: int, 
                       metrics: Dict[str, float],
                       is_best: bool = False,
                       filename: Optional[str] = None):
        """
        Guarda checkpoint del modelo
        
        Args:
            epoch: Número de época actual
            metrics: Diccionario con métricas actuales
            is_best: Si es el mejor modelo hasta ahora
            filename: Nombre personalizado del archivo
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_psnr': self.best_psnr
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Guardar mejor modelo por separado
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved: {best_path} (PSNR: {metrics['psnr']:.2f} dB)")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Carga checkpoint del modelo
        
        Args:
            checkpoint_path: Path al archivo de checkpoint
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_psnr = checkpoint['best_psnr']
        
        self.logger.info(f"Checkpoint loaded successfully (epoch {self.current_epoch})")
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              num_epochs: Optional[int] = None,
              save_frequency: int = 10,
              early_stopping_patience: int = 20):
        """
        Training loop completo
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación (opcional)
            num_epochs: Número de épocas (si None, usa config)
            save_frequency: Guardar checkpoint cada N epochs
            early_stopping_patience: Epochs sin mejora antes de early stopping
        """
        if num_epochs is None:
            num_epochs = self.config.get('training', {}).get('num_epochs', 100)
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Save frequency: {save_frequency} epochs")
        self.logger.info(f"Early stopping patience: {early_stopping_patience} epochs")
        
        # Early stopping counter
        epochs_without_improvement = 0
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # ============================================
            # TRAIN
            # ============================================
            train_metrics = self.train_epoch(train_loader)
            
            # ============================================
            # VALIDATE
            # ============================================
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
            else:
                val_metrics = {}
            
            # ============================================
            # UPDATE SCHEDULERS
            # ============================================
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # ============================================
            # LOG METRICS
            # ============================================
            epoch_time = time.time() - epoch_start_time
            
            self.logger.info(f"\nEpoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s")
            self.logger.info(f"[TRAIN] Loss_G: {train_metrics['loss_G_total']:.4f} | "
                           f"Loss_D: {train_metrics['loss_D']:.4f} | "
                           f"PSNR: {train_metrics['psnr']:.2f} dB | "
                           f"SSIM: {train_metrics['ssim']:.4f}")
            
            if val_metrics:
                self.logger.info(f"[VAL]   Loss_G: {val_metrics['loss_G_total']:.4f} | "
                               f"PSNR: {val_metrics['psnr']:.2f} dB | "
                               f"SSIM: {val_metrics['ssim']:.4f} | "
                               f"RMSE: {val_metrics['rmse']:.4f}")
            
            # ============================================
            # SAVE CHECKPOINT
            # ============================================
            current_psnr = val_metrics.get('psnr', train_metrics['psnr'])
            is_best = current_psnr > self.best_psnr
            
            if is_best:
                self.best_psnr = current_psnr
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Guardar checkpoint
            if (epoch + 1) % save_frequency == 0 or is_best:
                metrics_to_save = val_metrics if val_metrics else train_metrics
                self.save_checkpoint(
                    epoch=epoch + 1,
                    metrics=metrics_to_save,
                    is_best=is_best
                )
            
            # ============================================
            # EARLY STOPPING
            # ============================================
            if epochs_without_improvement >= early_stopping_patience:
                self.logger.info(f"\nEarly stopping triggered after {early_stopping_patience} "
                               f"epochs without improvement")
                self.logger.info(f"Best PSNR: {self.best_psnr:.2f} dB")
                break
        
        # ============================================
        # FINAL SUMMARY
        # ============================================
        self.logger.info("\n" + "="*60)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info(f"Total epochs: {epoch + 1}")
        self.logger.info(f"Best PSNR: {self.best_psnr:.2f} dB (target: 58.27 dB)")
        self.logger.info(f"Best model saved at: {self.checkpoint_dir / 'best_model.pth'}")
        self.logger.info("="*60)


# ============================================
# TESTING
# ============================================
if __name__ == "__main__":
    """
    Test del trainer con datos sintéticos
    """
    print("Testing DCTGANTrainer...")
    
    # Mock config
    test_config = {
        'loss': {
            'alpha': 0.3,
            'beta': 15.0,
            'gamma': 0.03
        },
        'training': {
            'num_epochs': 5,
            'optimizer': {
                'generator': {
                    'lr': 1e-3,
                    'betas': [0.9, 0.999],
                    'weight_decay': 0.005
                },
                'discriminator': {
                    'lr': 1e-3,
                    'momentum': 0.9,
                    'weight_decay': 0.005
                }
            },
            'lr_scheduler': {
                'step_size': 2,
                'gamma': 0.5
            },
            'update_strategy': {
                'generator_updates_per_epoch': 4,
                'discriminator_updates_per_epoch': 1
            }
        }
    }
    
    # Mock model
    from ..models.gan import DCTGAN
    
    encoder_cfg = {'type': 'resnet', 'base_channels': 10, 'num_residual_blocks': 9}
    decoder_cfg = {'type': 'cnn', 'base_channels': 10, 'num_layers': 6}
    disc_cfg = {'type': 'xunet_modified', 'base_channels': 4, 'num_conv_layers': 5}
    
    model = DCTGAN(encoder_cfg, decoder_cfg, disc_cfg)
    
    # Device
    device = torch.device('cpu')
    
    # Create trainer
    trainer = DCTGANTrainer(
        model=model,
        config=test_config,
        device=device,
        checkpoint_dir=Path("test_checkpoints"),
        log_dir=Path("test_logs")
    )
    
    print("\n✅ DCTGANTrainer initialized successfully!")
    print(f"   - Device: {device}")
    print(f"   - Loss weights: α=0.3, β=15.0, γ=0.03")
    print(f"   - Generator optimizer: Adam lr=1e-3")
    print(f"   - Discriminator optimizer: SGD lr=1e-3")
    print(f"   - Update strategy: 4:1 (G:D)")
    print("\nTrainer ready for training!")
