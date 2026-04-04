"""
Training Module - Módulo de Entrenamiento

Contiene:
- losses.py: Funciones de pérdida (Ecuación 5)
- trainer.py: Pipeline de entrenamiento
- metrics.py: Métricas de evaluación
"""

from .losses import (
    MSELoss,
    BCERecoveryLoss,
    WassersteinGANLoss,
    GradientPenalty,
    HybridLoss,
    calculate_psnr
)

from .metrics import (
    calculate_ssim,
    calculate_rmse,
    calculate_mse,
    calculate_recovery_accuracy,
    calculate_bit_error_rate,
    calculate_all_metrics
)

from .trainer import DCTGANTrainer

__all__ = [
    # Losses
    'MSELoss',
    'BCERecoveryLoss',
    'WassersteinGANLoss',
    'GradientPenalty',
    'HybridLoss',
    
    # Metrics
    'calculate_psnr',
    'calculate_ssim',
    'calculate_rmse',
    'calculate_mse',
    'calculate_recovery_accuracy',
    'calculate_bit_error_rate',
    'calculate_all_metrics',
    
    # Trainer
    'DCTGANTrainer'
]
