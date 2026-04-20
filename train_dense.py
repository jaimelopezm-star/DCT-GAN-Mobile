"""
Train Dense - Entrenamiento con arquitectura inspirada en SteganoGAN

Este script usa la metodología de SteganoGAN (MIT) que ha demostrado funcionar:
1. DenseEncoder con dense connections
2. DenseDecoder con dense connections
3. Loss weights: 100*MSE + 1*BCE (invertido vs nuestro original)
4. Sin DCT inicialmente (dominio espacial directo)

Uso:
    python train_dense.py --config configs/exp18_steganogan_style.yaml
    
    O con argumentos directos:
    python train_dense.py --epochs 100 --batch_size 8 --lr 1e-4
"""

import os
import sys
import argparse
import yaml
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Añadir src al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.dense_encoder import DenseEncoder, DenseEncoderLarge
from src.models.dense_decoder import DenseDecoder, DenseDecoderLarge, DenseDecoderWithSkip


class ImagePairDataset(Dataset):
    """Dataset que carga pares de imágenes (cover, secret) del mismo directorio"""
    
    def __init__(self, data_dir, image_size=256, max_images=None):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        # Buscar todas las imágenes
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        self.images = []
        for ext in extensions:
            self.images.extend(list(self.data_dir.glob(ext)))
            self.images.extend(list(self.data_dir.glob(ext.upper())))
        
        self.images = sorted(self.images)
        
        if max_images:
            self.images = self.images[:max_images]
            
        if len(self.images) < 2:
            raise ValueError(f"Se necesitan al menos 2 imágenes en {data_dir}")
        
        # Transform: resize y normalizar a [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -> [-1, 1]
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Cover: imagen actual
        cover_path = self.images[idx]
        cover = Image.open(cover_path).convert('RGB')
        cover = self.transform(cover)
        
        # Secret: imagen aleatoria diferente
        secret_idx = np.random.randint(0, len(self.images))
        while secret_idx == idx and len(self.images) > 1:
            secret_idx = np.random.randint(0, len(self.images))
        
        secret_path = self.images[secret_idx]
        secret = Image.open(secret_path).convert('RGB')
        secret = self.transform(secret)
        
        return cover, secret


def calculate_psnr(img1, img2):
    """Calcula PSNR entre dos tensores en rango [-1, 1]"""
    # Convertir de [-1,1] a [0,1]
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2
    
    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()


def train_epoch(encoder, decoder, dataloader, optimizer, device, 
                mse_weight=100.0, bce_weight=1.0, epoch=0):
    """
    Entrena una época completa
    
    Loss estilo SteganoGAN:
        L = mse_weight * MSE(cover, stego) + bce_weight * MSE(secret, recovered)
    
    Note: Usamos MSE para recovery en lugar de BCE porque nuestro output es imagen RGB,
    no bits. SteganoGAN usa BCE porque trabaja con bits binarios.
    """
    encoder.train()
    decoder.train()
    
    total_loss = 0
    total_mse_loss = 0
    total_rec_loss = 0
    total_psnr = 0
    total_rec_psnr = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (cover, secret) in enumerate(pbar):
        cover = cover.to(device)
        secret = secret.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        stego = encoder(cover, secret)  # Genera imagen stego
        
        # El decoder recibe stego y debe recuperar secret
        # Pero secret está en [-1,1] y decoder output en [0,1]
        # Convertimos secret a [0,1] para comparar
        secret_01 = (secret + 1) / 2  # [-1,1] -> [0,1]
        
        recovered = decoder(stego)  # Output en [0,1]
        
        # Losses
        # 1. MSE entre cover y stego (calidad visual)
        loss_mse = nn.functional.mse_loss(stego, cover)
        
        # 2. MSE entre secret y recovered (recuperación)
        loss_rec = nn.functional.mse_loss(recovered, secret_01)
        
        # Loss total estilo SteganoGAN
        loss = mse_weight * loss_mse + bce_weight * loss_rec
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Métricas
        with torch.no_grad():
            psnr = calculate_psnr(cover, stego)
            
            # Para PSNR de recovery, convertir recovered a [-1,1]
            recovered_11 = recovered * 2 - 1  # [0,1] -> [-1,1]
            rec_psnr = calculate_psnr(secret, recovered_11)
        
        total_loss += loss.item()
        total_mse_loss += loss_mse.item()
        total_rec_loss += loss_rec.item()
        total_psnr += psnr
        total_rec_psnr += rec_psnr
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'psnr': f'{psnr:.2f}',
            'rec_psnr': f'{rec_psnr:.2f}'
        })
    
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'mse_loss': total_mse_loss / n_batches,
        'rec_loss': total_rec_loss / n_batches,
        'psnr': total_psnr / n_batches,
        'rec_psnr': total_rec_psnr / n_batches
    }


def validate(encoder, decoder, dataloader, device, mse_weight=100.0, bce_weight=1.0):
    """Validación"""
    encoder.eval()
    decoder.eval()
    
    total_loss = 0
    total_psnr = 0
    total_rec_psnr = 0
    
    with torch.no_grad():
        for cover, secret in dataloader:
            cover = cover.to(device)
            secret = secret.to(device)
            
            stego = encoder(cover, secret)
            secret_01 = (secret + 1) / 2
            recovered = decoder(stego)
            
            loss_mse = nn.functional.mse_loss(stego, cover)
            loss_rec = nn.functional.mse_loss(recovered, secret_01)
            loss = mse_weight * loss_mse + bce_weight * loss_rec
            
            psnr = calculate_psnr(cover, stego)
            recovered_11 = recovered * 2 - 1
            rec_psnr = calculate_psnr(secret, recovered_11)
            
            total_loss += loss.item()
            total_psnr += psnr
            total_rec_psnr += rec_psnr
    
    n_batches = len(dataloader)
    return {
        'val_loss': total_loss / n_batches,
        'val_psnr': total_psnr / n_batches,
        'val_rec_psnr': total_rec_psnr / n_batches
    }


def main():
    parser = argparse.ArgumentParser(description='Train Dense SteganoGAN-style')
    parser.add_argument('--config', type=str, default=None, help='Config YAML file')
    parser.add_argument('--data_dir', type=str, default='data/div2k/train', help='Training data directory')
    parser.add_argument('--val_dir', type=str, default='data/div2k/val', help='Validation data directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--mse_weight', type=float, default=100.0, help='MSE loss weight (SteganoGAN=100)')
    parser.add_argument('--rec_weight', type=float, default=1.0, help='Recovery loss weight (SteganoGAN=1)')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden channels')
    parser.add_argument('--encoder_type', type=str, default='dense', choices=['dense', 'dense_large'])
    parser.add_argument('--decoder_type', type=str, default='dense', choices=['dense', 'dense_large', 'dense_skip'])
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/exp18', help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override args with config
        for key, value in config.get('training', {}).items():
            if hasattr(args, key):
                setattr(args, key, value)
        for key, value in config.get('model', {}).items():
            if hasattr(args, key):
                setattr(args, key, value)
        for key, value in config.get('loss', {}).items():
            if hasattr(args, key):
                setattr(args, key, value)
        
        # Ensure numeric types
        args.lr = float(args.lr)
        args.mse_weight = float(args.mse_weight)
        args.rec_weight = float(args.rec_weight)
        args.epochs = int(args.epochs)
        args.batch_size = int(args.batch_size)
        args.hidden_size = int(args.hidden_size)
        args.patience = int(args.patience)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"EXP 18: SteganoGAN-Style Dense Architecture")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Encoder: {args.encoder_type}, Decoder: {args.decoder_type}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Loss weights: MSE={args.mse_weight}, Recovery={args.rec_weight}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*60}\n")
    
    # Create checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create models
    if args.encoder_type == 'dense':
        encoder = DenseEncoder(data_depth=3, hidden_size=args.hidden_size)
    else:
        encoder = DenseEncoderLarge(data_depth=3, hidden_size=args.hidden_size)
    
    if args.decoder_type == 'dense':
        decoder = DenseDecoder(data_depth=3, hidden_size=args.hidden_size)
    elif args.decoder_type == 'dense_large':
        decoder = DenseDecoderLarge(data_depth=3, hidden_size=args.hidden_size)
    else:
        decoder = DenseDecoderWithSkip(data_depth=3, hidden_size=args.hidden_size)
    
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    print(f"Encoder parameters: {encoder.get_num_params():,}")
    print(f"Decoder parameters: {decoder.get_num_params():,}")
    print(f"Total parameters: {encoder.get_num_params() + decoder.get_num_params():,}")
    
    # Dataset
    train_dataset = ImagePairDataset(args.data_dir, args.image_size)
    val_dataset = ImagePairDataset(args.val_dir, args.image_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Optimizer (joint training like SteganoGAN)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    
    # Training loop
    best_psnr = 0
    best_rec_psnr = 0
    patience_counter = 0
    
    history = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            encoder, decoder, train_loader, optimizer, device,
            mse_weight=args.mse_weight, bce_weight=args.rec_weight,
            epoch=epoch
        )
        
        # Validate
        val_metrics = validate(
            encoder, decoder, val_loader, device,
            mse_weight=args.mse_weight, bce_weight=args.rec_weight
        )
        
        # Combine metrics
        metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
        history.append(metrics)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, PSNR: {train_metrics['psnr']:.2f} dB, Rec PSNR: {train_metrics['rec_psnr']:.2f} dB")
        print(f"  Val   - Loss: {val_metrics['val_loss']:.4f}, PSNR: {val_metrics['val_psnr']:.2f} dB, Rec PSNR: {val_metrics['val_rec_psnr']:.2f} dB")
        
        # Check for improvement (usando combinación de PSNR visual y recovery)
        combined_score = val_metrics['val_psnr'] + val_metrics['val_rec_psnr']
        best_combined = best_psnr + best_rec_psnr
        
        if combined_score > best_combined:
            best_psnr = val_metrics['val_psnr']
            best_rec_psnr = val_metrics['val_rec_psnr']
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_metrics['val_psnr'],
                'val_rec_psnr': val_metrics['val_rec_psnr'],
                'history': history,
                'config': vars(args)
            }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            
            print(f"  *** New best! PSNR: {best_psnr:.2f}, Rec PSNR: {best_rec_psnr:.2f} ***")
        else:
            patience_counter += 1
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best Visual PSNR: {best_psnr:.2f} dB")
    print(f"Best Recovery PSNR: {best_rec_psnr:.2f} dB")
    print(f"Target: Visual ≥ 30 dB, Recovery ≥ 20 dB")
    print(f"{'='*60}")
    
    # Save final model
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'final_metrics': {
            'best_psnr': best_psnr,
            'best_rec_psnr': best_rec_psnr
        },
        'history': history,
    }, os.path.join(args.checkpoint_dir, 'final_model.pth'))


if __name__ == '__main__':
    main()
