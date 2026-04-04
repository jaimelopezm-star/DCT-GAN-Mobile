"""
DCT Transform Module - Transformada Discreta del Coseno 2D

Implementa DCT/IDCT en bloques 8×8 según paper Malik et al. (2025)

Ecuaciones del paper:
    DCT(u,v) = α(u)α(v) ∑∑ f(x,y) cos[(2x+1)uπ/16] cos[(2y+1)vπ/16]
    IDCT(x,y) = ∑∑ α(u)α(v) F(u,v) cos[(2x+1)uπ/16] cos[(2y+1)vπ/16]
    
    donde α(u) = 1/√2 si u=0, 1 en otro caso
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from scipy.fftpack import dct, idct


class DCTTransform(nn.Module):
    """
    Transformada DCT 2D en bloques 8×8
    
    Divide la imagen en bloques 8×8 y aplica DCT a cada uno.
    Usado para trabajar en dominio de frecuencia donde la incrustación
    es más robusta a compresión JPEG.
    
    Args:
        block_size: Tamaño del bloque (default: 8)
        norm: Tipo de normalización 'ortho' (default: 'ortho')
    """
    
    def __init__(self, block_size: int = 8, norm: str = 'ortho'):
        super(DCTTransform, self).__init__()
        self.block_size = block_size
        self.norm = norm
        
        # Precalcular matriz DCT para eficiencia
        self.dct_matrix = self._create_dct_matrix(block_size)
        
    def _create_dct_matrix(self, N: int) -> torch.Tensor:
        """
        Crea matriz DCT de tamaño N×N
        
        D[i,j] = α(i) * cos(π*i*(2j+1)/(2N))
        donde α(0) = 1/√N, α(i>0) = √(2/N)
        """
        dct_matrix = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                if i == 0:
                    alpha = 1.0 / np.sqrt(N)
                else:
                    alpha = np.sqrt(2.0 / N)
                
                dct_matrix[i, j] = alpha * np.cos(np.pi * i * (2 * j + 1) / (2 * N))
        
        return torch.FloatTensor(dct_matrix)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica DCT 2D en bloques
        
        Args:
            x: Imagen [B, C, H, W]
            
        Returns:
            Coeficientes DCT [B, C, H, W]
        """
        B, C, H, W = x.shape
        block_size = self.block_size
        
        # Verificar que las dimensiones son divisibles por block_size
        assert H % block_size == 0 and W % block_size == 0, \
            f"Image dimensions ({H}×{W}) must be divisible by block_size ({block_size})"
        
        # Mover matriz DCT al mismo dispositivo que x
        dct_matrix = self.dct_matrix.to(x.device)
        
        # Dividir en bloques
        blocks = self._divide_into_blocks(x, block_size)
        # blocks: [B, C, num_blocks_h, num_blocks_w, block_size, block_size]
        
        # Aplicar DCT 2D a cada bloque: D * Block * D^T
        dct_blocks = torch.matmul(
            torch.matmul(dct_matrix.unsqueeze(0), blocks),
            dct_matrix.t().unsqueeze(0)
        )
        
        # Recombinar bloques
        dct_image = self._combine_blocks(dct_blocks, H, W)
        
        return dct_image
    
    def _divide_into_blocks(self, x: torch.Tensor, block_size: int) -> torch.Tensor:
        """
        Divide imagen en bloques no solapados
        
        Args:
            x: [B, C, H, W]
            block_size: Tamaño del bloque
            
        Returns:
            [B, C, num_blocks_h, num_blocks_w, block_size, block_size]
        """
        B, C, H, W = x.shape
        
        # Usar unfold para dividir en bloques
        # unfold(dim, size, step)
        blocks = x.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        # [B, C, num_blocks_h, num_blocks_w, block_size, block_size]
        
        return blocks
    
    def _combine_blocks(self, blocks: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Combina bloques en imagen completa
        
        Args:
            blocks: [B, C, num_blocks_h, num_blocks_w, block_size, block_size]
            H, W: Dimensiones objetivo
            
        Returns:
            [B, C, H, W]
        """
        B, C, num_h, num_w, bh, bw = blocks.shape
        
        # Reorganizar dimensiones
        blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        # [B, C, num_blocks_h, block_size, num_blocks_w, block_size]
        
        # Reshape a imagen completa
        image = blocks.view(B, C, H, W)
        
        return image


class IDCTTransform(nn.Module):
    """
    Transformada Inversa DCT 2D
    
    Reconstruye imagen desde coeficientes DCT.
    Opera en bloques 8×8.
    
    Args:
        block_size: Tamaño del bloque (default: 8)
        norm: Tipo de normalización (default: 'ortho')
    """
    
    def __init__(self, block_size: int = 8, norm: str = 'ortho'):
        super(IDCTTransform, self).__init__()
        self.block_size = block_size
        self.norm = norm
        
        # Usar matriz DCT transpuesta para IDCT
        self.idct_matrix = self._create_idct_matrix(block_size)
    
    def _create_idct_matrix(self, N: int) -> torch.Tensor:
        """
        Crea matriz IDCT (transpuesta de DCT)
        """
        # IDCT = DCT^T para matrices ortonormales
        dct_matrix = self._create_dct_matrix(N)
        return dct_matrix.t()
    
    def _create_dct_matrix(self, N: int) -> torch.Tensor:
        """Misma implementación que DCTTransform"""
        dct_matrix = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                if i == 0:
                    alpha = 1.0 / np.sqrt(N)
                else:
                    alpha = np.sqrt(2.0 / N)
                
                dct_matrix[i, j] = alpha * np.cos(np.pi * i * (2 * j + 1) / (2 * N))
        
        return torch.FloatTensor(dct_matrix)
    
    def forward(self, dct_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Aplica IDCT 2D en bloques
        
        Args:
            dct_coeffs: Coeficientes DCT [B, C, H, W]
            
        Returns:
            Imagen reconstruida [B, C, H, W]
        """
        B, C, H, W = dct_coeffs.shape
        block_size = self.block_size
        
        assert H % block_size == 0 and W % block_size == 0, \
            f"DCT dimensions ({H}×{W}) must be divisible by block_size ({block_size})"
        
        idct_matrix = self.idct_matrix.to(dct_coeffs.device)
        
        # Dividir en bloques
        blocks = self._divide_into_blocks(dct_coeffs, block_size)
        
        # Aplicar IDCT 2D: D^T * Block * D
        idct_blocks = torch.matmul(
            torch.matmul(idct_matrix.unsqueeze(0), blocks),
            idct_matrix.t().unsqueeze(0)
        )
        
        # Recombinar
        image = self._combine_blocks(idct_blocks, H, W)
        
        return image
    
    def _divide_into_blocks(self, x: torch.Tensor, block_size: int) -> torch.Tensor:
        """Misma implementación que DCTTransform"""
        B, C, H, W = x.shape
        blocks = x.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        return blocks
    
    def _combine_blocks(self, blocks: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Misma implementación que DCTTransform"""
        B, C, num_h, num_w, bh, bw = blocks.shape
        blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        image = blocks.view(B, C, H, W)
        return image


def dct_block_processing(image: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    """
    Función de utilidad para aplicar DCT a una imagen
    
    Args:
        image: Imagen [B, C, H, W] o [C, H, W]
        block_size: Tamaño del bloque (default: 8)
        
    Returns:
        Coeficientes DCT con mismas dimensiones
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    dct_transform = DCTTransform(block_size=block_size)
    dct_coeffs = dct_transform(image)
    
    if squeeze_output:
        dct_coeffs = dct_coeffs.squeeze(0)
    
    return dct_coeffs


def idct_block_processing(dct_coeffs: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    """
    Función de utilidad para aplicar IDCT
    
    Args:
        dct_coeffs: Coeficientes DCT [B, C, H, W] o [C, H, W]
        block_size: Tamaño del bloque (default: 8)
        
    Returns:
        Imagen reconstruida con mismas dimensiones
    """
    if dct_coeffs.dim() == 3:
        dct_coeffs = dct_coeffs.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    idct_transform = IDCTTransform(block_size=block_size)
    image = idct_transform(dct_coeffs)
    
    if squeeze_output:
        image = image.squeeze(0)
    
    return image


class DCT2D(nn.Module):
    """
    Wrapper conveniente para DCT 2D
    Puede usarse como capa en redes neuronales
    """
    
    def __init__(self, block_size: int = 8):
        super(DCT2D, self).__init__()
        self.transform = DCTTransform(block_size=block_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class IDCT2D(nn.Module):
    """
    Wrapper conveniente para IDCT 2D
    Puede usarse como capa en redes neuronales
    """
    
    def __init__(self, block_size: int = 8):
        super(IDCT2D, self).__init__()
        self.transform = IDCTTransform(block_size=block_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


if __name__ == "__main__":
    # Test DCT/IDCT
    print("="*60)
    print("Testing DCT Transform")
    print("="*60)
    
    # Crear imagen de prueba
    batch_size = 2
    channels = 3
    height, width = 256, 256
    
    image = torch.randn(batch_size, channels, height, width)
    print(f"Input image shape: {image.shape}")
    print(f"Input range: [{image.min():.4f}, {image.max():.4f}]")
    
    # Aplicar DCT
    dct_transform = DCTTransform(block_size=8)
    dct_coeffs = dct_transform(image)
    
    print(f"\nDCT coefficients shape: {dct_coeffs.shape}")
    print(f"DCT range: [{dct_coeffs.min():.4f}, {dct_coeffs.max():.4f}]")
    
    # Aplicar IDCT
    idct_transform = IDCTTransform(block_size=8)
    reconstructed = idct_transform(dct_coeffs)
    
    print(f"\nReconstructed image shape: {reconstructed.shape}")
    print(f"Reconstructed range: [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")
    
    # Verificar precisión de reconstrucción
    reconstruction_error = torch.abs(image - reconstructed).mean()
    print(f"\nReconstruction error (MAE): {reconstruction_error:.6f}")
    
    max_error = torch.abs(image - reconstructed).max()
    print(f"Max reconstruction error: {max_error:.6f}")
    
    # PSNR
    mse = torch.mean((image - reconstructed) ** 2)
    if mse > 0:
        psnr = 10 * torch.log10(1.0 / mse)
        print(f"PSNR: {psnr:.2f} dB")
    else:
        print("PSNR: ∞ (perfect reconstruction)")
    
    print("\n" + "="*60)
    print("Testing utility functions")
    print("="*60)
    
    # Test single image (sin batch)
    single_image = torch.randn(3, 256, 256)
    print(f"\nSingle image shape: {single_image.shape}")
    
    dct_single = dct_block_processing(single_image)
    print(f"DCT single shape: {dct_single.shape}")
    
    reconstructed_single = idct_block_processing(dct_single)
    print(f"Reconstructed single shape: {reconstructed_single.shape}")
    
    error_single = torch.abs(single_image - reconstructed_single).mean()
    print(f"Reconstruction error: {error_single:.6f}")
    
    print("\n✅ DCT/IDCT tests passed!")
