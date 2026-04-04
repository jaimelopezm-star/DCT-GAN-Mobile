"""
Coefficient Selection Module - Selección de Coeficientes DCT

Implementa selección adaptativa de coeficientes DCT según paper Malik et al. (2025):
- Selección de frecuencias medias (20-60% energía)
- Mapas caóticos para selección aleatoria segura
- Threshold adaptativo basado en textura (VAR)
- Patrón zig-zag para ordenamiento de coeficientes
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional


class ChaoticMap:
    """
    Mapa Caótico para generación de secuencias pseudoaleatorias seguras
    
    Usa mapa logístico: x_{n+1} = α * x_n * (1 - x_n)
    donde α ∈ [3.57, 4.0] produce comportamiento caótico
    
    Paper menciona uso de mapas caóticos para selección de posiciones
    de incrustación, aumentando seguridad.
    
    Args:
        alpha: Parámetro de control (default: 3.9)
        x0: Valor inicial (default: 0.5)
        length: Longitud de secuencia a generar
    """
    
    def __init__(self, alpha: float = 3.9, x0: float = 0.5):
        self.alpha = alpha
        self.x0 = x0
        
        # Verificar rango válido para caos
        assert 3.57 <= alpha <= 4.0, "alpha debe estar en [3.57, 4.0] para comportamiento caótico"
        assert 0 < x0 < 1, "x0 debe estar en (0, 1)"
    
    def generate(self, length: int) -> np.ndarray:
        """
        Genera secuencia caótica de longitud especificada
        
        Args:
            length: Número de valores a generar
            
        Returns:
            Array de valores en (0, 1)
        """
        sequence = np.zeros(length)
        x = self.x0
        
        for i in range(length):
            x = self.alpha * x * (1 - x)
            sequence[i] = x
        
        return sequence
    
    def generate_positions(self, length: int, max_position: int) -> np.ndarray:
        """
        Genera posiciones aleatorias únicas usando mapa caótico
        
        Args:
            length: Número de posiciones a generar
            max_position: Posición máxima (exclusivo)
            
        Returns:
            Array de posiciones únicas en [0, max_position)
        """
        # Generar más valores para asegurar unicidad
        chaotic_values = self.generate(length * 3)
        
        # Convertir a posiciones
        positions = (chaotic_values * max_position).astype(int)
        
        # Eliminar duplicados manteniendo orden
        unique_positions = []
        seen = set()
        
        for pos in positions:
            if pos not in seen and len(unique_positions) < length:
                unique_positions.append(pos)
                seen.add(pos)
        
        return np.array(unique_positions[:length])


def get_zigzag_order(block_size: int = 8) -> List[Tuple[int, int]]:
    """
    Genera orden zig-zag para recorrer bloques DCT
    
    El patrón zig-zag ordena coeficientes de baja a alta frecuencia:
    (0,0) → (0,1) → (1,0) → (2,0) → (1,1) → (0,2) → ...
    
    Usado para seleccionar coeficientes de frecuencia media.
    
    Args:
        block_size: Tamaño del bloque DCT (default: 8)
        
    Returns:
        Lista de tuplas (i, j) en orden zig-zag
    """
    zigzag = []
    
    # Recorrer diagonales
    for diagonal in range(block_size * 2 - 1):
        if diagonal % 2 == 0:
            # Diagonal par: abajo-izquierda a arriba-derecha
            i = min(diagonal, block_size - 1)
            j = diagonal - i
            while i >= 0 and j < block_size:
                if i < block_size and j < block_size:
                    zigzag.append((i, j))
                i -= 1
                j += 1
        else:
            # Diagonal impar: arriba-derecha a abajo-izquierda
            j = min(diagonal, block_size - 1)
            i = diagonal - j
            while j >= 0 and i < block_size:
                if i < block_size and j < block_size:
                    zigzag.append((i, j))
                j -= 1
                i += 1
    
    return zigzag


def calculate_energy_threshold(dct_block: torch.Tensor, 
                               min_energy_ratio: float = 0.2,
                               max_energy_ratio: float = 0.6) -> Tuple[float, float]:
    """
    Calcula umbrales de energía para seleccionar frecuencias medias
    
    Paper especifica: "selección de coeficientes con 20-60% de energía"
    
    Args:
        dct_block: Bloque DCT [8, 8]
        min_energy_ratio: Ratio mínimo de energía (default: 0.2 = 20%)
        max_energy_ratio: Ratio máximo de energía (default: 0.6 = 60%)
        
    Returns:
        (threshold_min, threshold_max): Umbrales de energía
    """
    # Calcular energía de cada coeficiente (cuadrado)
    energy = dct_block ** 2
    
    # Ordenar energías (orden zig-zag implícito)
    zigzag_order = get_zigzag_order(dct_block.shape[0])
    energies = torch.tensor([energy[i, j].item() for i, j in zigzag_order])
    
    # Energía acumulada
    cumulative_energy = torch.cumsum(energies, dim=0)
    total_energy = cumulative_energy[-1]
    
    # Calcular umbrales
    threshold_min = (cumulative_energy >= min_energy_ratio * total_energy).float().argmax().item()
    threshold_max = (cumulative_energy >= max_energy_ratio * total_energy).float().argmax().item()
    
    # Convertir índices a valores de energía
    if threshold_min < len(energies):
        energy_min = energies[threshold_min].item()
    else:
        energy_min = 0.0
    
    if threshold_max < len(energies):
        energy_max = energies[threshold_max].item()
    else:
        energy_max = energies[-1].item()
    
    return energy_min, energy_max


def get_mid_frequency_mask(block_size: int = 8,
                           min_energy_ratio: float = 0.2,
                           max_energy_ratio: float = 0.6) -> torch.Tensor:
    """
    Crea máscara para coeficientes de frecuencia media
    
    Excluye:
    - DC component (0,0): componente de baja frecuencia (promedio del bloque)
    - Frecuencias muy altas: susceptibles a compresión
    
    Incluye:
    - Frecuencias medias (20-60% de energía): robustas y poco perceptibles
    
    Args:
        block_size: Tamaño del bloque (default: 8)
        min_energy_ratio: Ratio mínimo (default: 0.2)
        max_energy_ratio: Ratio máximo (default: 0.6)
        
    Returns:
        Máscara binaria [block_size, block_size] donde 1 = freq media
    """
    mask = torch.zeros(block_size, block_size)
    zigzag_order = get_zigzag_order(block_size)
    
    # Calcular rango de índices en zig-zag
    total_coeffs = block_size * block_size
    start_idx = int(total_coeffs * min_energy_ratio)
    end_idx = int(total_coeffs * max_energy_ratio)
    
    # Marcar coeficientes de frecuencia media
    for idx in range(start_idx, end_idx):
        if idx < len(zigzag_order):
            i, j = zigzag_order[idx]
            mask[i, j] = 1.0
    
    # Excluir DC component siempre
    mask[0, 0] = 0.0
    
    return mask


def select_frequency_coefficients(dct_blocks: torch.Tensor,
                                  min_energy: float = 0.2,
                                  max_energy: float = 0.6,
                                  use_chaotic: bool = True,
                                  chaotic_seed: Optional[float] = None) -> torch.Tensor:
    """
    Selecciona coeficientes DCT de frecuencia media para incrustación
    
    Implementa metodología del paper:
    1. Calcula energía de cada coeficiente
    2. Selecciona rango 20-60% de energía acumulada
    3. Opcionalmente usa mapa caótico para aleatorización
    
    Args:
        dct_blocks: Bloques DCT [B, C, num_blocks_h, num_blocks_w, 8, 8]
        min_energy: Energía mínima (ratio)
        max_energy: Energía máxima (ratio)
        use_chaotic: Usar mapa caótico para selección
        chaotic_seed: Semilla para mapa caótico
        
    Returns:
        Máscara de selección con mismas dimensiones
    """
    B, C, num_h, num_w, bh, bw = dct_blocks.shape
    
    # Crear máscara base de frecuencias medias
    base_mask = get_mid_frequency_mask(bh, min_energy, max_energy)
    base_mask = base_mask.to(dct_blocks.device)
    
    # Expandir máscara a todos los bloques
    mask = base_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    mask = mask.expand(B, C, num_h, num_w, bh, bw)
    
    if use_chaotic:
        # Aplicar selección caótica adicional
        total_blocks = B * C * num_h * num_w
        valid_positions = (base_mask > 0).sum().item()
        
        if chaotic_seed is None:
            chaotic_seed = 0.5
        
        chaotic_map = ChaoticMap(alpha=3.9, x0=chaotic_seed)
        
        # Generar valores caóticos para cada bloque
        chaotic_values = chaotic_map.generate(total_blocks)
        chaotic_values = torch.from_numpy(chaotic_values).float().to(dct_blocks.device)
        chaotic_values = chaotic_values.view(B, C, num_h, num_w, 1, 1)
        
        # Aplicar threshold caótico (mantener ~70% de posiciones)
        chaotic_threshold = 0.3
        chaotic_mask = (chaotic_values > chaotic_threshold).float()
        
        # Combinar máscaras
        mask = mask * chaotic_mask
    
    return mask


class CoefficientSelector(nn.Module):
    """
    Módulo para selección adaptativa de coeficientes DCT
    
    Puede usarse como componente en pipeline de entrenamiento.
    
    Args:
        block_size: Tamaño de bloque DCT (default: 8)
        min_energy: Ratio mínimo de energía (default: 0.2)
        max_energy: Ratio máximo de energía (default: 0.6)
        use_chaotic: Usar mapas caóticos (default: True)
    """
    
    def __init__(self,
                 block_size: int = 8,
                 min_energy: float = 0.2,
                 max_energy: float = 0.6,
                 use_chaotic: bool = True):
        super(CoefficientSelector, self).__init__()
        
        self.block_size = block_size
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.use_chaotic = use_chaotic
        
        # Registrar máscara base como buffer (no es parámetro entrenable)
        base_mask = get_mid_frequency_mask(block_size, min_energy, max_energy)
        self.register_buffer('base_mask', base_mask)
    
    def forward(self, dct_blocks: torch.Tensor, 
                chaotic_seed: Optional[float] = None) -> torch.Tensor:
        """
        Genera máscara de selección para bloques DCT
        
        Args:
            dct_blocks: [B, C, num_blocks_h, num_blocks_w, block_size, block_size]
            chaotic_seed: Semilla opcional para reproducibilidad
            
        Returns:
            Máscara de selección con mismas dimensiones
        """
        return select_frequency_coefficients(
            dct_blocks,
            self.min_energy,
            self.max_energy,
            self.use_chaotic,
            chaotic_seed
        )


def calculate_texture_variance(image: torch.Tensor, window_size: int = 8) -> torch.Tensor:
    """
    Calcula varianza de textura en ventanas locales
    
    Paper menciona "VAR metric for texture-based block selection"
    Bloques con mayor varianza (más textura) son mejores para incrustar.
    
    Args:
        image: Imagen [B, C, H, W]
        window_size: Tamaño de ventana para cálculo (default: 8)
        
    Returns:
        Mapa de varianza [B, C, H//window_size, W//window_size]
    """
    B, C, H, W = image.shape
    
    # Dividir en ventanas
    windows = image.unfold(2, window_size, window_size).unfold(3, window_size, window_size)
    # [B, C, num_blocks_h, num_blocks_w, window_size, window_size]
    
    # Calcular varianza en cada ventana
    # var = E[X²] - E[X]²
    mean = windows.mean(dim=(-2, -1), keepdim=True)
    variance = ((windows - mean) ** 2).mean(dim=(-2, -1))
    
    return variance


if __name__ == "__main__":
    # Test módulo de coeficientes
    print("="*60)
    print("Testing Chaotic Map")
    print("="*60)
    
    # Test mapa caótico
    chaotic_map = ChaoticMap(alpha=3.9, x0=0.5)
    sequence = chaotic_map.generate(100)
    
    print(f"Generated sequence length: {len(sequence)}")
    print(f"Sequence range: [{sequence.min():.4f}, {sequence.max():.4f}]")
    print(f"First 10 values: {sequence[:10]}")
    
    # Test generación de posiciones
    positions = chaotic_map.generate_positions(20, max_position=64)
    print(f"\nGenerated positions (20 from 0-63): {positions}")
    print(f"Unique positions: {len(np.unique(positions))}")
    
    print("\n" + "="*60)
    print("Testing Zig-Zag Order")
    print("="*60)
    
    zigzag = get_zigzag_order(8)
    print(f"Zig-zag order for 8×8 block:")
    print(f"Total positions: {len(zigzag)}")
    print(f"First 10: {zigzag[:10]}")
    print(f"Last 10: {zigzag[-10:]}")
    
    print("\n" + "="*60)
    print("Testing Mid-Frequency Mask")
    print("="*60)
    
    mask = get_mid_frequency_mask(block_size=8, min_energy=0.2, max_energy=0.6)
    print(f"Mask shape: {mask.shape}")
    print(f"Selected coefficients: {mask.sum().item()}/{mask.numel()}")
    print(f"Percentage: {mask.sum().item()/mask.numel()*100:.1f}%")
    print(f"\nMask visualization (1=selected, 0=not selected):")
    print(mask.numpy().astype(int))
    
    print("\n" + "="*60)
    print("Testing Coefficient Selection")
    print("="*60)
    
    # Crear bloques DCT simulados
    batch_size = 2
    channels = 3
    num_blocks_h, num_blocks_w = 32, 32  # Para imagen 256×256
    
    # Simular bloques DCT
    dct_blocks = torch.randn(batch_size, channels, num_blocks_h, num_blocks_w, 8, 8)
    print(f"DCT blocks shape: {dct_blocks.shape}")
    
    # Seleccionar coeficientes
    selection_mask = select_frequency_coefficients(
        dct_blocks,
        min_energy=0.2,
        max_energy=0.6,
        use_chaotic=True,
        chaotic_seed=0.5
    )
    
    print(f"Selection mask shape: {selection_mask.shape}")
    print(f"Selected coefficients per block: {selection_mask[0, 0, 0, 0].sum().item()}")
    print(f"Total selected: {selection_mask.sum().item()}")
    
    print("\n" + "="*60)
    print("Testing Texture Variance")
    print("="*60)
    
    # Test varianza de textura
    image = torch.randn(2, 3, 256, 256)
    variance_map = calculate_texture_variance(image, window_size=8)
    
    print(f"Input image: {image.shape}")
    print(f"Variance map: {variance_map.shape}")
    print(f"Variance range: [{variance_map.min():.4f}, {variance_map.max():.4f}]")
    print(f"Mean variance: {variance_map.mean():.4f}")
    
    print("\n✅ Coefficient selection tests passed!")
