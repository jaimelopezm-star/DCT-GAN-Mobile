"""
Análisis de configuración de parámetros para DCT-GAN
Objetivo: Determinar configuración óptima de canales para ~50K parámetros
"""

def calculate_conv_params(in_channels, out_channels, kernel_size, has_bias=False):
    """Calcula parámetros de una capa convolucional"""
    kernel_params = in_channels * out_channels * kernel_size * kernel_size
    bias_params = out_channels if has_bias else 0
    return kernel_params + bias_params

def calculate_resnet_encoder_params(base_channels, num_blocks=9):
    """
    Calcula params del encoder ResNet
    Estructura (según paper):
    - Conv 6→base_channels (3×3, padding=1)
    - num_blocks × ResidualBlock(base_channels)
    - Conv base_channels→3 (3×3, padding=1)
    """
    # Primera convolución: 6→base_channels
    params = calculate_conv_params(6, base_channels, 3, has_bias=False)
    
    # Bloques residuales: cada uno tiene 2 conv (channels→channels)
    for _ in range(num_blocks):
        # Conv1: channels→channels (3×3)
        params += calculate_conv_params(base_channels, base_channels, 3, has_bias=False)
        # Conv2: channels→channels (3×3)
        params += calculate_conv_params(base_channels, base_channels, 3, has_bias=False)
    
    # Última convolución: base_channels→3
    params += calculate_conv_params(base_channels, 3, 3, has_bias=False)
    
    return params

def calculate_cnn_decoder_params(base_channels, num_layers=6):
    """
    Calcula params del decoder CNN
    Estructura:
    - Conv 3→base_channels (3×3)
    - (num_layers-2) × Conv base_channels→base_channels (3×3)
    - Conv base_channels→3 (3×3)
    """
    # Primera capa: 3→base_channels
    params = calculate_conv_params(3, base_channels, 3, has_bias=False)
    
    # Capas intermedias: base_channels→base_channels
    for _ in range(num_layers - 2):
        params += calculate_conv_params(base_channels, base_channels, 3, has_bias=False)
    
    # Última capa: base_channels→3
    params += calculate_conv_params(base_channels, 3, 3, has_bias=False)
    
    return params

def calculate_xunet_discriminator_params(base_channels, num_conv_layers=5):
    """
    Calcula params del discriminador XuNet
    Estructura simplificada:
    - 5 capas conv con stride-2 downsampling
    - Channels: base→2×base→4×base→8×base→16×base
    - FC: (16×base × 4 × 4) → 1
    """
    channels_progression = [3]  # Input: 3 channels RGB
    current_channels = base_channels
    
    for i in range(num_conv_layers):
        channels_progression.append(current_channels)
        if i < num_conv_layers - 1:  # No doblar en la última capa
            current_channels = min(current_channels * 2, 64)  # Max 64 channels
    
    # Convoluciones
    params = 0
    for i in range(len(channels_progression) - 1):
        # Stride-2 conv de 3×3 (excepto última que es stride-1)
        params += calculate_conv_params(
            channels_progression[i], 
            channels_progression[i+1], 
            3,  # kernel 3×3
            has_bias=True
        )
    
    # Fully connected: después de 5 conv con stride-2, 256×256 → 8×8
    # Luego pooling → 4×4 (aprox)
    final_spatial = 4
    fc_input = channels_progression[-1] * final_spatial * final_spatial
    params += fc_input * 1  # FC → 1 output
    params += 1  # bias
    
    return params

# ============================================================================
# ANÁLISIS: Probar diferentes configuraciones
# ============================================================================

print("=" * 80)
print("ANÁLISIS DE CONFIGURACIONES PARA ~50K PARÁMETROS")
print("=" * 80)
print()

target_params = 49_950  # Del paper (presentación página 6)

test_configs = [
    # (encoder_channels, decoder_channels, disc_channels)
    (16, 16, 8),   # Configuración actual
    (14, 14, 6),   # Más reducido
    (12, 12, 5),   # Aún más reducido
    (10, 10, 4),   # Muy reducido
    (12, 14, 5),   # Balanceado 1
    (14, 12, 6),   # Balanceado 2
    (10, 12, 5),   # Balanceado 3
    (8, 10, 4),    # Mínimo
]

print("Config                  Encoder    Decoder    Discrim     Total    Diff")
print("-" * 80)

best_config = None
best_diff = float('inf')

for enc_ch, dec_ch, disc_ch in test_configs:
    enc_params = calculate_resnet_encoder_params(enc_ch, num_blocks=9)
    dec_params = calculate_cnn_decoder_params(dec_ch, num_layers=6)
    disc_params = calculate_xunet_discriminator_params(disc_ch, num_conv_layers=5)
    
    total = enc_params + dec_params + disc_params
    diff = total - target_params
    
    print(f"Enc:{enc_ch:2d} Dec:{dec_ch:2d} Disc:{disc_ch:2d}    "
          f"{enc_params:7,}    {dec_params:7,}    {disc_params:7,}    "
          f"{total:7,}    {diff:+7,}")
    
    if abs(diff) < abs(best_diff):
        best_diff = diff
        best_config = (enc_ch, dec_ch, disc_ch, enc_params, dec_params, disc_params, total)

print("-" * 80)
print()

if best_config:
    enc_ch, dec_ch, disc_ch, enc_p, dec_p, disc_p, total_p = best_config
    print(f"MEJOR CONFIGURACIÓN ENCONTRADA:")
    print(f"  Encoder base_channels: {enc_ch}")
    print(f"  Decoder base_channels: {dec_ch}")
    print(f"  Discriminator base_channels: {disc_ch}")
    print()
    print(f"  Parámetros:")
    print(f"    Encoder:        {enc_p:7,} ({enc_p/target_params*100:.1f}%)")
    print(f"    Decoder:        {dec_p:7,} ({dec_p/target_params*100:.1f}%)")
    print(f"    Discriminator:  {disc_p:7,} ({disc_p/target_params*100:.1f}%)")
    print(f"    Total:          {total_p:7,}")
    print()
    print(f"  Diferencia con target ({target_params:,}): {best_diff:+,} ({best_diff/target_params*100:+.2f}%)")
    print()
    
print("=" * 80)
print("CONFIGURACIÓN ACTUAL vs TARGET")
print("=" * 80)

current_enc = calculate_resnet_encoder_params(16, 9)
current_dec = calculate_cnn_decoder_params(16, 6)
current_disc = calculate_xunet_discriminator_params(8, 5)
current_total = current_enc + current_dec + current_disc

print(f"Configuración Actual (Enc:16, Dec:16, Disc:8):")
print(f"  Encoder:        {current_enc:7,}")
print(f"  Decoder:        {current_dec:7,}")
print(f"  Discriminator:  {current_disc:7,}")
print(f"  Total:          {current_total:7,}")
print(f"  Target:         {target_params:7,}")
print(f"  Exceso:         {current_total - target_params:+7,} ({(current_total - target_params)/target_params*100:+.1f}%)")
print()

# Análisis del discriminador (principal problema)
print("=" * 80)
print("ANÁLISIS DETALLADO DEL DISCRIMINADOR")
print("=" * 80)
print()
print("El discriminador actual tiene más parámetros de lo esperado.")
print("Posibles causas:")
print("  1. Demasiados canales en capas intermedias")
print("  2. Fully connected layer muy grande")
print("  3. Necesita versión más ligera de XuNet")
print()

for disc_ch in [8, 6, 5, 4, 3, 2]:
    disc_p = calculate_xunet_discriminator_params(disc_ch, 5)
    print(f"  base_channels={disc_ch}: {disc_p:6,} params")

print()
print("NOTA: El paper menciona 'XuNet modificado para 3 canales'.")
print("Posiblemente usa una versión MUY simplificada con menos capas o canales.")
print()
