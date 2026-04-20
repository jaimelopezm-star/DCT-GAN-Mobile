# EXP 18: SteganoGAN-Style Dense Architecture

## Cambio Metodológico

Después de 17 experimentos fallidos, analizamos **SteganoGAN del MIT** que sí funciona (~37 dB PSNR con recovery exitoso).

### Diferencias Clave Identificadas

| Aspecto | Nuestros Exp 1-17 | SteganoGAN MIT | Exp 18 |
|---------|-------------------|----------------|--------|
| **MSE Weight** | 0.3 - 1.5 | **100** | **100** |
| **Recovery Weight** | 10 - 15 | **1** | **1** |
| **Encoder** | Residual simple | Dense connections | Dense |
| **Decoder** | CNN/U-Net simple | Dense connections | Dense |
| **Dominio** | DCT | Espacial | Espacial |

### ¿Por qué fallamos antes?

1. **Loss weights invertidos**: Nosotros priorizábamos recovery (15×) sobre MSE (0.3×)
   - SteganoGAN hace lo opuesto: 100× MSE, 1× recovery
   
2. **Sin dense connections**: Nuestro encoder usaba residual simple
   - SteganoGAN concatena features de todas las capas anteriores

3. **DCT complicando**: El dominio DCT puede dificultar la recuperación
   - SteganoGAN trabaja directamente en RGB

---

## Comandos para RunPod

### 1. Setup inicial
```bash
cd /workspace
git clone https://github.com/gusseppe/DCT-GAN-Mobile.git
cd DCT-GAN-Mobile
pip install -r requirements_gpu.txt
```

### 2. Descargar DIV2K (si no está)
```bash
bash download_div2k.sh
python prepare_div2k.py
```

### 3. Entrenar Exp 18
```bash
# Opción A: Con archivo de configuración
python train_dense.py --config configs/exp18_steganogan_style.yaml

# Opción B: Con argumentos directos
python train_dense.py \
    --epochs 200 \
    --batch_size 8 \
    --lr 1e-4 \
    --mse_weight 100.0 \
    --rec_weight 1.0 \
    --encoder_type dense \
    --decoder_type dense \
    --hidden_size 64 \
    --checkpoint_dir checkpoints/exp18_steganogan \
    --patience 100
```

### 4. Monitorear progreso
```bash
# Ver logs en tiempo real
tail -f checkpoints/exp18_steganogan/training.log

# Ver último checkpoint
ls -la checkpoints/exp18_steganogan/
```

---

## Criterios de Éxito

| Métrica | Target | Exp 7 (mejor anterior) | SteganoGAN ref |
|---------|--------|------------------------|----------------|
| Visual PSNR | ≥ 30 dB | 23.35 dB | ~37 dB |
| Recovery PSNR | ≥ 20 dB | ~20 dB | exitoso |

**Éxito = Visual ≥ 30 dB Y Recovery ≥ 20 dB**

---

## Arquitectura Exp 18

### DenseEncoder
```
Input: cover(3) + secret(3) = 6 channels

Conv1: 6 → 64 + LeakyReLU + BN
Conv2: 64+6 → 64 + LeakyReLU + BN  (dense: concat input)
Conv3: 64+64+6 → 64 + LeakyReLU + BN  (dense: concat all)
Conv4: 64+64+64+6 → 3 + Tanh

Output: stego = cover + output (residual)
```

### DenseDecoder
```
Input: stego (3 channels)

Conv1: 3 → 64 + LeakyReLU + BN
Conv2: 64 → 64 + LeakyReLU + BN
Conv3: 64+64 → 64 + LeakyReLU + BN  (dense)
Conv4: 64+64+64 → 3 + Sigmoid

Output: recovered secret [0,1]
```

### Loss Function
```python
loss = 100 * MSE(cover, stego) + 1 * MSE(secret, recovered)
#      ^^^                       ^^^
#      Alta prioridad visual     Prioridad estándar recovery
```

---

## Comparación con Paper Original

El paper de Malik et al. reporta 58.27 dB PSNR, pero:
- No hay código disponible
- SteganoGAN (código abierto) logra ~37 dB con recovery funcional
- 37 dB es un excelente resultado para esteganografía

**Meta realista: 30-40 dB con recovery ≥ 20 dB**

---

## Siguiente Paso si Funciona

Si Exp 18 logra recovery exitoso:
1. Añadir discriminador (Exp 19)
2. Probar con DCT sobre esta base (Exp 20)
3. Optimizar para móviles (Exp 21)

Si falla, el problema es más profundo en nuestra implementación base.
