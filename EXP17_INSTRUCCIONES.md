# 🚀 EXPERIMENTO 17: STRONG DECODER - ÚLTIMO INTENTO

## Resumen de Cambios

| Parámetro | Exp 7 (baseline) | Exp 16 | Exp 17 (este) |
|-----------|------------------|--------|---------------|
| **Decoder** | CNN 151K | CNN 151K | **U-Net ~1.2M** |
| **residual_scale** | 0.1 | 0.1 | **0.2** |
| **alpha** | 2.0 | 2.0 | **1.5** |
| **beta** | 10.0 | 10.0 | **15.0** |
| **Dataset** | Sintético | DIV2K | DIV2K |
| **Recovery** | ~20 dB ✅ | -0.25 dB ❌ | ??? |

## Hipótesis

1. El decoder débil (151K params) no puede extraer el secreto
2. residual_scale=0.1 es muy pequeño para que el decoder lo detecte
3. Con decoder 8× más grande + señal 2× más fuerte → recovery funcional

---

## Instrucciones en RunPod

### 1. Actualizar Código (ya tienes el dataset)

```bash
cd /workspace/DCT-GAN-Mobile
git pull origin main
```

### 2. Verificar que el nuevo decoder existe

```bash
python -c "
from src.models.decoder import StrongDecoder
d = StrongDecoder()
print(f'StrongDecoder params: {d.get_num_params():,}')
"
```

**Esperado:** ~1,200,000 params (vs 151,000 del anterior)

### 3. Iniciar Entrenamiento

```bash
python train.py --config configs/exp17_strong_decoder.yaml
```

### 4. Monitorear (en otra terminal)

```bash
watch -n 30 "tail -5 logs/training.log"
```

---

## Qué Observar Durante el Entrenamiento

### Señales BUENAS:
- Loss_G disminuye gradualmente
- PSNR visual ~25-30 dB (menor que Exp 16 está BIEN)
- SSIM visual ~0.85-0.90

### Señales MALAS:
- NaN en cualquier métrica → parar y reducir LR
- Loss_G explota (>100) → parar
- PSNR < 15 dB → posible inestabilidad

---

## Evaluación Post-Entrenamiento

```bash
python evaluate_recovery.py \
    --checkpoint checkpoints/best_model.pth \
    --config configs/exp17_strong_decoder.yaml
```

### Criterios de Éxito:

| Métrica | Mínimo | Ideal |
|---------|--------|-------|
| Visual PSNR | >20 dB | >25 dB |
| **Recovery PSNR** | **>10 dB** | **>20 dB** |
| Recovery SSIM | >0.3 | >0.7 |

**Lo importante es Recovery PSNR**, no visual.

---

## Comparación con Paper

| Modelo | Visual PSNR | Recovery PSNR | ¿Funcional? |
|--------|-------------|---------------|-------------|
| Paper | 58.27 dB | ~58 dB | ✅ |
| Exp 7 | 23.35 dB | ~20 dB | ✅ |
| Exp 16 | 33.56 dB | -0.25 dB | ❌ |
| **Exp 17** | ??? | ??? | ??? |

---

## Si Exp 17 Falla

Si recovery < 10 dB, entonces:

1. **El problema es arquitectural profundo** - no es solo el decoder
2. **Posible bug en encoder** - el residual no se está calculando bien
3. **Aceptamos Exp 7** como máximo alcanzable

En ese caso, documentamos:
- "Nuestra implementación logra 23 dB (40% del paper)"
- "Limitaciones identificadas: arquitectura del decoder"
- "Trabajo futuro: implementar arquitectura exacta del paper"

---

## Tiempo Estimado

- Época: ~15-20 segundos (decoder más grande)
- Total 300 épocas: ~90-120 minutos (~1.5-2 horas)
- Costo RunPod (RTX 4090): ~$1.50-2.00

---

## Comandos Rápidos

```bash
# Todo en uno:
cd /workspace/DCT-GAN-Mobile && \
git pull origin main && \
python train.py --config configs/exp17_strong_decoder.yaml

# Evaluación:
python evaluate_recovery.py \
    --checkpoint checkpoints/best_model.pth \
    --config configs/exp17_strong_decoder.yaml
```
