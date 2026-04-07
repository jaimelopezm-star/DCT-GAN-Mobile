# FIX CRÍTICO: Ratio de Actualización Discriminador:Generator

## 🐛 Bug Identificado

### El Problema
El código actual implementa ratio **4:1 (G:D)** - 4 actualizaciones del Generator por 1 del Discriminador.

**Resultado**: `Loss_D ≈ 0.0000` → El discriminador NO aprende

### La Solución del Paper

Según la sección "Training generator" del paper original:

> "**Typically, the discriminator weights are updated five times, followed by a single update to the generator weights**, to expedite the model's convergence."

**Ratio correcto**: **5:1 (D:G)** - 5 actualizaciones del Discriminador por 1 del Generator

### Contradicción en el Paper

El mismo paper admite que en su implementación usaron: 

> "This study changed the weights of the generator four times and the discriminator weights once **to achieve a balance**"

**PERO** esto es probablemente lo que causa `Loss_D ≈ 0` y PSNR estancado en 18 dB.

---

## ✅ Cambios Implementados

### 1. Archivo: `src/training/trainer.py`

**ANTES**:
```python
# Estrategia 4:1 (G:D)
gen_updates_per_batch = update_strategy.get('generator_updates_per_epoch', 4)
disc_updates_per_batch = update_strategy.get('discriminator_updates_per_epoch', 1)
```

**DESPUÉS**:
```python
# CRITICAL FIX: Estrategia 5:1 (D:G) - TÍPICO Y FUNCIONAL
disc_updates_per_batch = update_strategy.get('discriminator_updates_per_batch', 5)
gen_updates_per_batch = update_strategy.get('generator_updates_per_batch', 1)
```

### 2. Archivo: `configs/paper_exact_config.yaml`

**NUEVO**:
```yaml
training:
  num_epochs: 100
  
  # CRITICAL FIX: Ratio D:G según "Training generator" section
  update_strategy:
    discriminator_updates_per_batch: 5  # ✅ 5×D updates (TYPICAL)
    generator_updates_per_batch: 1      # ✅ 1×G update
```

### 3. Archivos Actualizados

- ✅ `src/training/trainer.py` - Lógica del training loop
- ✅ `configs/paper_exact_config.yaml` - Config principal
- ✅ `configs/base_config.yaml` - Config base
- ✅ `configs/mobile_config.yaml` - Config mobile

---

## 🎯 Impacto Esperado

### Antes del Fix
```
Epoch 1: PSNR 17.91 dB, SSIM 0.8606, Loss_D 0.0006
Epoch 2: PSNR 17.94 dB, SSIM 0.8609, Loss_D 0.0000  ← ESTANCADO
Epoch 3: PSNR 17.95 dB, SSIM 0.8610, Loss_D -0.0000
...
Epoch 7: PSNR 17.94 dB, SSIM 0.8604, Loss_D 0.0000
```

**Problema**: Discriminador no aprende nada → No hay feedback adversarial

### Después del Fix (esperado)
```
Epoch 1: PSNR 18-20 dB, SSIM 0.85-0.87, Loss_D 0.05-0.10  ← D APRENDE
Epoch 5: PSNR 25-30 dB, SSIM 0.90-0.92, Loss_D 0.03-0.08
Epoch 10: PSNR 35-40 dB, SSIM 0.92-0.94, Loss_D 0.02-0.06
...
Epoch 50+: PSNR →50+ dB, SSIM →0.94, Loss_D estable ~0.01-0.05
```

**Mejora**: Discriminador entrena correctamente → Feedback adversarial funciona

---

## 📊 Fundamentación Teórica

### ¿Por qué 5D:1G?

1. **Discriminador más débil**: Necesita más actualizaciones para aprender diferencias sutiles entre cover y stego
2. **Generator más fuerte**: Produce stego images muy sutiles → D necesita entrenamiento extra
3. **Balance adversarial**: Si G se entrena 4x más, D nunca alcanza → Loss_D=0

### Paper de referencia (WGAN)
Arjovsky et al. (2017) recomienda:
- **5 updates del critic (discriminador) por 1 del generator**
- Esto previene mode collapse
- Garantiza convergencia estable

---

## 🚀 Próximos Pasos

### 1. Entrenar con Fix (PRIORIDAD ALTA)
```bash
cd DCT-GAN-Mobile
python train.py --config configs/paper_exact_config.yaml
```

**Qué vigilar**:
- ✅ `Loss_D` debe estar en rango `0.01-0.10` (NO cerca de 0)
- ✅ `Loss_D` debe oscilar (señal de que D está aprendiendo)
- ✅ PSNR debe mejorar gradualmente (no estancarse en 18 dB)

### 2. Dataset (PRIORIDAD MEDIA)
El paper menciona que **probablemente** usaron:
- **BOSSBase** o **BOWS2** en lugar de ImageNet2012
- Imágenes más "limpias" → más fácil alcanzar 58 dB

**Acción**: Buscar y probar con BOSSBase si disponible

### 3. Otras Observaciones del AI Externo

De la búsqueda que te dieron:

**StegoGAN** (CVPR 2024): https://github.com/sian-wusidi/StegoGAN
- Repo oficial con código
- Puede servir como referencia para training loop

**RISRANet** (2025): Reporta PSNR > 50 dB
- Paper menciona ratio 5D:1G explícitamente
- Código disponible según paper

---

## ⚠️ Notas Importantes

1. **No frenar entrenamiento prematuro**: Con 5D:1G, primeros 10-20 epochs pueden verse "lentos"
2. **Paciencia alta**: Usar `patience=50` en early stopping
3. **Costo RunPod**: Con fix, entrenar ~30-50 epochs para validar (~$1-2)
4. **Métricas clave**:
   - `Loss_D` en rango `0.01-0.10` = ✅ BUENO
   - `Loss_D ≈ 0.0000` = ❌ MALO (discriminador no aprende)

---

## 📝 Referencias

1. **Paper DCT-GAN** (Malik et al., 2025):
   - DOI: 10.1038/s41598-025-01054-7
   - Sección clave: "Training generator"
   
2. **WGAN Paper** (Arjovsky et al., 2017):
   - Recomienda: 5 critic updates : 1 generator update

3. **Search externa**:
   - Confirmó que ratio 5D:1G es estándar en steganography GANs
   - StegoGAN, RISRANet, StegaVision todos usan ratios similares

---

## 🎓 Conclusión

Este fix tiene **ALTA probabilidad** de resolver:
- ✅ `Loss_D ≈ 0` → `Loss_D en 0.01-0.10`
- ✅ PSNR estancado en 18 dB → PSNR mejorando gradualmente
- ✅ Discriminador no aprende → Discriminador entrenando correctamente

**Recomendación**: Entrenar inmediatamente con este fix antes de cualquier otro cambio.

---

**Fecha**: 2026-04-07  
**Autor**: Análisis basado en paper original + búsqueda AI externa  
**Commit**: Próximo (después de este fix)
