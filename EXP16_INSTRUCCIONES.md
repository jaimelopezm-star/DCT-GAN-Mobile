# 🎯 Experimento 16: Dataset Real DIV2K

## 📋 Resumen

- **Base:** Configuración Exp 7 (ganador - 23.35 dB)
- **Cambio principal:** Dataset sintético → DIV2K real (800 imágenes)
- **Épocas:** 150 → 300
- **Expectativa:** 25-30 dB PSNR (+2-7 dB mejora)
- **Tiempo estimado:** 8-12 horas
- **Costo estimado:** $25-35 USD en RunPod

---

## 🚀 Instrucciones Paso a Paso

### **Paso 1: Iniciar RunPod**

1. Ve a [RunPod.io](https://runpod.io)
2. Selecciona GPU: **RTX 4090** (recomendado) o **RTX 4080**
3. Template: PyTorch 2.0+
4. Storage: **50 GB** mínimo (DIV2K ocupa ~4 GB)
5. Inicia el pod

---

### **Paso 2: Clonar Repositorio**

```bash
cd /workspace
git clone https://github.com/jaimelopezm-star/DCT-GAN-Mobile.git
cd DCT-GAN-Mobile
```

---

### **Paso 3: Descargar Dataset DIV2K**

DIV2K es un dataset de 1000 imágenes 2K de alta calidad:
- Training: 800 imágenes (~3.5 GB)
- Validation: 100 imágenes (~450 MB)

```bash
# Dar permisos de ejecución al script
chmod +x download_div2k.sh

# Descargar DIV2K (tarda ~10-15 minutos)
bash download_div2k.sh
```

**Salida esperada:**
```
✅ Training images: 800 (esperado: 800)
✅ Validation images: 100 (esperado: 100)
✅ Descarga completa y verificada!
```

---

### **Paso 4: Preparar Dataset**

Convertir DIV2K a estructura compatible con ImageFolder:

```bash
python prepare_div2k.py
```

**Esto tomará ~15-20 minutos** y procesará:
- 800 imágenes de training → 256×256px
- 100 imágenes de validación → 256×256px

**Salida esperada:**
```
✅ PREPARACIÓN COMPLETADA
📊 Resumen:
   Train: 800 imágenes
   Val: 100 imágenes
   Total: 900 imágenes

📁 Dataset preparado en: /workspace/DIV2K_prepared
```

---

### **Paso 5: Verificar Estructura**

```bash
ls -lh /workspace/DIV2K_prepared/train/images/ | head -5
ls -lh /workspace/DIV2K_prepared/val/images/ | head -5
```

**Estructura correcta:**
```
DIV2K_prepared/
  train/
    images/
      0001.png
      0002.png
      ...
      0800.png
  val/
    images/
      0001.png
      ...
      0100.png
```

---

### **Paso 6: Iniciar Entrenamiento**

```bash
python train.py \
    --config configs/exp16_real_dataset.yaml \
    --dataset imagenet \
    --dataset-path /workspace/DIV2K_prepared
```

**Parámetros importantes:**
- `--config`: Usa configuración Exp 16 (300 épocas, batch 32)
- `--dataset imagenet`: Activa loader de dataset real
- `--dataset-path`: Ruta al dataset preparado

---

### **Paso 7: Monitorear Entrenamiento**

El entrenamiento mostrará progreso cada 10 batches:

```
Epoch 1/300:
  [TRAIN] Loss_G: 7.05 | PSNR: 28.5 dB | SSIM: 0.992
  [VAL]   Loss_G: 7.02 | PSNR: 29.1 dB | SSIM: 0.993

Epoch 10/300:
  [TRAIN] Loss_G: 6.82 | PSNR: 24.3 dB | SSIM: 0.968
  [VAL]   Loss_G: 6.79 | PSNR: 25.1 dB | SSIM: 0.971
  
✅ New best model saved: 25.1 dB
```

**Métricas clave a observar:**
- **PSNR:** Debe mejorar gradualmente (meta: >25 dB)
- **SSIM:** Debe mantenerse >0.95
- **Loss_G:** Debe disminuir suavemente

**Early Stopping:**
- Si no mejora en **100 épocas**, se detendrá automáticamente
- Checkpoints se guardan cada **20 épocas**

---

### **Paso 8: Tiempos Estimados**

Con **RTX 4090**:
- Por época: ~2-3 minutos
- 100 épocas: ~3-5 horas
- 300 épocas completas: ~8-12 horas

Con **RTX 4080**:
- Por época: ~3-4 minutos  
- 100 épocas: ~5-7 horas
- 300 épocas completas: ~12-16 horas

**Costo RunPod (RTX 4090 @ $0.44/hr):**
- 300 épocas: $3.50 - $5.30

---

### **Paso 9: Evaluar Recovery**

Una vez completado el entrenamiento:

```bash
# Encontrar el mejor checkpoint
ls -lh checkpoints/

# Evaluar recovery con el mejor modelo
python evaluate_recovery.py \
    --checkpoint checkpoints/best_model.pth \
    --config configs/exp16_real_dataset.yaml
```

**Resultado esperado:**
```
📊 CALIDAD VISUAL (Cover vs Stego):
   PSNR: 26.5 dB  ← Meta: >25 dB
   SSIM: 0.970

🔓 RECUPERACIÓN DEL SECRETO:
   PSNR: 22.3 dB  ← Meta: >20 dB ✅
   SSIM: 0.881
```

---

## 📊 Criterios de Éxito

### ✅ **Éxito Completo:**
- Visual PSNR: **>26 dB** (+3 dB vs Exp 7)
- Recovery PSNR: **>20 dB** (mantiene funcionalidad)
- **Conclusión:** Dataset real ayuda, pero límite sigue siendo arquitectural

### ⚠️ **Éxito Parcial:**
- Visual PSNR: **24-26 dB** (+1-3 dB vs Exp 7)
- Recovery PSNR: **>20 dB**
- **Conclusión:** Mejora marginal, arquitectura es el cuello de botella

### ❌ **Sin Mejora:**
- Visual PSNR: **23-24 dB** (igual a Exp 7)
- **Conclusión:** Dataset NO es el problema, 100% arquitectural

---

## 🔧 Solución de Problemas

### **Problema: Error "dataset_path no encontrado"**
```bash
# Verificar que existe
ls /workspace/DIV2K_prepared/

# Si no existe, volver a Paso 4
python prepare_div2k.py
```

### **Problema: CUDA out of memory**
```yaml
# Editar configs/exp16_real_dataset.yaml
data:
  batch_size: 16  # Reducir de 32 a 16
```

### **Problema: Training muy lento**
- Verificar que `mixed_precision: true` en config
- Usar GPU más rápida (RTX 4090)
- Reducir `num_workers` si hay bottleneck de CPU

### **Problema: PSNR no mejora**
- Normal en primeras 50 épocas
- Esperar al menos 100 épocas
- Si después de 150 épocas no pasa de 24 dB, probablemente no mejorará

---

## 📁 Archivos Generados

```
DCT-GAN-Mobile/
  checkpoints/
    checkpoint_epoch_020.pth
    checkpoint_epoch_040.pth
    ...
    best_model.pth          ← El mejor modelo
  logs/
    training.log
    metrics.json
  configs/
    exp16_real_dataset.yaml  ← Configuración usada
```

---

## 📝 Siguiente Paso Después de Exp 16

### **Si mejora a 26-30 dB:**
→ Documentar resultados  
→ Comparar con Exp 7  
→ Considerar cambios arquitecturales para cerrar gap (58 dB meta)

### **Si no mejora (23-24 dB):**
→ **Confirma límite arquitectural**  
→ Opciones:
   - **Opción B:** Reforzar decoder (151K → 500K params) - 2-3 semanas
   - Aceptar 23 dB como baseline funcional y documentar hallazgos

---

## 💰 Resumen de Costos

| Concepto | Costo |
|----------|-------|
| Descarga DIV2K | Gratis (4 GB) |
| Preparación dataset | $0.10 (15-20 min @ RTX 4090) |
| Entrenamiento 300 épocas | $3.50-$5.30 (8-12 hrs @ RTX 4090) |
| Evaluación | $0.05 (5 min) |
| **TOTAL ESTIMADO** | **~$4-6 USD** |

---

## 🎯 Expectativa Realista

Basado en Exp 1-15, el **escenario más probable** es:

**Visual PSNR: 24-26 dB** (+1-3 dB mejora)  
**Recovery PSNR: >20 dB** (funcional)

Esto confirmaría que:
- Dataset real ayuda, pero marginalmente
- **El verdadero límite es el decoder débil (151K params)**
- Para llegar a 35-40 dB se necesita arquitectura nueva
- Para llegar a 58 dB (paper) se necesita investigación profunda

---

**¿Listo para empezar?** 🚀

Ejecuta en RunPod:
```bash
cd /workspace
git clone https://github.com/jaimelopezm-star/DCT-GAN-Mobile.git
cd DCT-GAN-Mobile
bash download_div2k.sh
```
