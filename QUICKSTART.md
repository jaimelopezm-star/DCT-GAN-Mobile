# 🚀 Quick Start - Replicación Paper (Revisión 1)

## ✅ Estado Actual

**Código: 100% LISTO** ✅
- Arquitecturas implementadas (45,998 params)
- Training pipeline completo
- Métricas implementadas
- Scripts de dataset listos

**Dataset: PENDIENTE** ⏳
- Necesitas descargar ImageNet 2012
- ~6.3 GB, 1-2 horas

---

## ⚡ SETUP AUTOMÁTICO (RECOMENDADO)

### Una línea, hace todo:

```powershell
# Testing + Descarga en paralelo
.\quick_setup.ps1
```

**Esto hace automáticamente:**
1. ✅ Testing rápido (2 epochs sintético) → 5-10 min
2. 🔽 Descarga ImageNet en nueva ventana → 1-2 horas
3. 📦 Prepara splits automáticamente
4. ✅ Todo listo para entrenar

**Ventajas:**
- En 10 minutos sabes si hay errores
- Descarga corre en background
- No bloquea tu trabajo
- Prepara todo solo

**Ver detalles:** [docs/QUICK_SETUP_GUIDE.md](docs/QUICK_SETUP_GUIDE.md)

---

## 📥 SETUP MANUAL (Paso a Paso)

### PASO 1: Descargar ImageNet (~1-2 horas)

### Opción A: Kaggle (Automático - RECOMENDADO)

```powershell
# 1. Crear cuenta gratuita en Kaggle
# Ir a: https://www.kaggle.com

# 2. Aceptar reglas de ImageNet
# Ir a: https://www.kaggle.com/c/imagenet-object-localization-challenge
# Click en "Join Competition"

# 3. Descargar API token
# Ir a: https://www.kaggle.com/account
# Scroll a "API" → "Create New API Token"
# Guardar kaggle.json en C:\Users\Lopez\.kaggle\

# 4. Ejecutar descarga
cd "DCT-GAN-Mobile"
python scripts/download_imagenet.py --method kaggle
```

**Esto descarga y organiza todo automáticamente** (6.3 GB)

---

### Opción B: Dataset Reducido (Para testing rápido)

Si quieres solo probar que funciona antes de descargar todo:

```powershell
# Crear subset sintético de prueba (solo para validar código)
python train.py --config configs/base_config.yaml --dataset synthetic
```

⚠️ **NO replicará métricas del paper**, solo valida que el código funciona.

---

## 📦 PASO 2: Preparar Splits (10 minutos)

```powershell
# Después de descargar ImageNet, crear splits train/val/test
python scripts/prepare_dataset.py

# Resultado esperado:
# data/imagenet2012/splits/
# ├── train/all/    40,000 imágenes ✅
# ├── val/all/       5,000 imágenes ✅
# └── test/all/      5,000 imágenes ✅
```

---

## 🏋️ PASO 3: Entrenar Modelo (2-7 días)

```powershell
# Entrenar con ImageNet (GPU recomendado)
python train.py --config configs/base_config.yaml --dataset imagenet

# Los checkpoints se guardan en: checkpoints/
# Los logs en: logs/
```

**Tiempo estimado:**
- **GPU (RTX 3080):** 12-24 horas ⚡
- **GPU (GTX 1060):** 2-3 días 🐢
- **CPU:** ~7 días 🐌 (NO recomendado)

**¿No tienes GPU?** Usa Google Colab gratuito:
1. Subir código a Google Drive
2. Abrir Colab: https://colab.research.google.com
3. Runtime → Change runtime type → GPU
4. Ejecutar entrenamiento

---

## 📊 PASO 4: Monitorear Entrenamiento

```powershell
# Ver progreso en tiempo real
Get-Content logs\training_*.log -Tail 50 -Wait

# Buscar métricas clave:
# - PSNR: debe subir a ~58 dB
# - SSIM: debe subir a ~0.94
# - Loss: debe bajar y estabilizarse
```

**Early stopping automático:**
- Si PSNR no mejora en 20 epochs → para entrenamiento
- Best model guardado en: `checkpoints/best_model.pth`

---

## ✅ PASO 5: Validar Resultados

Cuando termine el entrenamiento:

```powershell
# Ver mejor checkpoint
ls checkpoints\best_model.pth

# Verificar métricas finales
# (buscar en logs la última validación)
Select-String -Path "logs\training_*.log" -Pattern "Best PSNR"
```

**Métricas objetivo:**
- ✅ PSNR ≥ 58 dB (paper: 58.27 dB)
- ✅ SSIM ≥ 0.94 (paper: 0.942)
- ✅ Recovery accuracy ≥ 95%

---

## 🎯 Resumen de Comandos

```powershell
# Todo en secuencia (copiar y pegar)

# 1. Descargar dataset
python scripts/download_imagenet.py --method kaggle

# 2. Crear splits
python scripts/prepare_dataset.py

# 3. Entrenar
python train.py --config configs/base_config.yaml --dataset imagenet

# 4. (Esperar 2-7 días...)

# 5. Verificar resultados
Select-String -Path "logs\training_*.log" -Pattern "PSNR"
```

---

## 🆘 Solución de Problemas

### Error: "No module named 'kaggle'"
```powershell
& .venv\Scripts\pip.exe install kaggle
```

### Error: "kaggle.json not found"
1. Ir a https://www.kaggle.com/account
2. "Create New API Token"
3. Guardar en: `C:\Users\Lopez\.kaggle\kaggle.json`

### Error: "CUDA out of memory"
```powershell
# Reducir batch size en configs/base_config.yaml
# Cambiar batch_size: 32 → 16 o 8
```

### Training muy lento en CPU
**Solución:** Usar Google Colab con GPU gratuita
1. Subir código a Drive
2. Colab → GPU runtime
3. Entrenar ahí

---

## 📋 Checklist Completo

### Preparación
- [ ] Cuenta Kaggle creada
- [ ] Reglas de ImageNet aceptadas
- [ ] API token descargado (kaggle.json)

### Dataset
- [ ] ImageNet descargado (6.3 GB)
- [ ] Splits creados (40K/5K/5K)
- [ ] Verificado con `ls data\imagenet2012\splits\train\all\`

### Entrenamiento
- [ ] GPU configurada (o Colab)
- [ ] Training iniciado (`python train.py ...`)
- [ ] Logs monitoreados
- [ ] Checkpoints guardándose cada 10 epochs

### Validación
- [ ] 100 epochs completados
- [ ] PSNR ≥ 58 dB alcanzado
- [ ] SSIM ≥ 0.94 alcanzado
- [ ] Best model guardado

### Documentación
- [ ] Screenshots de métricas
- [ ] Gráficas de training
- [ ] Presentación actualizada

---

## 🎓 Siguiente Paso

Una vez alcanzadas las métricas:

**Revisión 1:** COMPLETADA ✅
→ Presentar resultados con métricas reales

**Revisión 2:** Agregar BOSSBase para validación robusta
→ Ver [ROADMAP.md](ROADMAP.md) para detalles

---

## 💡 Tips

1. **Empieza con synthetic data** para validar que el código funciona
   ```powershell
   python train.py --dataset synthetic  # 5 minutos
   ```

2. **Luego descarga ImageNet** mientras haces otras cosas (1-2 horas en background)

3. **Entrena en GPU** (Colab gratis si no tienes GPU local)

4. **Monitorea cada 1-2 horas** para verificar que PSNR sube

5. **Ten paciencia:** 100 epochs toman días, pero es normal

---

**¿Listo?** Comienza con:
```powershell
python scripts/download_imagenet.py --method kaggle
```

🚀 ¡Éxito en la replicación!
