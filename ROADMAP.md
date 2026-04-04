# Roadmap DCT-GAN Mobile - Revisiones

## 🎯 Estrategia General

**Revisión 1 (Actual):** Replicación Mínima del Paper
- Dataset: ImageNet 2012 (50K imágenes)
- Objetivo: PSNR 58.27 dB, SSIM 0.942
- Resultado: Validar que la arquitectura funciona

**Revisión 2 (Siguiente):** Validación Robusta
- Dataset adicional: BOSSBase 1.01
- Objetivo: Resistencia a steganalysis
- Resultado: Probar robustez del modelo

---

## 📅 Revisión 1 - Replicación del Paper (Actual)

### ✅ Completado (80%)

**1. Arquitecturas Implementadas**
- [x] ResNet Encoder (17,010 params)
- [x] CNN Decoder (4,143 params)
- [x] XuNet Discriminator (24,845 params)
- [x] Total: 45,998 params (-7.9% vs paper ✅)

**2. Módulo DCT**
- [x] transform.py (DCT/IDCT 2D)
- [x] coefficients.py (selección de frecuencias)
- [x] embedding.py (LSB reference)

**3. Training Components**
- [x] trainer.py (500 líneas, estrategia 4:1)
- [x] metrics.py (PSNR, SSIM, RMSE, BER, Accuracy)
- [x] losses.py (HybridLoss: Ecuación 5)
- [x] train.py (script principal)

**4. Tests y Validación**
- [x] Tests de modelos (8/8 passing)
- [x] Tests de DCT (error < 1e-6)
- [x] Tests de métricas (PSNR, SSIM working)

### ⏳ En Progreso (20%)

**5. Dataset Preparation**
- [x] Scripts de descarga (download_imagenet.py)
- [x] Scripts de splits (prepare_dataset.py)
- [x] Dataset loader (SteganographyDataset)
- [ ] **SIGUIENTE:** Descargar ImageNet 2012

**6. Entrenamiento**
- [ ] Ejecutar 100 epochs
- [ ] Alcanzar PSNR ~58 dB
- [ ] Validar SSIM ~0.942
- [ ] Guardar best model

---

## 🚀 Pasos para Completar Revisión 1

### Paso 1: Preparar Dataset (1-2 días)

```bash
# Opción A: Descarga automática (Kaggle)
python scripts/download_imagenet.py --method kaggle

# Opción B: Descarga manual
python scripts/download_imagenet.py --method manual
# (Seguir instrucciones en pantalla)

# Crear splits (40K/5K/5K)
python scripts/prepare_dataset.py
```

**Requisitos:**
- Cuenta gratuita en Kaggle
- ~8 GB espacio en disco
- 1-2 horas tiempo de descarga

**Resultado esperado:**
```
data/imagenet2012/splits/
├── train/all/    (40,000 imágenes)
├── val/all/      (5,000 imágenes)
└── test/all/     (5,000 imágenes)
```

---

### Paso 2: Entrenar Modelo (3-7 días)

```bash
# Entrenar con ImageNet
python train.py --config configs/base_config.yaml --dataset imagenet

# Monitorear entrenamiento
# - Ver logs en: logs/training_*.log
# - Checkpoints en: checkpoints/
```

**Configuración:**
- 100 epochs
- Batch size: 32
- Estrategia: 4:1 (Generator:Discriminator)
- Hardware: CPU (lento) o GPU (recomendado)

**Tiempo estimado:**
- CPU: ~7 días (no recomendado)
- GPU (GTX 1060): ~2-3 días
- GPU (RTX 3080): ~12-24 horas

**Métricas objetivo:**
- PSNR: 58.27 dB
- SSIM: 0.942
- Loss estable

---

### Paso 3: Validar Resultados (1 día)

```bash
# Evaluar mejor modelo
python test.py --checkpoint checkpoints/best_model.pth --dataset imagenet

# Generar visualizaciones
python visualize_results.py --checkpoint checkpoints/best_model.pth
```

**Validaciones:**
- ✅ PSNR ≥ 58 dB (objetivo: 58.27 dB)
- ✅ SSIM ≥ 0.94 (objetivo: 0.942)
- ✅ Calidad visual (inspección manual)
- ✅ Recovery accuracy ≥ 95%

---

### Paso 4: Documentar Resultados (1 día)

**Crear:**
- Gráficas de entrenamiento (loss curves, PSNR evolution)
- Tabla comparativa (paper vs nuestro modelo)
- Ejemplos visuales (cover, stego, recovered)
- Actualizar presentación con métricas reales

**Archivos a actualizar:**
- `PROGRESS_LOG.md` (marcar Fase 1 completa)
- `revisiones/revision_1_marzo_2026/RESULTADOS.md`
- `CONTENIDO_PRESENTACION.md` (Diapositiva 15: métricas reales)

---

## 📊 Revisión 2 - Validación Robusta (Siguiente)

### Objetivo

Validar que el modelo es robusto contra:
- Steganalysis (XuNet, SR-Net)
- Compresión JPEG (95% accuracy @ Q=50)
- Ataques comunes

### Dataset Adicional

**BOSSBase 1.01**
```bash
# Descargar BOSSBase
python scripts/download_bossbase.py

# Evaluar modelo en BOSSBase
python test.py --checkpoint checkpoints/best_model.pth --dataset bossbase
```

### Tests Adicionales

**1. Robustez JPEG**
```bash
python test_jpeg_robustness.py --checkpoint best_model.pth
```
- Comprimir stego con JPEG (Q=50, 70, 90)
- Medir recovery accuracy
- Objetivo: >95% @ Q=50

**2. Steganalysis Resistance**
```bash
python test_steganalysis.py --checkpoint best_model.pth
```
- Entrenar XuNet detector
- Medir detection accuracy
- Objetivo: <55% (random guess = 50%)

**3. WhatsApp Simulation**
```bash
python test_whatsapp.py --checkpoint best_model.pth
```
- Simular pipeline de WhatsApp
- Validar recuperación después de compresión
- Objetivo: PSNR >50 dB después de WhatsApp

---

## 📈 Criterios de Éxito

### Revisión 1: COMPLETADA si:
- ✅ PSNR ≥ 58 dB (target: 58.27 dB)
- ✅ SSIM ≥ 0.94 (target: 0.942)
- ✅ Parámetros: ~50K (actual: 45,998 ✅)
- ✅ Recovery accuracy ≥ 95%
- ✅ Training converge (loss estable)

### Revisión 2: COMPLETADA si:
- ✅ JPEG robustness ≥ 95% @ Q=50
- ✅ XuNet detection <55%
- ✅ WhatsApp PSNR >50 dB
- ✅ Tests en BOSSBase passing

---

## 🎓 Presentaciones

### Revisión 1 Presentación
**Contenido:**
- ✅ Arquitectura implementada (45,998 params)
- ✅ Métricas alcanzadas (PSNR, SSIM)
- ✅ Training exitoso (loss curves)
- ✅ Comparación con paper
- ⏳ Limitaciones conocidas (SRM filter)

**Diapositivas clave:**
- Slide 14: Progreso actualizado a 100%
- Slide 15: Métricas REALES (no "Pendiente")
- Slide 20: Logros principales (con evidencia)

### Revisión 2 Presentación
**Contenido nuevo:**
- Validación robusta (BOSSBase)
- Tests de robustez (JPEG, steganalysis)
- Comparación con SOTA
- Limitaciones y trabajo futuro

---

## ⚠️ Riesgos y Mitigaciones

### Riesgo 1: No alcanzar PSNR 58 dB
**Probabilidad:** Media
**Mitigación:**
- Ajustar learning rate
- Más epochs (150-200)
- Ajustar loss weights (α, β, γ)

### Riesgo 2: Overfitting
**Probabilidad:** Baja
**Mitigación:**
- Early stopping (patience=20)
- Monitorear val loss
- Data augmentation

### Riesgo 3: Training muy lento en CPU
**Probabilidad:** Alta
**Mitigación:**
- Usar GPU (Colab, Kaggle, cloud)
- Reducir batch size para caber en memoria
- Subset de data para testing rápido

---

## 📝 Checklist de Entrega

### Revisión 1
- [ ] Dataset ImageNet descargado y preparado
- [ ] Modelo entrenado 100 epochs
- [ ] PSNR ≥ 58 dB alcanzado
- [ ] SSIM ≥ 0.94 alcanzado
- [ ] Checkpoints guardados
- [ ] Logs de entrenamiento
- [ ] Gráficas generadas
- [ ] Presentación actualizada
- [ ] PROGRESS_LOG.md actualizado
- [ ] README con instrucciones

### Revisión 2
- [ ] BOSSBase descargado
- [ ] Tests de robustez ejecutados
- [ ] Steganalysis evaluado
- [ ] Documento de comparación SOTA
- [ ] Limitaciones documentadas
- [ ] Trabajo futuro planificado

---

## 🔄 Flexibilidad

**Si no se logra PSNR 58 dB en Revisión 1:**
- Documentar PSNR alcanzado (ej: 55 dB)
- Analizar diferencias con paper
- Proponer mejoras para Revisión 2
- Mostrar progreso (learning curves)

**Si ImageNet no está disponible:**
- Usar subset más pequeño (10K imágenes)
- Documentar como limitación
- Estimar métricas con subset
- Plan para dataset completo después

---

## 📞 Soporte

**Dudas técnicas:**
- Ver DATASET_GUIDE.md para datasets
- Ver README.md para instrucciones generales
- Ver PROGRESS_LOG.md para historial

**Problemas comunes:**
- Error de memoria: reducir batch_size
- Training lento: usar GPU o subset
- Dataset no found: verificar paths

---

**Última actualización:** Marzo 18, 2026
**Estado actual:** Revisión 1 - 80% completado
**Próximo milestone:** Descargar ImageNet y entrenar modelo
