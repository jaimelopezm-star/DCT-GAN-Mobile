# DCT-GAN Mobile Steganography

## 📋 Descripción

Implementación del framework híbrido DCT-GAN para esteganografía de imágenes con optimización para dispositivos móviles. Basado en el paper de Malik et al. (2025) con mejoras propuestas para arquitecturas ligeras.

## 🎯 Objetivos del Proyecto

### ✅ Fase 1: Baseline de Alta Calidad (Validada)
- ✅ Implementación de línea densa (Dense Encoder + Dense Decoder)
- ✅ Evaluación en DIV2K real (no sintético)
- ✅ Resultado validado: **PSNR visual 48.81 dB**
- ✅ Resultado validado: **Recovery PSNR 17.43 dB**

**Estado:** Baseline fuerte y reproducible con checkpoint `exp19_quality_base`.

### ✅ Fase 2: Robustez Geométrica (Validación Inicial Completada)
- ✅ Evaluación de robustez JPEG y ataques geométricos (rotación/traslación/escala)
- ✅ Compensación automática para **rotación pequeña y traslación pequeña**
- ✅ Ganancia significativa en recuperación frente a ataques geométricos

**Contribución actual:** sistema de esteganografía con baseline fuerte y extensión de robustez geométrica automática.

### 🚀 Fase 3: Mobile-StegoNet (Propuesta futura)
- Reemplazar ResNet por MobileNetV3-Small (reducción 60% parámetros)
- Implementar cuantización INT8 y pruning de canales
- Optimizar para TensorFlow Lite (objetivo: <500ms CPU, <50MB memoria)

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────┐
│                  Cover Image (256×256)              │
│                  Secret Image (256×256)             │
└────────────────────┬────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   DCT Transform     │
          │   (Stage 1)         │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   Encoder (ResNet)  │
          │   9 Residual Blocks │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   Stego Image       │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   Decoder (CNN)     │
          │   6 Conv Layers     │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   Recovered Secret  │
          └─────────────────────┘
          
          ┌──────────────────────┐
          │  Discriminator       │
          │  (XuNet Modified)    │
          └──────────────────────┘
```

## 📁 Estructura del Proyecto

```
DCT-GAN-Mobile/
├── src/
│   ├── models/
│   │   ├── encoder.py          # ResNet/MobileNet encoder
│   │   ├── decoder.py          # CNN decoder
│   │   ├── discriminator.py    # XuNet discriminator
│   │   └── gan.py             # GAN completo
│   ├── dct/
│   │   ├── transform.py        # Transformada DCT
│   │   ├── coefficients.py     # Selección de coeficientes
│   │   └── embedding.py        # Lógica de inserción
│   ├── training/
│   │   ├── trainer.py          # Loop de entrenamiento
│   │   ├── losses.py           # Funciones de pérdida
│   │   └── metrics.py          # PSNR, SSIM, BER, CC
│   ├── evaluation/
│   │   ├── evaluate.py         # Scripts de evaluación
│   │   └── steganalysis.py     # Pruebas de detección
│   └── utils/
│       ├── dataset.py          # DataLoaders
│       ├── preprocessing.py    # Normalización, resize
│       └── visualization.py    # Plots y comparaciones
├── data/
│   ├── scripts/
│   │   ├── download_bossbase.py
│   │   ├── download_imagenet.py
│   │   └── download_uscsipi.py
│   └── README.md
├── configs/
│   ├── base_config.yaml        # Configuración paper original
│   └── mobile_config.yaml      # Configuración Mobile-StegoNet
├── notebooks/
│   ├── 01_exploracion_datos.ipynb
│   ├── 02_prueba_dct.ipynb
│   └── 03_visualizacion_resultados.ipynb
├── tests/
│   ├── test_dct.py
│   ├── test_models.py
│   └── test_training.py
├── experiments/
│   └── logs/                   # TensorBoard logs
├── checkpoints/                # Modelos guardados
├── requirements.txt
├── setup.py
├── Dockerfile
├── .gitignore
└── README.md
```

## 🔧 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/DCT-GAN-Mobile.git
cd DCT-GAN-Mobile
```

### 2. Crear entorno virtual
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Descargar datasets
```bash
python data/scripts/download_bossbase.py
python data/scripts/download_uscsipi.py
```

## 🚀 Quick Start

### ⚡ Setup Automático (MÁS RÁPIDO)

```powershell
# Una línea: Testing + Descarga en paralelo
.\quick_setup.ps1
```

**Hace automáticamente:**
- ✅ Testing rápido (10 min) → Valida que código funciona
- 🔽 Descarga ImageNet (1-2 horas) → En nueva ventana, background
- 📦 Prepara splits (40K/5K/5K) → Automático
- ✅ Todo listo para entrenar!

**Ver detalles:** [docs/QUICK_SETUP_GUIDE.md](docs/QUICK_SETUP_GUIDE.md)

---

### 📖 Setup Manual (Paso a Paso)

Para replicar el paper (Revisión 1):

```powershell
# 1. Descargar ImageNet 2012 (~6.3 GB)
python scripts/download_imagenet.py --method kaggle

# 2. Preparar splits (40K/5K/5K)
python scripts/prepare_dataset.py

# 3. Entrenar modelo
python train.py --config configs/base_config.yaml --dataset imagenet

# 4. Esperar 2-7 días (depende de GPU/CPU)
# Checkpoints en: checkpoints/
# Logs en: logs/
```

**📖 Guía completa:** Ver [QUICKSTART.md](QUICKSTART.md) con detalles paso a paso

### 🧪 Para testing rápido (sin descargar dataset):

```powershell
# Validar que el código funciona (datos sintéticos)
python train.py --config configs/base_config.yaml --dataset synthetic

# Nota: Esto NO replicará métricas del paper, solo valida infraestructura
```

### 📊 Para ver progreso del entrenamiento:

```powershell
# Monitorear logs en tiempo real
Get-Content logs\training_*.log -Tail 50 -Wait

# Ver checkpoints guardados
ls checkpoints\
```

### 📈 Archivos de Documentación:

- **[QUICKSTART.md](QUICKSTART.md)** - Guía paso a paso para comenzar
- **[ROADMAP.md](ROADMAP.md)** - Plan de revisiones y milestones
- **[PROGRESS_LOG.md](PROGRESS_LOG.md)** - Historial detallado de desarrollo
- **[DATASET_GUIDE.md](DATASET_GUIDE.md)** - Información sobre datasets

## 📊 Métricas de Evaluación

| Métrica | Paper Original | Nuestra Implementación (Abril 2026) | Estado |
|---------|---------------|--------------------------------------|--------|
| **Calidad visual** |
| PSNR visual (cover vs stego) | 58.27 dB | **48.81 dB** | ✅ Validado |
| SSIM visual | 0.942 | Pendiente (línea Dense) | ⏳ |
| **Recuperación** |
| Recovery PSNR (baseline) | - | **17.43 dB** | ✅ Validado |
| **Robustez JPEG** |
| JPEG Q=95 (recovery) | - | 12.47 dB | ⚠️ |
| JPEG Q=75 (recovery) | - | 12.20 dB | ⚠️ |
| JPEG Q=50 (recovery) | 95%* | 12.00 dB | ⚠️ En mejora |
| **Robustez geométrica** |
| Rotate +5° (sin compensación) | - | 13.59 dB | ✅ |
| Rotate +5° (auto compensación) | - | **15.10 dB** | ✅ Mejora |
| Translate 4:0 (sin compensación) | - | 15.81 dB | ✅ |
| Translate 4:0 (auto compensación) | - | **17.11 dB** | ✅ Mejora |

\* El paper reporta robustez con criterio distinto; aquí se reporta Recovery PSNR bajo ataque.

## 🧪 Resultados de Robustez (Abril 2026)

El script `evaluate_dense_robustness.py` valida tres escenarios para ataques geométricos:
- Sin compensación
- Compensación automática (estimada por registro)
- Compensación oracle (techo superior)

Resultado clave:
- En rotación y traslación pequeñas, la compensación automática recupera gran parte de la caída.
- En varios casos, el resultado `auto` queda muy cercano al `oracle`.
- JPEG sigue siendo el frente principal de mejora futura.

Comando de evaluación usado:

```bash
python evaluate_dense_robustness.py \
    --checkpoint checkpoints/exp19_quality_base/best_model.pth \
    --val_dir /workspace/DIV2K_prepared/val/images \
    --samples 120 \
    --auto_compensation \
    --oracle_compensation
```

## 📚 Referencias

1. **Malik et al. (2025)** - "Hybrid Steganography using the Discrete Cosine Transform and Generative Adversarial Networks for Secure Big Data Communication", *Scientific Reports*, 15:19630.

2. **JPEG-Resistant DCT Steganography** - Adaptive texture-based mid-frequency coefficient embedding, PSNR: 38-41 dB, 95% JPEG robustness.

3. **DCT Domain Digital Image Steganography** - Entropy Thresholding (ET) and Selective Embedding in Coefficients (SEC) schemes.

## 🛠️ Tecnologías

- **Framework**: PyTorch 2.0+
- **Optimización Móvil**: TensorFlow Lite, ONNX Runtime
- **Procesamiento**: OpenCV, Pillow, NumPy
- **Visualización**: Matplotlib, TensorBoard
- **Testing**: pytest, unittest

## 📈 Roadmap

- [x] Estructura del proyecto
- [x] Baseline Dense entrenado y evaluado en DIV2K
- [x] Evaluación de robustez JPEG/geométrica
- [x] Compensación automática en rotación/traslación pequeñas
- [ ] Mejora de robustez JPEG
- [ ] Métricas adicionales (SSIM, BER robusto) en línea Dense
- [ ] Validación extendida en transformaciones compuestas
- [ ] Implementación MobileNetV3 encoder
- [ ] Cuantización INT8
- [ ] Optimización TensorFlow Lite
- [ ] Benchmarking en Android

## 🤝 Contribuciones

Este proyecto es parte de una tesis de maestría en Ciberseguridad e Infraestructura Web. Sugerencias y mejoras son bienvenidas.

## 📄 Licencia

MIT License - Ver archivo LICENSE para detalles.

## 👤 Autor

**Estudiante de Maestría en Ciberseguridad**  
*Universidad [Tu Universidad]*  
Línea de investigación: Esteganografía, Trust Boundaries, Seguridad en Dispositivos Móviles

---

**Última actualización**: Abril 2026
