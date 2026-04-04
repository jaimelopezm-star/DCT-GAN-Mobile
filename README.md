# DCT-GAN Mobile Steganography

## 📋 Descripción

Implementación del framework híbrido DCT-GAN para esteganografía de imágenes con optimización para dispositivos móviles. Basado en el paper de Malik et al. (2025) con mejoras propuestas para arquitecturas ligeras.

## 🎯 Objetivos del Proyecto

### ✅ Fase 1: Replicación del Paper Base (ACTUAL - 80% completado)
- ✅ Implementar arquitectura DCT-GAN original (ResNet encoder + CNN decoder + XuNet discriminator)
- ✅ Optimizar parámetros: 45,998 params (-7.9% vs paper ✓)
- ⏳ Entrenar en ImageNet 2012 (50K imágenes)
- ⏳ Validar métricas: PSNR ~58 dB, SSIM ~0.94

**Estado:** Código completo, pendiente entrenamiento con dataset real  
**Ver:** [QUICKSTART.md](QUICKSTART.md) para instrucciones de uso

### 📋 Fase 2: Validación Robusta (Siguiente revisión)
- Agregar BOSSBase 1.01 para steganalysis testing
- Validar robustez JPEG (95% accuracy @ Q=50)
- Comparar con métodos SOTA

**Ver:** [ROADMAP.md](ROADMAP.md) para plan completo

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

| Métrica | Paper Original | Nuestra Implementación | Estado |
|---------|---------------|----------------------|--------|
| **Arquitectura** |
| Parámetros | 49.95K | 45.998K | ✅ -7.9% |
| Encoder params | ~25K | 17.010K | ✅ |
| Decoder params | ~15K | 4.143K | ✅ |
| Discriminator params | ~10K | 24.845K | ⚠️ |
| **Calidad (Imagen)** |
| PSNR (dB) | 58.27 | Pendiente* | ⏳ |
| SSIM | 0.942 | Pendiente* | ⏳ |
| **Recuperación (Secret)** |
| Accuracy | 96.10% | 96.03%** | ✅ |
| BER | <0.05 | 0.0397** | ✅ |
| **Robustez** |
| JPEG (Q=50) | 95% | Pendiente* | ⏳ |
| Inferencia (ms) | 17-18 | Pendiente* | ⏳ |

\* Requiere entrenamiento completo con ImageNet  
\*\* Medido en datos sintéticos (testing)

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
- [ ] Implementación DCT transform
- [ ] Implementación encoder ResNet
- [ ] Implementación decoder CNN
- [ ] Implementación discriminador XuNet
- [ ] Loop de entrenamiento GAN
- [ ] Validación métricas paper original
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

**Última actualización**: Marzo 2026
