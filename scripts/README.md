# Scripts para Preparación de Datasets

Este directorio contiene scripts para descargar y preparar los datasets necesarios para entrenar DCT-GAN.

## Scripts Disponibles

### 1. download_imagenet.py
Descarga ImageNet 2012 validation set (50,000 imágenes, ~6.3 GB)

```powershell
# Descarga automática con Kaggle
python scripts/download_imagenet.py --method kaggle

# Instrucciones para descarga manual
python scripts/download_imagenet.py --method manual
```

### 2. prepare_dataset.py
Crea splits de train/val/test desde ImageNet descargado

```powershell
# Crear splits estándar del paper (40K/5K/5K)
python scripts/prepare_dataset.py

# Splits personalizados
python scripts/prepare_dataset.py --train-size 30000 --val-size 5000 --test-size 5000
```

## Estructura de Datos Resultante

```
data/imagenet2012/
├── organized/
│   └── all/                    # Todas las imágenes (50,000)
│       ├── ILSVRC2012_val_00000001.JPEG
│       ├── ILSVRC2012_val_00000002.JPEG
│       └── ...
│
└── splits/
    ├── train/all/              # Training set (40,000 imágenes)
    ├── val/all/                # Validation set (5,000 imágenes)
    └── test/all/               # Test set (5,000 imágenes)
```

## Uso en Entrenamiento

Una vez preparado el dataset:

```powershell
# Entrenar con ImageNet
python train.py --config configs/base_config.yaml --dataset imagenet

# Especificar path personalizado
python train.py --dataset imagenet --dataset-path data/imagenet2012/splits
```

## Requisitos

- **Espacio en disco:** ~8 GB libres
- **Memoria RAM:** 4 GB mínimo
- **Tiempo de descarga:** 1-2 horas (depende de tu conexión)

## Descarga de BOSSBase (Revisión 2)

Para la siguiente revisión, se agregará:

```powershell
# Descarga BOSSBase 1.01 (10,000 imágenes, ~1.5 GB)
python scripts/download_bossbase.py
```

**Nota:** BOSSBase se usará en Revisión 2 para validación robusta contra steganalysis.
