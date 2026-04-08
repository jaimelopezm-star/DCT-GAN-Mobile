# Guía de Entrenamiento con BOSSBase Dataset

**Objetivo**: Entrenar DCT-GAN con el dataset BOSSBase para mejorar PSNR de 17.63 dB (sintético) a objetivo de 58.27 dB

**Dataset**: BOSSBase v1.01 - 10,000 imágenes grayscale 512×512 px
**Paper de referencia**: Usado en trabajos de esteganografía desde 2010

---

## 1. Descargar BOSSBase

### Opción A: Descarga directa (Recomendado)
```bash
# En RunPod o servidor con GPU
cd /workspace
wget http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip

# Descomprimir
unzip BOSSbase_1.01.zip -d BOSSbase
```

### Opción B: Descarga manual
- URL: http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip
- Tamaño: ~1.8 GB
- Subir a RunPod vía web UI o SCP

---

## 2. Preparar Dataset

### Estructura esperada:
```
/workspace/BOSSbase/
├── 1.pgm
├── 2.pgm
├── 3.pgm
...
└── 10000.pgm
```

### Script de preparación

**Nota**: El script `prepare_bossbase.py` ya está incluido en el repositorio DCT-GAN-Mobile.

Contenido del script:
```python
import os
from pathlib import Path
from PIL import Image
import numpy as np

def prepare_bossbase(source_dir, output_dir, split_ratios=(0.8, 0.1, 0.1)):
    """
    Prepara BOSSBase para entrenamiento
    
    Args:
        source_dir: Directorio con imágenes PGM
        output_dir: Directorio de salida
        split_ratios: (train, val, test)
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Crear subdirectorios
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Listar todas las imágenes
    images = sorted(list(source_path.glob('*.pgm')))
    total = len(images)
    
    print(f"Total imágenes encontradas: {total}")
    
    # Calcular splits
    train_end = int(total * split_ratios[0])
    val_end = train_end + int(total * split_ratios[1])
    
    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }
    
    # Convertir y guardar
    for split_name, split_images in splits.items():
        print(f"\nProcesando {split_name}: {len(split_images)} imágenes")
        
        for i, img_path in enumerate(split_images):
            if i % 100 == 0:
                print(f"  {split_name}: {i}/{len(split_images)}")
            
            # Leer imagen grayscale
            img = Image.open(img_path)
            
            # Convertir a RGB (replicar canal grayscale)
            img_rgb = img.convert('RGB')
            
            # Redimensionar a 256x256 (modelo espera 256x256)
            img_resized = img_rgb.resize((256, 256), Image.LANCZOS)
            
            # Guardar como PNG
            output_file = output_path / split_name / f"{img_path.stem}.png"
            img_resized.save(output_file, 'PNG')
    
    print(f"\n✅ Dataset preparado en: {output_path}")
    print(f"   - Train: {len(splits['train'])} imágenes")
    print(f"   - Val: {len(splits['val'])} imágenes")
    print(f"   - Test: {len(splits['test'])} imágenes")

if __name__ == "__main__":
    prepare_bossbase(
        source_dir="/workspace/BOSSbase",
        output_dir="/workspace/BOSSbase_prepared",
        split_ratios=(0.8, 0.1, 0.1)  # 8000 train, 1000 val, 1000 test
    )
```

### Ejecutar preparación:
```bash
# Descomprimir BOSSBase si aún no está descomprimido
cd /workspace
unzip BOSSbase_1.01.zip -d BOSSbase

# Ejecutar script de preparación
cd DCT-GAN-Mobile
python prepare_bossbase.py --source /workspace/BOSSbase --output /workspace/BOSSbase_prepared

# Verificar resultado
ls -la /workspace/BOSSbase_prepared/train/ | head -20
```

**Tiempo estimado**: 5-10 minutos

**Opciones del script**:
```bash
# Con parámetros personalizados
python prepare_bossbase.py \
  --source /ruta/a/imagenes/pgm \
  --output /ruta/salida \
  --split 0.8 0.1 0.1  # train/val/test ratios
```

---

## 3. Crear Config para BOSSBase

Crear `/workspace/DCT-GAN-Mobile/configs/bossbase_config.yaml`:

```yaml
# Config para entrenamiento con BOSSBase
project:
  name: "DCT-GAN-BOSSBase-Train"
  description: "Entrenamiento con BOSSBase dataset real"
  seed: 42

# Dataset Configuration - BOSSBASE
data:
  dataset_name: "BOSSBase"
  dataset_path: "/workspace/BOSSbase_prepared"  # Path al dataset preparado
  train_size: 8000
  val_size: 1000
  test_size: 1000
  image_size: 256
  num_workers: 8
  pin_memory: true
  batch_size: 128       # Optimizado para RTX 4090

# Model Architecture (mismo que antes)
model:
  type: "dct_gan"
  
  encoder:
    type: "resnet"
    num_residual_blocks: 9
    base_channels: 16
    input_channels: 6
  
  decoder:
    type: "cnn"
    num_layers: 6
    base_channels: 16
    output_channels: 3
    activation: "relu"
  
  discriminator:
    type: "xunet_modified"
    input_channels: 3
    base_channels: 8
    kernel_size: 5
    num_conv_layers: 5
    activation: "leaky_relu"
    leaky_slope: 0.2
    use_srm: false

# DCT Configuration
dct:
  block_size: 8
  use_zig_zag: true
  frequency_selection: "chaotic_map"
  embed_in_ac: true

# Loss Functions
loss:
  alpha: 0.3
  beta: 15.0
  gamma: 0.03
  
  mse_loss:
    type: "mse"
    reduction: "mean"
  
  crossentropy_loss:
    type: "crossentropy"
    reduction: "mean"
  
  adversarial_loss:
    type: "wgan"
    clip_value: 0.01

# Training Configuration - OPTIMIZADO
training:
  num_epochs: 100             # Full training (paper usa 100)
  
  update_strategy:
    discriminator_updates_per_batch: 5
    generator_updates_per_batch: 1
  
  gradient_clipping:
    enabled: true
    max_norm: 1.0
  
  optimizer:
    generator:
      type: "adam"
      lr: 1.0e-3
      betas: [0.9, 0.999]
      weight_decay: 0.005
    
    discriminator:
      type: "adam"
      lr: 5.0e-3
      betas: [0.5, 0.999]
      weight_decay: 0.005

# Learning Rate Scheduler
scheduler:
  type: "step"
  step_size: 30
  gamma: 0.1

# Hardware - OPTIMIZADO
hardware:
  device: "cuda"
  num_gpus: 1
  mixed_precision: true

# Logging & Checkpoints
logging:
  log_frequency: 10
  save_frequency: 10        # Guardar cada 10 epochs
  
  metrics:
    - psnr
    - ssim
    - mse
    - rmse

# Early Stopping
early_stopping:
  enabled: true
  patience: 20              # Parar si no mejora en 20 epochs
  metric: "psnr"
  mode: "max"
```

---

## 4. Modificar train.py para Soportar BOSSBase

Necesitamos agregar un dataset loader para BOSSBase. Crear `/workspace/DCT-GAN-Mobile/src/data/bossbase_dataset.py`:

```python
"""
BOSSBase Dataset Loader
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import random

class BOSSBaseDataset(Dataset):
    """
    Dataset loader para BOSSBase
    
    Carga imágenes desde estructura:
    dataset_path/
        train/
        val/
        test/
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Path al directorio preparado
            split: 'train', 'val', o 'test'
            transform: Transformaciones a aplicar
        """
        self.root_dir = Path(root_dir) / split
        self.transform = transform or self._default_transform()
        
        # Listar todas las imágenes
        self.images = sorted(list(self.root_dir.glob('*.png')))
        
        print(f"[{split.upper()}] BOSSBase: {len(self.images)} imágenes")
    
    def _default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                               std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Cargar cover image
        cover_path = self.images[idx]
        cover = Image.open(cover_path).convert('RGB')
        
        # Cargar secret image (aleatorio)
        secret_idx = random.randint(0, len(self.images) - 1)
        secret_path = self.images[secret_idx]
        secret = Image.open(secret_path).convert('RGB')
        
        # Aplicar transformaciones
        cover = self.transform(cover)
        secret = self.transform(secret)
        
        return {
            'cover': cover,
            'secret': secret
        }
```

Modificar `train.py` para usar BOSSBase cuando se especifique:

```python
# En train.py, línea ~140 (función create_dataloaders)

if args.dataset == 'bossbase':
    from src.data.bossbase_dataset import BOSSBaseDataset
    
    print(f"📊 Usando dataset BOSSBase")
    print(f"   Path: {args.dataset_path}")
    
    train_dataset = BOSSBaseDataset(
        root_dir=args.dataset_path,
        split='train'
    )
    val_dataset = BOSSBaseDataset(
        root_dir=args.dataset_path,
        split='val'
    )
```

---

## 5. Entrenar con BOSSBase

### Test rápido (30 epochs):
```bash
# IMPORTANTE: Ejecutar desde el directorio del proyecto
cd /workspace/DCT-GAN-Mobile

python train.py \
  --config configs/bossbase_config.yaml \
  --dataset bossbase \
  --dataset-path /workspace/BOSSbase_prepared \
  --device cuda
```

### Entrenamiento completo (100 epochs):
Cambiar en `bossbase_config.yaml`:
```yaml
training:
  num_epochs: 100
```

**Tiempo estimado**:
- 30 epochs: ~4 horas (~$2.00)
- 100 epochs: ~13 horas (~$6.50)

---

## 6. Comparar Resultados

### Métricas esperadas con BOSSBase:

| Epoch | PSNR (dB) | SSIM | Comentario |
|-------|-----------|------|------------|
| 10 | 22-25 | 0.85 | Mejora sobre sintético (17.63 dB) |
| 30 | 30-35 | 0.90 | Progreso significativo |
| 50 | 38-45 | 0.92 | Acercándose a objetivo |
| 100 | 45-55 | 0.94 | Objetivo realista con dataset real |

**Objetivo paper**: 58.27 dB (puede requerir >100 epochs o arquitectura más grande)

---

## 7. Monitoreo durante entrenamiento

```bash
# Ver logs en tiempo real
tail -f /workspace/DCT-GAN-Mobile/logs/*/training*.log

# Ver métricas específicas
grep "Epoch.*VAL" /workspace/DCT-GAN-Mobile/logs/*/training*.log | tail -20

# Ver PSNR y SSIM
grep "PSNR\|SSIM" /workspace/DCT-GAN-Mobile/logs/*/training*.log | tail -30
```

---

## 8. Troubleshooting

### Problema: Out of Memory
**Solución**: Reducir `batch_size` en config:
```yaml
data:
  batch_size: 64  # o 32
```

### Problema: Imágenes no se cargan
**Verificar**:
```bash
ls -la /workspace/BOSSbase_prepared/train/ | head -20
```
Debe mostrar archivos `.png`

### Problema: PSNR no mejora después de epoch 30
**Posibles causas**:
1. LR decay (scheduler) puede estar bajando LR demasiado
2. Aumentar `patience` en early stopping
3. Probar con `base_channels` más grande (16 → 32)

---

## 9. Guardar mejores modelos

Los checkpoints se guardan automáticamente en:
```
/workspace/DCT-GAN-Mobile/checkpoints/
├── best_model.pth          # Mejor PSNR
├── checkpoint_epoch_010.pth
├── checkpoint_epoch_020.pth
...
```

Para descargar el mejor modelo:
```bash
# Desde RunPod terminal
zip -r best_model.zip checkpoints/best_model.pth

# Descargar vía web UI
```

---

## 10. Resumir entrenamiento desde checkpoint

Si se interrumpe, reanudar con:
```bash
python train.py \
  --config configs/bossbase_config.yaml \
  --dataset bossbase \
  --dataset-path /workspace/BOSSbase_prepared \
  --resume checkpoints/checkpoint_epoch_030.pth \
  --device cuda
```

---

## Referencias

- BOSSBase: http://dde.binghamton.edu/download/
- Paper original: Bas et al., "Break Our Steganographic System" (BOSS competition)
- Usado en: Holub & Fridrich (2012), Qian et al. (2015), y muchos otros trabajos de esteganografía
