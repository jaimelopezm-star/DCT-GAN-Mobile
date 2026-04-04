# Guía de Datasets - DCT-GAN Steganography

## 📊 Datasets Requeridos

### 1. ImageNet 2012 (Principal - Paper Original)

**Descripción:**
- Dataset usado en el paper Malik et al. (2025)
- 50,000 imágenes RGB del validation set
- 1,000 categorías (animales, objetos, escenas)
- Tamaño: 256×256 (resize necesario)

**Descarga:**
```bash
# Opción 1: Kaggle (requiere cuenta gratuita)
# https://www.kaggle.com/c/imagenet-object-localization-challenge/data

# Opción 2: Official (requiere registro académico)
# https://image-net.org/download

# Opción 3: Torrent (más rápido)
# Validation set: ~6.3 GB
```

**Split del Paper:**
```
Total: 50,000 imágenes
├── Train: 40,000 (80%)
├── Validation: 5,000 (10%)
└── Test: 5,000 (10%)
```

**Uso en el Código:**
```python
from torchvision import datasets, transforms

# Transformaciones
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Cargar dataset
imagenet_val = datasets.ImageFolder(
    root='data/imagenet2012/val',
    transform=transform
)
```

---

### 2. BOSSBase 1.01 (Para Steganalysis)

**Descripción:**
- Dataset estándar en esteganografía
- 10,000 imágenes grayscale 512×512
- Sin compresión, alta calidad
- Usado para evaluar resistencia a steganalysis

**Descarga:**
```bash
# URL oficial (Universidad de Binghamton)
wget http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip

# Extraer
unzip BOSSbase_1.01.zip -d data/bossbase/
```

**Conversión a RGB:**
```python
from PIL import Image
import os

def convert_grayscale_to_rgb(input_dir, output_dir):
    """Convierte imágenes grayscale a RGB"""
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = Image.open(img_path).convert('L')
        
        # Convertir a RGB (replicar canal)
        img_rgb = Image.merge('RGB', (img, img, img))
        
        # Resize a 256×256
        img_rgb = img_rgb.resize((256, 256), Image.LANCZOS)
        
        # Guardar
        output_path = os.path.join(output_dir, img_name)
        img_rgb.save(output_path)

convert_grayscale_to_rgb('data/bossbase/raw', 'data/bossbase/rgb_256')
```

**Split Recomendado:**
```
Total: 10,000 imágenes
├── Train: 8,000 (80%)
├── Validation: 1,000 (10%)
└── Test: 1,000 (10%)
```

---

### 3. USC-SIPI (Opcional - Texturas)

**Descripción:**
- 512 imágenes de texturas y objetos
- Categorías: texturas, rostros, misc
- Alta variabilidad visual
- Útil para testing adicional

**Descarga:**
```bash
# Manual desde website
# https://sipi.usc.edu/database/

# Categorías principales:
# - Textures: http://sipi.usc.edu/database/database.php?volume=textures
# - Misc: http://sipi.usc.edu/database/database.php?volume=misc
# - Aerials: http://sipi.usc.edu/database/database.php?volume=aerials
```

**Uso:**
```python
# Solo para testing/validation adicional
# No necesario para replicar el paper
```

---

### 4. WhatsApp-Compressed (Custom)

**Descripción:**
- Dataset personalizado para evaluar robustez
- Simula compresión de WhatsApp/redes sociales
- Se crea comprimiendo ImageNet con JPEG variable

**Creación:**
```python
from PIL import Image
import random
import os

def create_whatsapp_compressed(input_dir, output_dir):
    """
    Simula compresión de WhatsApp aplicando JPEG con
    calidad variable (50-95)
    """
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = Image.open(img_path)
        
        # Comprimir con calidad aleatoria (WhatsApp típico: 60-85)
        quality = random.randint(60, 85)
        
        output_path = os.path.join(output_dir, img_name)
        img.save(output_path, 'JPEG', quality=quality)

# Uso
create_whatsapp_compressed(
    'data/imagenet2012/val_subset',
    'data/whatsapp_compressed/'
)
```

---

## 🎯 Estrategia Recomendada

### Para Replicar el Paper (Mínimo):
```
✅ REQUERIDO: ImageNet 2012 validation (50K imágenes)
   - Split: 40K train / 5K val / 5K test
   - Formato: RGB 256×256
   - Normalización: ImageNet stats
```

### Para Validación Completa (Recomendado):
```
✅ ImageNet 2012 (principal)
✅ BOSSBase 1.01 (steganalysis robustness)
⚙️ WhatsApp-Compressed (opcional, robustez JPEG)
⚙️ USC-SIPI (opcional, diversidad adicional)
```

---

## 📁 Estructura de Directorios

```
DCT-GAN-Mobile/
├── data/
│   ├── imagenet2012/
│   │   ├── train/          # 40,000 imágenes
│   │   ├── val/            # 5,000 imágenes
│   │   └── test/           # 5,000 imágenes
│   │
│   ├── bossbase/
│   │   ├── raw/            # Originales grayscale 512×512
│   │   └── rgb_256/        # Convertidas RGB 256×256
│   │       ├── train/      # 8,000 imágenes
│   │       ├── val/        # 1,000 imágenes
│   │       └── test/       # 1,000 imágenes
│   │
│   ├── usc_sipi/           # Opcional
│   │   └── textures/
│   │
│   └── whatsapp_compressed/  # Custom
│       └── test/
│
└── scripts/
    ├── download_imagenet.py
    ├── download_bossbase.py
    └── create_whatsapp_dataset.py
```

---

## 🚀 Scripts de Descarga Automática

### Script 1: Descargar ImageNet

```python
# scripts/download_imagenet.py
import os
import requests
from tqdm import tqdm

def download_imagenet_kaggle():
    """
    Descarga ImageNet desde Kaggle
    Requiere: kaggle API token (~/.kaggle/kaggle.json)
    """
    print("Descargando ImageNet 2012 validation set...")
    
    # Instalar Kaggle CLI
    os.system("pip install kaggle")
    
    # Descargar
    os.system("kaggle competitions download -c imagenet-object-localization-challenge")
    
    # Extraer
    print("Extrayendo archivos...")
    os.system("unzip imagenet-object-localization-challenge.zip -d data/imagenet2012/")
    
    print("✅ ImageNet descargado en: data/imagenet2012/")

if __name__ == "__main__":
    download_imagenet_kaggle()
```

### Script 2: Descargar BOSSBase

```python
# scripts/download_bossbase.py
import urllib.request
import zipfile
import os
from tqdm import tqdm

def download_bossbase():
    """Descarga y extrae BOSSBase 1.01"""
    
    url = "http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip"
    output_dir = "data/bossbase/"
    zip_path = os.path.join(output_dir, "BOSSbase_1.01.zip")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Descargando BOSSBase 1.01 (~1.5 GB)...")
    
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded / total_size * 100, 100)
        print(f"\rProgreso: {percent:.1f}%", end='')
    
    urllib.request.urlretrieve(url, zip_path, download_progress)
    
    print("\n\nExtrayendo archivos...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(output_dir, 'raw'))
    
    print(f"✅ BOSSBase descargado en: {output_dir}")
    
    # Convertir a RGB
    print("\nConvirtiendo a RGB 256×256...")
    convert_grayscale_to_rgb(
        os.path.join(output_dir, 'raw'),
        os.path.join(output_dir, 'rgb_256')
    )
    
    print("✅ Conversión completada")

def convert_grayscale_to_rgb(input_dir, output_dir):
    """Convierte grayscale a RGB y resize a 256×256"""
    from PIL import Image
    
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.endswith(('.pgm', '.jpg', '.png'))]
    
    for img_name in tqdm(files, desc="Convirtiendo"):
        img_path = os.path.join(input_dir, img_name)
        img = Image.open(img_path).convert('L')
        
        # Convertir a RGB
        img_rgb = Image.merge('RGB', (img, img, img))
        
        # Resize
        img_rgb = img_rgb.resize((256, 256), Image.LANCZOS)
        
        # Guardar como PNG
        output_name = os.path.splitext(img_name)[0] + '.png'
        output_path = os.path.join(output_dir, output_name)
        img_rgb.save(output_path)

if __name__ == "__main__":
    download_bossbase()
```

---

## 🔧 Uso en train.py

### Dataset Loader Actualizado

```python
# train.py - Reemplazar SteganoDataset

from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split
import random

class SteganographyPairDataset(Dataset):
    """
    Dataset de pares (cover, secret) para esteganografía
    
    Args:
        image_dataset: Dataset de imágenes (ImageNet, BOSSBase, etc.)
        transform: Transformaciones a aplicar
    """
    
    def __init__(self, image_dataset, transform=None):
        self.images = image_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Cover image
        cover, _ = self.images[idx]
        
        # Secret image (aleatorio)
        secret_idx = random.randint(0, len(self.images) - 1)
        secret, _ = self.images[secret_idx]
        
        return {
            'cover': cover,
            'secret': secret
        }

# Crear datasets
def create_imagenet_dataloaders(config):
    """Crea DataLoaders desde ImageNet"""
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    
    # Cargar ImageNet
    full_dataset = datasets.ImageFolder(
        root='data/imagenet2012/val',
        transform=transform
    )
    
    # Split 40K/5K/5K
    train_size = 40000
    val_size = 5000
    test_size = len(full_dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        full_dataset, 
        [train_size, val_size, test_size]
    )
    
    # Crear pares
    train_pairs = SteganographyPairDataset(train_ds)
    val_pairs = SteganographyPairDataset(val_ds)
    
    # DataLoaders
    train_loader = DataLoader(train_pairs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_pairs, batch_size=32, shuffle=False)
    
    return train_loader, val_loader
```

---

## ⚡ Quick Start

### Opción 1: Datos Sintéticos (Testing Rápido)
```bash
# Ya implementado en train.py actual
python train.py --config configs/base_config.yaml

⚠️ Usa datos sintéticos (solo para testing rápido)
⚠️ PSNR no será realista
```

### Opción 2: ImageNet (Replicación Paper)
```bash
# 1. Descargar ImageNet
python scripts/download_imagenet.py

# 2. Actualizar train.py para usar ImageNet real
# 3. Entrenar
python train.py --config configs/base_config.yaml
```

### Opción 3: Validación Completa
```bash
# 1. Descargar ambos datasets
python scripts/download_imagenet.py
python scripts/download_bossbase.py

# 2. Entrenar con ImageNet
python train.py --config configs/base_config.yaml --dataset imagenet

# 3. Validar robustez con BOSSBase
python test.py --checkpoint best_model.pth --dataset bossbase
```

---

## ❓ Preguntas Frecuentes

**Q: ¿Puedo usar solo BOSSBase en lugar de ImageNet?**
A: Sí, pero no sería replicación exacta del paper. BOSSBase tiene menos variabilidad.

**Q: ¿Cuánto espacio en disco necesito?**
A: 
- ImageNet val: ~6.3 GB
- BOSSBase: ~1.5 GB 
- Total mínimo: ~8 GB

**Q: ¿Cuánto tarda la descarga?**
A: 
- ImageNet: 30-60 min (depende de conexión)
- BOSSBase: 10-20 min

**Q: ¿Puedo usar un subset de ImageNet?**
A: Sí, pero para replicar exactamente el paper necesitas las 50K imágenes del validation set.

---

## 📌 Siguientes Pasos

1. ✅ **Decidir estrategia de datos:**
   - [ ] Solo ImageNet (replicación exacta)
   - [ ] ImageNet + BOSSBase (validación robusta)

2. ✅ **Descargar datasets:**
   - [ ] Ejecutar `python scripts/download_imagenet.py`
   - [ ] Ejecutar `python scripts/download_bossbase.py`

3. ✅ **Actualizar train.py:**
   - [ ] Reemplazar `SteganoDataset` con `SteganographyPairDataset`
   - [ ] Usar datasets reales en lugar de sintéticos

4. ✅ **Entrenar modelo:**
   - [ ] `python train.py --config configs/base_config.yaml`
   - [ ] Esperar 100 epochs (~varias horas/días)

5. ✅ **Validar métricas:**
   - [ ] PSNR ≥ 58 dB
   - [ ] SSIM ≥ 0.942
