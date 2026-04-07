# Datasets para Esteganografía: BOSSBase y Alternativas

## 📦 Datasets Recomendados

### 1. BOSSBase 1.01 (PREFERIDO - Mencionado en el AI search)

**Descripción**: 
- 10,000 imágenes naturales sin comprimir
- 512×512 píxeles, escala de grises
- Usado como estándar en competencias de esteganografía
- Imágenes "limpias" → Más fácil alcanzar PSNR alto

**Descarga**:
- Sitio oficial: http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip
- Alternativa: https://www.kaggle.com/datasets (buscar "bossbase steganography")
- Tamaño: ~700 MB

**Uso en código**:
```python
# configs/paper_exact_config.yaml
data:
  dataset_name: "BOSSBase"
  base_path: "/path/to/BOSSbase_1.01/"
  image_size: 512  # Nativo
  train_size: 8000
  val_size: 1000
  test_size: 1000
```

---

### 2. BOWS2 (Break Our Watermarking System)

**Descripción**:
- 10,000 imágenes naturales
- Usado en competencia BOSS
- Similar a BOSSBase pero con variaciones

**Descarga**:
- http://bows2.ec-lille.fr/

---

### 3. DIV2K (Alternativa moderna)

**Descripción**:
- 1,000 imágenes de alta resolución (2K)
- Usado en RISRANet (paper citado como alternativa)
- Reporta PSNR > 50 dB

**Descarga**:
- https://data.vision.ee.ethz.ch/cvl/DIV2K/

**Ventajas**:
- Imágenes de muy alta calidad
- Usado en papers recientes con buenos resultados
- Dataset pequeño (fácil de trabajar)

---

### 4. COCO (Microsoft Common Objects)

**Descripción**:
- 330K imágenes variadas
- Usado por StegaVision (citado en search)
- Más complejo pero muy versátil

**Descarga**:
- https://cocodataset.org/

---

## 🎯 Recomendación Prioritaria

Según el **AI search externo** y el **paper**:

1. **PRIMERA OPCIÓN**: BOSSBase 1.01
   - Razón: Paper menciona imágenes "limpias"
   - PSNR 58 dB es más alcanzable con BOSSBase
   - Dataset estándar en steganography

2. **SEGUNDA OPCIÓN**: DIV2K
   - Razón: RISRANet logra PSNR > 50 dB con este dataset
   - Más moderno
   - Código de referencia disponible

3. **TERCERA OPCIÓN**: Continuar con ImageNet2012 sintético
   - Razón: Ya está funcionando
   - Si con fix 5D:1G mejora significativamente, no cambiar dataset

---

## 📥 Cómo Descargar BOSSBase

### Opción 1: Directo desde Binghamton
```bash
# Método 1: wget
wget http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip
unzip BOSSbase_1.01.zip

# Método 2: curl
curl -O http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip
unzip BOSSbase_1.01.zip
```

### Opción 2: Kaggle
```bash
# Instalar Kaggle CLI
pip install kaggle

# Buscar dataset
kaggle datasets list -s "bossbase"

# Descargar (si existe)
kaggle datasets download -d <dataset-name>
```

### Opción 3: Google Drive / Papers with Code
- Buscar en: https://paperswithcode.com/dataset/boss
- Muchas veces hay mirrors en Google Drive

---

## 🔄 Adaptar Código para BOSSBase

### 1. Crear Data Loader Específico

Archivo: `src/data/bossbase_dataset.py`

```python
import os
from PIL import Image
from torch.utils.data import Dataset

class BOSSBaseDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # BOSSBase: 10K images (512x512 grayscale)
        all_images = sorted([f for f in os.listdir(root_dir) if f.endswith('.pgm')])
        
        # Split: 80% train, 10% val, 10% test
        n = len(all_images)
        if split == 'train':
            self.images = all_images[:int(0.8*n)]
        elif split == 'val':
            self.images = all_images[int(0.8*n):int(0.9*n)]
        else:  # test
            self.images = all_images[int(0.9*n):]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        
        # Load grayscale image
        image = Image.open(img_path).convert('RGB')  # Convert to RGB
        
        if self.transform:
            image = self.transform(image)
        
        # Return image as both cover and secret (standard in steganography)
        return {
            'cover': image,
            'secret': image  # Or load different image
        }
```

### 2. Actualizar Config

```yaml
data:
  dataset_name: "BOSSBase"
  base_path: "/path/to/BOSSbase_1.01/"
  image_size: 512  # BOSSBase native resolution
  num_workers: 4
  pin_memory: true
  batch_size: 16  # Reduce si 512x512 es muy pesado
  
  # Data augmentation para BOSSBase
  transforms:
    - RandomHorizontalFlip
    - RandomRotation: 15
    - ColorJitter:
        brightness: 0.1
        contrast: 0.1
```

---

## ✅ Plan de Acción

### Fase 1: Entrenar con Fix Actual (ImageNet sintético)
```bash
# Usar fix 5D:1G con dataset actual
python train.py --config configs/paper_exact_config.yaml
```

**Si mejora significativamente** (PSNR > 30 dB):
→ Continuar con ImageNet sintético

**Si mejora poco** (PSNR < 25 dB):
→ Pasar a Fase 2

### Fase 2: Probar con BOSSBase

1. Descargar BOSSBase:
```bash
wget http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip
unzip BOSSbase_1.01.zip -d data/bossbase/
```

2. Crear data loader (código arriba)

3. Entrenar:
```bash
python train.py --config configs/bossbase_config.yaml
```

### Fase 3: Si TODOavía falla
→ Contactar autores (email template ya preparado)
→ O pivotar a alternativa (SteganoGAN, RISRANet)

---

## 📊 Comparación de Datasets

| Dataset | Tamaño | Resolución | Calidad | PSNR Típico | Disponibilidad |
|---------|--------|------------|---------|-------------|----------------|
| **BOSSBase** | 10K | 512×512 | Muy alta | 50-60 dB | ⚠️ Medio (link directo) |
| **DIV2K** | 1K | 2K | Excelente | 50-55 dB | ✅ Fácil |
| **COCO** | 330K | Variable | Buena | 40-50 dB | ✅ Fácil |
| **ImageNet** | 1.4M | 256×256 | Variable | 35-45 dB | ✅ Fácil |
| **Sintético** | Infinito | Variable | Media | 25-40 dB | ✅ Muy fácil |

---

## 🚨 NOTA IMPORTANTE

Según el AI search, el paper probablemente usó **BOSSBase** o **BOWS2** porque:

1. PSNR 58.27 dB es **extremadamente alto**
2. Solo alcanzable con imágenes muy "limpias" y baja complejidad
3. ImageNet2012 es muy diverso → Más difícil alcanzar 58 dB
4. BOSSBase es el estándar en papers de esteganografía

**Conclusión**: Si después del fix 5D:1G no alcanzas >40 dB con ImageNet sintético, **definitivamente** prueba con BOSSBase.

---

## 📞 Contactos para Datasets

Si no encuentras BOSSBase:

1. **Email a autores del paper**:
   - ateeq@ksu.edu.sa
   - Preguntar: "Could you confirm the exact dataset used?"

2. **Buscar en**:
   - Papers with Code: https://paperswithcode.com
   - Kaggle Datasets: https://www.kaggle.com/datasets
   - Academic torrents: http://academictorrents.com

3. **Último recurso**:
   - Pedir en Reddit r/MachineLearning
   - Preguntar en GitHub Issues de repos relacionados

---

**Fecha**: 2026-04-07  
**Creado por**: Análisis de recomendaciones del AI search externo
