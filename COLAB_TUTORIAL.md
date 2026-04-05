# 🚀 Tutorial: Entrenar DCT-GAN en Google Colab

## 📋 Paso a Paso (5 minutos setup)

### 1️⃣ Abrir Google Colab

Ve a: **https://colab.research.google.com/**

### 2️⃣ Subir el Notebook

**Opción A - Desde GitHub (RECOMENDADO)**:
1. En Colab: **File** → **Open notebook**
2. Pestaña **GitHub**
3. Pega esta URL en el buscador:
   ```
   https://github.com/jaimelopezm-star/DCT-GAN-Mobile
   ```
4. Selecciona el notebook: `train_gpu_cloud.ipynb`
5. Click **"Open in Colab"**

**Opción B - Subir archivo local**:
1. En Colab: **File** → **Upload notebook**
2. Selecciona: `train_gpu_cloud.ipynb` (de tu carpeta local)
3. Upload

### 3️⃣ Activar GPU T4 (CRÍTICO)

1. **Runtime** → **Change runtime type**
2. **Hardware accelerator**: Selecciona **GPU**
3. **GPU type**: Selecciona **T4** (gratis)
4. Click **Save**

**Verificar GPU**:
```python
import torch
print(torch.cuda.is_available())  # Debe ser True
print(torch.cuda.get_device_name(0))  # Debe mostrar "Tesla T4"
```

### 4️⃣ Ejecutar Todo el Notebook

**Forma Rápida**:
- **Runtime** → **Run all**
- ☕ Toma un café, tardará ~5-10 minutos en setup

**O ejecutar celda por celda**:
- Click en cada celda → `Shift + Enter`
- Ve validando que todo funciona

### 5️⃣ ¡Entrenar! (1-2 días)

El notebook hará automáticamente:
1. ✅ Verificar GPU
2. ✅ Clonar repositorio desde GitHub
3. ✅ Instalar PyTorch y dependencias
4. ✅ Descargar Tiny ImageNet (200 MB)
5. ✅ Crear splits train/val/test
6. ✅ Montar Google Drive (opcional, para guardar checkpoints)
7. ✅ **ENTRENAR 100 épocas** 🎯
8. ✅ Guardar checkpoints
9. ✅ Mostrar resultados

---

## 📊 Durante el Entrenamiento

Verás algo como:

```
======================================================================
VERIFICANDO GPU
======================================================================
✅ GPU Disponible: Tesla T4
✅ Memoria GPU: 15.0 GB
✅ CUDA Version: 12.2
✅ PyTorch Version: 2.1.0
======================================================================

Clonando repositorio...
✅ Repositorio clonado exitosamente

Instalando dependencias...
✅ Dependencias instaladas

Descargando Tiny ImageNet...
✅ 10,000 imágenes organizadas

Creando splits...
  Train: 8000 (80%)
  Val:   1000 (10%)
  Test:  1000 (10%)
✅ Splits creados

======================================================================
ENTRENAMIENTO INICIADO
======================================================================

Epoch 1/100: 100%|██████████| 500/500 [02:15<00:00, 3.70it/s]
  L_G: 11.5842  L_D: -0.0002  PSNR: 28.34 dB  SSIM: 0.6543
  Val PSNR: 29.12 dB  Val SSIM: 0.6821
  ✓ Checkpoint guardado (best PSNR)

Epoch 2/100: 100%|██████████| 500/500 [02:18<00:00, 3.61it/s]
  L_G: 10.2341  L_D: -0.0018  PSNR: 31.45 dB  SSIM: 0.7234
  Val PSNR: 32.01 dB  Val SSIM: 0.7456
  ✓ Checkpoint guardado (best PSNR)

...

Epoch 100/100: 100%|██████████| 500/500 [02:12<00:00, 3.77it/s]
  L_G: 2.1234  L_D: -0.1523  PSNR: 56.78 dB  SSIM: 0.9387
  Val PSNR: 57.45 dB  Val SSIM: 0.9412
  
======================================================================
🎉 ENTRENAMIENTO COMPLETADO!
======================================================================
📊 Mejor PSNR: 57.45 dB (Época 98)
📊 Mejor SSIM: 0.9412 (Época 100)
📍 Checkpoints guardados en Drive
```

---

## ⏱️ Tiempo Estimado

### Setup (una sola vez):
- Subir notebook: 1 min
- Activar GPU: 30 seg
- Clonar repo + instalar deps: 2-3 min
- Descargar dataset: 2-3 min
- **Total setup**: ~5-10 min

### Entrenamiento:
- **T4 GPU (Colab gratis)**: ~2-3 min por época
- **100 épocas**: ~4-5 horas ⚡
- **Con estrategia 4:1**: ~1-2 días

**💡 Tip**: Colab gratis te da ~12 horas continuas. Si se desconecta, los checkpoints en Drive se mantienen y puedes retomar.

---

## 💾 Guardar Checkpoints en Google Drive

El notebook preguntará si quieres montar Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Ventajas**:
- ✅ No pierdes progreso si Colab se desconecta
- ✅ Checkpoints se guardan cada época
- ✅ Puedes descargarlos después desde Drive

**Ubicación en Drive**:
```
Mi unidad/DCT-GAN-Checkpoints/
├── checkpoint_best_psnr.pth
├── checkpoint_best_ssim.pth
└── checkpoint_latest.pth
```

---

## 📥 Descargar Resultados

### Opción 1: Desde Drive
1. Abre Google Drive
2. Carpeta: `DCT-GAN-Checkpoints`
3. Descarga los `.pth` que necesites

### Opción 2: Ejecutar celda final del notebook
```python
from google.colab import files

# Comprimir
!zip -r checkpoints_final.zip checkpoints/

# Descargar
files.download('checkpoints_final.zip')
```

---

## 🔍 Verificar Progreso

### Ver métricas en tiempo real:

**TensorBoard** (en el notebook):
```python
%load_ext tensorboard
%tensorboard --logdir logs
```

Verás gráficas de:
- PSNR vs épocas
- SSIM vs épocas
- Pérdidas (generator, discriminator)

---

## ⚠️ Problemas Comunes

### Error: "No GPU available"
**Solución**: Runtime → Change runtime type → GPU

### Error: "Runtime disconnected"
**Solución**: 
- Colab gratis desconecta después de 12h de inactividad
- Si montaste Drive, tus checkpoints están seguros
- Re-ejecuta desde la celda de entrenamiento
- El código detectará el último checkpoint y continuará

### Error: "Out of memory"
**Solución**: Reduce batch size en `configs/base_config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce de 32 a 16
```

### Dataset no se descarga
**Solución alternativa**:
```python
# En el notebook, usar Kaggle en lugar de Stanford
!kaggle datasets download -d xuliangchina/tiny-imagenet-200
```

---

## 📊 Métricas Objetivo (Paper)

Al terminar 100 épocas, deberías ver:

| Métrica | Target (Paper) | Esperado |
|---------|---------------|----------|
| **PSNR** | 58.27 dB | ~55-58 dB |
| **SSIM** | 0.942 | ~0.93-0.94 |
| **Recovery PSNR** | >40 dB | ~38-42 dB |
| **Parámetros** | 50K | 46K (-7.9%) |

**¿Por qué pueden diferir?**
- Dataset diferente (Tiny ImageNet vs ImageNet completo)
- 10K imágenes vs 50K del paper
- Implementación puede tener diferencias sutiles

---

## 🚀 Después del Entrenamiento

### 1. Analizar Resultados
```python
# En nueva celda
import torch
from src.models.gan import DCTGAN

# Cargar mejor modelo
checkpoint = torch.load('checkpoints/checkpoint_best_psnr.pth')
print(f"Mejor PSNR: {checkpoint['metrics']['psnr']:.2f} dB")
print(f"Época: {checkpoint['epoch']}")
```

### 2. Probar Esteganografía
```python
# Ocultar imagen
cover = load_image('test/cover.jpg')
secret = load_image('test/secret.jpg')

model = DCTGAN(...).cuda()
model.load_state_dict(checkpoint['model_state'])

stego, recovered = model(cover, secret)

# Ver resultados
show_images([cover, stego, secret, recovered])
print(f"PSNR (cover vs stego): {calculate_psnr(cover, stego):.2f} dB")
```

### 3. Compartir Resultados
- Sube checkpoints a Drive
- Comparte carpeta con colaboradores
- Actualiza README con tus métricas

---

## 🎓 Checklist Final

Antes de cerrar Colab:

- [ ] Entrenamiento completo (100/100 épocas)
- [ ] Checkpoints guardados en Drive
- [ ] Mejor PSNR anotado
- [ ] Mejor SSIM anotado
- [ ] Gráficas de TensorBoard capturadas
- [ ] Checkpoints descargados (backup local)

---

## 📞 Ayuda

**Repositorio GitHub**: https://github.com/jaimelopezm-star/DCT-GAN-Mobile

**Issues**: Si algo no funciona, abre un issue en GitHub

**Documentación adicional**:
- [README.md](https://github.com/jaimelopezm-star/DCT-GAN-Mobile/blob/main/README.md)
- [GPU_TRAINING_GUIDE.md](https://github.com/jaimelopezm-star/DCT-GAN-Mobile/blob/main/GPU_TRAINING_GUIDE.md)

---

## ✨ ¡Éxito!

Si terminaste el entrenamiento con PSNR >55 dB, ¡felicitaciones! 🎉

Has replicado exitosamente el paper DCT-GAN Mobile.

**Próximos pasos sugeridos**:
1. Probar con dataset más grande (ImageNet completo)
2. Experimentar con diferentes arquitecturas
3. Evaluar robustez JPEG
4. Comparar con métodos SOTA

¡Buena suerte! 🚀
