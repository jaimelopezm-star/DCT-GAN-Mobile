# 🚀 Entrenamiento en GPU Cloud - Guía Rápida

## ⚡ Velocidad por GPU

| Plataforma | GPU | Costo | Tiempo (100 épocas) | Velocidad |
|------------|-----|-------|---------------------|-----------|
| **Google Colab** | T4 | Gratis | ~1-2 días | 2-3 min/época |
| **Kaggle** | P100 | Gratis (30h/sem) | ~1 día | 1-2 min/época |
| **RunPod** | A100 | $0.79/h | ~12 horas | 30 seg/época |
| **Vast.ai** | RTX 4090 | $0.40/h | ~8 horas | 20 seg/época |
| **Tu PC (CPU)** | - | - | ~19 días | 4.6 hrs/época |

---

## 📋 Opción 1: Google Colab (Recomendada - Gratis)

### Paso 1: Subir a GitHub

```powershell
# En tu PC local (DCT-GAN-Mobile/)
cd "C:\Users\Lopez\OneDrive\Documents\MAESTRIA\Semestre 2\Esteganografia y marcas de agua\papers esteganografia DCT\DCT-GAN-Mobile"

# Inicializar Git
git init
git add .
git commit -m "Initial commit - DCT-GAN Mobile"

# Subir a GitHub (crea el repo primero en github.com)
git remote add origin https://github.com/TU_USUARIO/DCT-GAN-Mobile.git
git branch -M main
git push -u origin main
```

### Paso 2: Usar en Colab

1. **Abrir Colab**: https://colab.research.google.com
2. **Upload notebook**: Sube `train_gpu_cloud.ipynb`
3. **Activar GPU**: 
   - **Runtime** → **Change runtime type**
   - **Hardware accelerator**: **GPU** (T4)
   - **Save**
4. **Modificar celda 2** del notebook:
   ```python
   # Descomentar y cambiar
   !git clone https://github.com/TU_USUARIO/DCT-GAN-Mobile.git
   ```
5. **Run All**: **Runtime** → **Run all**
6. **Esperar**: ~1-2 días (puedes cerrar la pestaña, seguirá corriendo)

### Paso 3: Descargar Modelo

1. Cuando termine, ejecuta celda final (descargar checkpoints)
2. Se descarga `checkpoints_final.zip` con el modelo entrenado
3. Extraer y usar en tu PC local

---

## 📦 Opción 2: Kaggle (30 horas gratis/semana)

### Ventajas
- GPU P100 (más rápida que T4 de Colab)
- 30 horas/semana gratis
- Puede descargar datasets directamente de Kaggle

### Pasos

1. **Ir a Kaggle**: https://www.kaggle.com
2. **New Notebook** → Import `train_gpu_cloud.ipynb`
3. **Settings** (derecha):
   - **Accelerator**: GPU P100
   - **Internet**: ON
4. **Modificar repo** en celda 2
5. **Run All**
6. **Descargar outputs** al finalizar

---

## 🔥 Opción 3: RunPod / Vast.ai (Más rápido - de pago)

### RunPod (GPU A100 - $0.79/hora)

1. **Crear cuenta**: https://runpod.io
2. **Deploy Pod**:
   - Template: **PyTorch**
   - GPU: **A100** (40GB)
   - Disk: 50 GB
3. **Connect** → Jupyter Notebook
4. **Upload** `train_gpu_cloud.ipynb`
5. **Modificar** celda 2 (clonar tu repo)
6. **Run All**
7. **Tiempo**: ~12 horas (~$9.50 total)
8. **Descargar checkpoints** antes de detener pod

### Vast.ai (GPU RTX 4090 - $0.40/hora)

Similar a RunPod pero más barato:
1. https://vast.ai
2. Buscar: **RTX 4090** + **Jupyter**
3. Rent instance
4. Upload notebook
5. Run (~8 horas, ~$3.20)

---

## 📊 Qué Esperar

### Progreso Típico (GPU)

**Épocas 1-20**:
- PSNR: 10 dB → 40 dB
- SSIM: 0.3 → 0.80
- Imágenes visualmente mejorando

**Épocas 20-60**:
- PSNR: 40 dB → 52 dB
- SSIM: 0.80 → 0.90
- Calidad casi imperceptible

**Épocas 60-100**:
- PSNR: 52 dB → 54-55 dB
- SSIM: 0.90 → 0.92
- Refinamiento fino

### Checkpoints Automáticos

El script guarda automáticamente:
- `checkpoint_latest.pth` - Último modelo
- `checkpoint_best.pth` - Mejor PSNR
- `checkpoint_epoch_*.pth` - Cada 10 épocas

---

## 🛡️ Tips Importantes

### Google Colab
- ⚠️ **Límite 12 horas continuas** - El notebook se reinicia
- ✅ **Solución**: Guarda en Drive, re-ejecuta desde checkpoint
- ✅ **Truco**: Ejecuta en 2 sesiones (50 épocas cada una)

### Kaggle
- ⚠️ **Límite 30 horas/semana**
- ✅ **Planifica**: 100 épocas = ~24 horas (cabe en una semana)

### RunPod/Vast.ai
- ✅ Sin límites, pagas por hora
- ✅ Puedes pausar y resume
- ⚠️ **Guarda checkpoints** frecuentemente

---

## 📂 Estructura Necesaria en GitHub

```
DCT-GAN-Mobile/
├── train.py                    # Script de entrenamiento
├── train_gpu_cloud.ipynb       # Notebook para Colab/Kaggle
├── requirements_gpu.txt        # Dependencias
├── configs/
│   └── base_config.yaml        # Configuración
├── models/
│   ├── encoder.py
│   ├── decoder.py
│   ├── discriminator.py
│   └── __init__.py
├── utils/
│   ├── trainer.py
│   ├── metrics.py
│   ├── losses.py
│   └── __init__.py
└── README.md
```

---

## ❓ Preguntas Frecuentes

**P: ¿Puedo entrenar gratis?**
R: Sí, con Google Colab o Kaggle (ambos gratuitos)

**P: ¿Cuánto tiempo en Colab?**
R: ~1-2 días (puedes ejecutar en 2 sesiones de 50 épocas)

**P: ¿Mejor opción de pago?**
R: Vast.ai con RTX 4090 (~$3-4 total, 8 horas)

**P: ¿Y si se interrumpe?**
R: El código guarda checkpoints cada época. Re-ejecuta desde ahí.

**P: ¿Necesito GPU de 40GB?**
R: No, con 16GB basta (T4, P100, 4090 funcionan)

---

## 🎯 Próximos Pasos

1. **Ahora**: Sube código a GitHub
2. **Hoy**: Ejecuta en Colab (gratis) - 50 épocas
3. **Mañana**: Continúa otras 50 épocas
4. **Resultado**: Modelo entrenado en ~2 días gratis

¿Listo para empezar? Sigue la **Opción 1** (Google Colab) 🚀
