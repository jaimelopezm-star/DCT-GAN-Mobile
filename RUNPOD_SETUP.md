# 🚀 RunPod Setup - Guía Rápida

## ⚡ Por qué RunPod

**Ventajas vs Colab**:
- ✅ **Mucho más rápido**: RTX 4090 es 7× más rápida que T4
- ✅ **Sin límites de sesión**: No hay límite de 12 horas
- ✅ **No necesitas mantener laptop encendida**: Corre en la nube
- ✅ **Checkpoints separados**: No se mezclan con Colab (`checkpoints_runpod/`)

**Costos comparados**:
| GPU | Tiempo | Costo | vs Colab |
|-----|--------|-------|----------|
| Colab T4 | 20h | $0 (gratis) | Baseline |
| RunPod RTX 4090 | 3h | $1.50-1.80 | 7× más rápido |
| RunPod RTX 5090 | 2h | $1.50-1.80 | 10× más rápido |

---

## 📋 Paso 1: Crear cuenta en RunPod

1. Ir a: https://www.runpod.io/
2. **Sign Up** (puedes usar Google/GitHub)
3. **Add Credits**: Mínimo $10 (alcanza para ~17 entrenamientos en RTX 4090)
   - Aceptan tarjeta de crédito/débito
   - También crypto si prefieres

---

## 🖥️ Paso 2: Rentar GPU

### Opción A - GPU Recomendada (RTX 4090)

1. Click en **Deploy** (barra izquierda)
2. En **GPU Type**, buscar: `RTX 4090`
3. **Filtros**:
   - VRAM: Mínimo 20 GB ✅ (RTX 4090 tiene 24 GB)
   - Precio: Ordenar por más barato
   - Availability: Medium o High

4. Seleccionar pod con:
   - **GPU**: RTX 4090
   - **Precio**: ~$0.59/hr ($0.50/hr "green" si hay disponible)
   - **Location**: No importa (elige el más barato)

### Características del Pod

**Container Template**: Seleccionar `RunPod PyTorch`
   - Ya incluye Python + PyTorch + CUDA
   - Jupyter Lab preinstalado

**Container Disk**: `20 GB` suficiente
   - Dataset: 200 MB
   - Código: <1 MB
   - Checkpoints: ~1 GB total
   - PyTorch ya viene instalado

**Volume (Persistente)**: OPCIONAL
   - ⚠️ Cuesta extra: $0.10/GB/mes
   - 💡 **NO NECESARIO** si descargas checkpoints al terminar
   - Solo útil si planeas entrenar múltiples veces

**Expose HTTP/TCP Ports**: No cambiar (defaults OK)

5. Click **Deploy On-Demand**

---

## 🔗 Paso 3: Conectar a Jupyter

1. Espera ~30-60 segundos mientras el pod inicia
2. En tu pod, verás botón **Connect**
3. Click **Connect** → **Jupyter Lab** (puerto 8888)
   - Se abre Jupyter Lab en nueva pestaña
   - Ya está autenticado (no requiere token)

---

## 📤 Paso 4: Subir notebook

### Opción A - Desde GitHub (RECOMENDADO)

En Jupyter Lab terminal:
```bash
git clone https://github.com/jaimelopezm-star/DCT-GAN-Mobile.git
cd DCT-GAN-Mobile
```

Luego abrir: `train_runpod.ipynb`

### Opción B - Upload Manual

1. En Jupyter Lab: **Upload** button (arriba)
2. Seleccionar `train_runpod.ipynb` desde tu PC
3. Esperar a que suba (~100 KB)

---

## ▶️ Paso 5: Ejecutar entrenamiento

1. Abrir `train_runpod.ipynb` en Jupyter Lab
2. **Run** → **Run All Cells** (o Shift+Enter en cada celda)

**El notebook hará automáticamente**:
- ✅ Verificar GPU (debe mostrar RTX 4090)
- ✅ Clonar repositorio desde GitHub
- ✅ Instalar dependencias
- ✅ Descargar Tiny ImageNet (200 MB)
- ✅ Crear splits train/val/test
- ✅ Entrenar 100 épocas
- ✅ Guardar checkpoints en `checkpoints_runpod/`
- ✅ Logs en `logs_runpod/`

**Tiempo estimado**: 2-3 horas en RTX 4090

---

## 📊 Paso 6: Monitorear progreso

### Opción A - Output del Notebook

Verás en tiempo real:
```
Época 1/100: 100% |██████████| 250/250 [00:51<00:00]
  Loss_G: 10.49  Loss_D: 0.00  PSNR: 15.32 dB  SSIM: 0.6210
```

### Opción B - TensorBoard (opcional)

En nueva celda:
```python
%load_ext tensorboard
%tensorboard --logdir logs_runpod
```

---

## 💾 Paso 7: Descargar checkpoints

### Cuando termine el entrenamiento:

**Opción A - Zip en el notebook**

El notebook automáticamente ejecuta la celda final que:
```python
!zip -r checkpoints_runpod.zip checkpoints_runpod/
```

Luego en Jupyter Lab:
1. File browser (izquierda)
2. Right-click en `checkpoints_runpod.zip`
3. **Download**

**Tamaño**: ~500-800 MB (comprimido)

### Opción B - Download individual

Si quieres solo el mejor modelo:
```python
# En nueva celda
from IPython.display import FileLink
FileLink('checkpoints_runpod/best_model.pth')
```

---

## 🛑 Paso 8: DETENER el pod

**⚠️ MUY IMPORTANTE** - Detén el pod cuando termine para no seguir pagando

1. Volver a panel de RunPod (runpod.io/console/pods)
2. Tu pod activo mostrará: `Running` 🟢
3. Click en **3 dots** (...) → **Terminate Pod**
4. Confirmar

**Costo final**: Verás en dashboard cuánto gastaste
- Ejemplo: 3 horas × $0.59/hr = $1.77

---

## 🔍 Paso 9: Comparar con Colab

Ahora tendrás **DOS entrenamientos paralelos**:

### En Colab:
- Checkpoints: `checkpoints/`
- Logs: `logs/`
- Status: Puede seguir corriendo en Colab T4

### En RunPod:
- Checkpoints: `checkpoints_runpod/`
- Logs: `logs_runpod/`
- Status: Terminado en 2-3 horas

### Comparar métricas:

```python
import torch

# Cargar último checkpoint de Colab
colab_ckpt = torch.load('checkpoints/checkpoint_epoch_100.pth')
print(f"Colab - PSNR: {colab_ckpt['val_metrics']['psnr']:.2f} dB")

# Cargar último checkpoint de RunPod  
runpod_ckpt = torch.load('checkpoints_runpod/checkpoint_epoch_100.pth')
print(f"RunPod - PSNR: {runpod_ckpt['val_metrics']['psnr']:.2f} dB")
```

**Usar el mejor** (probablemente sean casi iguales)

---

## 💡 Tips Avanzados

### Si el pod se desconecta (raro pero posible)

Los checkpoints se guardan cada época, así que puedes:
```python
# Reanudar desde último checkpoint
python train.py \
  --config configs/base_config.yaml \
  --checkpoint_dir checkpoints_runpod \
  --resume checkpoints_runpod/checkpoint_epoch_045.pth \
  --epochs 100
```

### Para entrenamientos más largos

Si planeas hacer múltiples experimentos:
- Considera rentar con **Volume** (persistencia)
- Costo: +$0.10/GB/mes por 50 GB = $5/mes extra
- Ventaja: Datasets y código persisten entre sesiones

### Alternativas más baratas

**L4** ($0.39/hr):
- Tiempo: 4-5 horas
- Costo total: $1.56-1.95
- Ahorro: $0.21 vs RTX 4090
- Trade-off: +1-2 horas más lento

**L40** ($0.79/hr):
- Entre L4 y RTX 4090
- Tiempo: ~2.5 horas
- Costo: ~$1.98

---

## 🆘 Troubleshooting

### No hay RTX 4090 disponible

**Solución**: Usar alternativas:
1. **RTX 5090** ($0.89/hr) - Aún más rápida
2. **L4** ($0.39/hr) - Más lenta pero más barata
3. **Esperar 10-20 min** - Las GPUs se liberan frecuentemente

### Error: "CUDA out of memory"

**Muy raro con RTX 4090 (24 GB)**, pero si pasa:
```python
# Reducir batch size en train_runpod.ipynb
"--batch_size", "16",  # En vez de 32
```

### Jupyter Lab no carga

1. Verifica que el pod esté `Running` 🟢
2. Espera 60 segundos (inicialización)
3. Refresh la página
4. Si persiste: Terminate pod y crea nuevo

### Training muy lento (>10 min/epoch)

Verifica que tengas GPU correcta:
```python
import torch
print(torch.cuda.get_device_name(0))
# Debe decir: "NVIDIA GeForce RTX 4090"
```

Si dice otra GPU:
- Detener pod
- Rentar RTX 4090 específicamente

---

## 📈 Métricas Esperadas

### En RTX 4090:

**Tiempo por época**: ~60-120 segundos
**100 épocas**: 2.5-3.5 horas total

**Costo final**: $1.50-2.00

**Métricas finales** (época 100):
- PSNR: ~58 dB (target del paper)
- SSIM: ~0.94 (target del paper)

Si obtienes estas métricas: ✅ **ÉXITO TOTAL**

---

## ✅ Checklist Final

Antes de terminar, verifica que tengas:

- [x] Checkpoints descargados (`checkpoints_runpod.zip`)
- [x] Pod detenido (no sigue cobrando)
- [x] Métricas finales guardadas
- [x] Comparación con Colab (si aplica)

**¡Listo!** Ahora tienes tu modelo entrenado profesionalmente 🎉

---

## 🔗 Links Útiles

- **RunPod Console**: https://www.runpod.io/console/pods
- **Pricing**: https://www.runpod.io/gpu-instance/pricing
- **Docs**: https://docs.runpod.io/
- **Support Discord**: https://discord.gg/runpod

---

**Tiempo total de setup**: 5-10 minutos  
**Tiempo de entrenamiento**: 2-3 horas  
**Costo**: $1.50-1.80  
**vs Colab**: 7× más rápido (pero Colab es gratis)
