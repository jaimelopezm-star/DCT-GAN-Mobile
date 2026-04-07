# 📂 Estructura de Checkpoints - Colab vs RunPod

## ¿Por qué carpetas separadas?

**Problema sin separación**:
```
checkpoints/
  ├── checkpoint_epoch_001.pth  # ¿De Colab o RunPod? 🤔
  ├── checkpoint_epoch_002.pth  # Se sobrescriben si corren paralelo ❌
  └── best_model.pth            # Conflicto ❌
```

**Solución con separación**:
```
DCT-GAN-Mobile/
├── checkpoints/              # Colab T4 (gratis, 20 horas)
│   ├── checkpoint_epoch_001.pth
│   ├── checkpoint_epoch_002.pth
│   ├── ...
│   ├── checkpoint_epoch_100.pth
│   └── best_model.pth
│
├── checkpoints_runpod/       # RunPod RTX 4090 ($/1.80, 3 horas)
│   ├── checkpoint_epoch_001.pth
│   ├── checkpoint_epoch_002.pth
│   ├── ...
│   ├── checkpoint_epoch_100.pth
│   └── best_model.pth
│
├── logs/                     # TensorBoard Colab
└── logs_runpod/              # TensorBoard RunPod
```

---

## 🔄 Entrenamientos Paralelos (Tu Estrategia)

### Ventajas de correr ambos:

1. **Backup automático**: Si Colab truena (desconexión, límite 12h), tienes RunPod
2. **Comparación**: Verificar que ambas GPUs lleguen a mismas métricas
3. **Velocidad**: RunPod termina en 3h, no esperas 20h de Colab
4. **Validación**: Si ambos dan ~58 dB PSNR, confirma que el código es correcto

### Escenarios posibles:

#### ✅ Escenario 1: Ambos exitosos
```
Colab:  PSNR 58.2 dB  SSIM 0.941  (20 horas)  $0
RunPod: PSNR 58.4 dB  SSIM 0.943  (3 horas)   $1.80

✅ Usar el modelo de RunPod (ligeramente mejor + terminó primero)
✅ Tienes respaldo de Colab si necesitas
```

#### ⚠️ Escenario 2: Colab se desconectó
```
Colab:  Desconectado en época 45  ❌
RunPod: PSNR 58.1 dB (completo)   ✅

✅ $1.80 bien gastados - salvó tu proyecto
✅ No perdiste 20 horas esperando
```

#### 🎉 Escenario 3: Colab OK pero lento
```
Colab:  Época 18/100 (después de 8 horas)  🐌
RunPod: Época 100/100 ✅ (después de 3 horas) 🚀

✅ Ya puedes analizar resultados mientras Colab sigue
✅ Si RunPod está bien, puedes cancelar Colab (ahorrar recursos)
```

---

## 📊 Comparación de Archivos

### Checkpoint file structure

Cada checkpoint (.pth) contiene:
```python
{
    'epoch': 100,
    'encoder_state': {...},
    'decoder_state': {...},
    'discriminator_state': {...},
    'optimizer_G_state': {...},
    'optimizer_D_state': {...},
    'val_metrics': {
        'psnr': 58.27,
        'ssim': 0.942,
        'rmse': 0.0012
    }
}
```

### Tamaños aproximados:

```
Colab (checkpoints/):
  checkpoint_epoch_001.pth:  ~700 KB
  checkpoint_epoch_050.pth:  ~700 KB
  checkpoint_epoch_100.pth:  ~700 KB
  best_model.pth:            ~700 KB
  TOTAL:                     ~70 MB (100 checkpoints)

RunPod (checkpoints_runpod/):
  checkpoint_epoch_001.pth:  ~700 KB
  checkpoint_epoch_050.pth:  ~700 KB
  checkpoint_epoch_100.pth:  ~700 KB
  best_model.pth:            ~700 KB
  TOTAL:                     ~70 MB (100 checkpoints)

TOTAL AMBOS:                 ~140 MB
```

---

## 🔍 Cómo comparar resultados

### Script de comparación:

```python
import torch
from pathlib import Path

# Cargar checkpoints finales
colab_path = Path("checkpoints/best_model.pth")
runpod_path = Path("checkpoints_runpod/best_model.pth")

print("="*70)
print("COMPARACIÓN COLAB vs RUNPOD")
print("="*70)

if colab_path.exists():
    colab_ckpt = torch.load(colab_path, map_location='cpu')
    colab_metrics = colab_ckpt.get('val_metrics', {})
    print(f"\n📘 Colab T4:")
    print(f"   Época: {colab_ckpt.get('epoch', 'N/A')}")
    print(f"   PSNR: {colab_metrics.get('psnr', 'N/A'):.2f} dB")
    print(f"   SSIM: {colab_metrics.get('ssim', 'N/A'):.4f}")
    print(f"   RMSE: {colab_metrics.get('rmse', 'N/A'):.4f}")
else:
    print("\n📘 Colab T4: No disponible (aún corriendo o cancelado)")

if runpod_path.exists():
    runpod_ckpt = torch.load(runpod_path, map_location='cpu')
    runpod_metrics = runpod_ckpt.get('val_metrics', {})
    print(f"\n🚀 RunPod RTX 4090:")
    print(f"   Época: {runpod_ckpt.get('epoch', 'N/A')}")
    print(f"   PSNR: {runpod_metrics.get('psnr', 'N/A'):.2f} dB")
    print(f"   SSIM: {runpod_metrics.get('ssim', 'N/A'):.4f}")
    print(f"   RMSE: {runpod_metrics.get('rmse', 'N/A'):.4f}")
else:
    print("\n🚀 RunPod RTX 4090: No disponible")

# Comparar si ambos existen
if colab_path.exists() and runpod_path.exists():
    psnr_diff = runpod_metrics['psnr'] - colab_metrics['psnr']
    ssim_diff = runpod_metrics['ssim'] - colab_metrics['ssim']
    
    print(f"\n📊 Diferencia:")
    print(f"   PSNR: {psnr_diff:+.2f} dB")
    print(f"   SSIM: {ssim_diff:+.4f}")
    
    if abs(psnr_diff) < 1.0 and abs(ssim_diff) < 0.01:
        print(f"\n✅ Resultados CASI IDÉNTICOS")
        print(f"   Ambos modelos son equivalentes - usar cualquiera")
    elif psnr_diff > 0:
        print(f"\n🚀 RunPod MEJOR (+{psnr_diff:.2f} dB)")
    else:
        print(f"\n📘 Colab MEJOR (+{abs(psnr_diff):.2f} dB)")

print("="*70)
```

---

## 💾 Backup y Descarga

### Desde Colab:

```python
# En Colab, comprimir checkpoints
!zip -r checkpoints_colab.zip checkpoints/

# Descargar
from google.colab import files
files.download('checkpoints_colab.zip')
```

### Desde RunPod:

```python
# Ya lo hace automáticamente en train_runpod.ipynb
!zip -r checkpoints_runpod.zip checkpoints_runpod/

# Descargar desde Jupyter Lab file browser
# Right-click → Download
```

---

## 📂 Organización Final Recomendada

### En tu PC local:

```
DCT-GAN-Proyecto/
├── codigo/
│   └── DCT-GAN-Mobile/         # Repo de GitHub
│
├── checkpoints/
│   ├── colab_t4/               # Descargar de Colab
│   │   ├── checkpoint_epoch_100.pth
│   │   └── best_model.pth
│   │
│   └── runpod_rtx4090/         # Descargar de RunPod
│       ├── checkpoint_epoch_100.pth
│       └── best_model.pth
│
├── experimento_final/
│   └── best_model.pth          # El MEJOR de ambos
│
└── resultados/
    ├── metricas_colab.txt      # PSNR, SSIM
    ├── metricas_runpod.txt
    └── comparacion.png          # Gráfica comparativa
```

---

## ✅ Checklist de Validación

Antes de decidir cuál usar:

- [ ] Ambos entrenamientos completados (100 épocas)
- [ ] PSNR de ambos ≥ 55 dB (mínimo aceptable)
- [ ] SSIM de ambos ≥ 0.90 (mínimo aceptable)
- [ ] Diferencia entre ambos < 2 dB (confirma consistencia)
- [ ] Checkpoints descargados y respaldados
- [ ] Comparación visual de imágenes generadas

**Si todo checks**: ¡Éxito! Usar cualquiera (preferiblemente el mejor)

---

## 🎯 Decisión Final

### Usar RunPod si:
- ✅ PSNR ≥ 58 dB
- ✅ Terminó completo
- ✅ Similar o mejor que Colab

### Usar Colab si:
- ✅ RunPod tuvo problemas
- ✅ Métricas ligeramente mejores
- ✅ No hay diferencia significativa (gratis vs $1.80)

### Usar AMBOS en paper/tesis:
- 📊 "Entrenamiento se replicó en 2 GPUs diferentes"
- 📊 "Resultados consistentes: T4=58.2±0.1 dB, RTX4090=58.4±0.1 dB"
- 📊 "Confirma robustez del modelo"

---

## 💡 Conclusión

**Tu estrategia es EXCELENTE** porque:
1. ✅ Minimizas riesgo (doble backup)
2. ✅ Ahorras tiempo (RunPod 3h vs Colab 20h)
3. ✅ Validas resultados (2 GPUs independientes)
4. ✅ Bajo costo ($1.80 es inversión mínima)

**Recomendación**: Ejecuta ambos ahora que ya están configurados 🚀
