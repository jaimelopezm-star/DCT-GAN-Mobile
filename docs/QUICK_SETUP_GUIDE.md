# 🚀 Quick Setup - Testing + Descarga Automática

## ¿Qué hace este script?

1. ✅ **Testing rápido** (5-10 min)
   - Ejecuta 2 epochs con datos sintéticos
   - Valida que TODO el código funciona
   - Sin descargar nada

2. 🔽 **Descarga automática** (1-2 horas)
   - Descarga ImageNet en NUEVA VENTANA
   - Puedes seguir trabajando mientras descarga
   - Prepara splits automáticamente

---

## 🎯 Uso Rápido (Windows)

### Opción 1: Script PowerShell (RECOMENDADO)

```powershell
# Ejecutar todo automáticamente
.\quick_setup.ps1
```

**Esto hace:**
1. Testing rápido (2 epochs sintético) → 5-10 min
2. Te pregunta si quieres descargar ImageNet
3. Si dices "Sí" → abre NUEVA VENTANA con descarga
4. Puedes cerrar la ventana original, descarga continúa

---

### Opción 2: Script Python (Cross-platform)

```powershell
# Descarga en background + testing
python quick_setup.py

# Solo testing (sin descarga)
python quick_setup.py --skip-download

# Solo descarga (sin testing)
python quick_setup.py --skip-test

# Con monitoreo de descarga
python quick_setup.py --monitor
```

---

## 📊 ¿Qué valida el testing rápido?

```
✅ Modelos cargan correctamente (Encoder, Decoder, Discriminator)
✅ Training loop funciona (4:1 strategy)
✅ Loss functions calculan bien (HybridLoss)
✅ Métricas funcionan (PSNR, SSIM, BER)
✅ Checkpoints se guardan
✅ Logs se generan
✅ DataLoaders funcionan
```

**Resultado:** Después de 5-10 minutos sabes si el código tiene errores.

---

## 🔽 Descarga en Background

### Descarga en Nueva Ventana (PowerShell)

```powershell
# Manual (si no usaste quick_setup.ps1)
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; & .venv\Scripts\Activate.ps1; python scripts/download_imagenet.py --method kaggle; python scripts/prepare_dataset.py"
```

**Ventajas:**
- Nueva ventana independiente
- Puedes cerrar la ventana original
- Descarga continúa en background
- Prepara splits automáticamente

### Monitorear Progreso

```powershell
# Ver log de descarga
Get-Content logs\download_imagenet.log -Tail 20 -Wait

# Ver si el proceso está corriendo
Get-Process | Where-Object {$_.ProcessName -eq "python"}
```

---

## 📋 Workflow Completo

### Día 1 - Mañana (15 minutos)

```powershell
# 1. Testing rápido
.\quick_setup.ps1

# 2. Responde "S" cuando pregunte por descarga
# 3. ¡Listo! Descarga corre en nueva ventana
```

### Día 1 - Tarde (verificar)

```powershell
# Verificar que descarga terminó
ls data\imagenet2012\splits\train\all\

# Debería mostrar 40,000 imágenes
```

### Día 2+ - Entrenar

```powershell
# Entrenar con ImageNet real
python train.py --config configs/base_config.yaml --dataset imagenet

# Esperar 2-7 días...
```

---

## 🎯 Comparación de Métodos

| Método | Testing | Descarga | Ventajas |
|--------|---------|----------|----------|
| **quick_setup.ps1** | ✅ Auto | ✅ Auto | Más fácil, todo en uno |
| **quick_setup.py** | ✅ Auto | ✅ Auto | Cross-platform, más opciones |
| **Manual** | Manual | Manual | Control total |

---

## 🆘 Solución de Problemas

### Error: "No se puede ejecutar .ps1"

```powershell
# Habilitar scripts PowerShell (ejecutar como Admin)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Testing falla

```powershell
# Ver logs detallados
Get-Content logs_test\training_*.log

# Verificar errores
python train.py --config configs/test_config.yaml --dataset synthetic
```

### Descarga muy lenta

```powershell
# Verificar conexión
# Kaggle limita velocidad, es normal 1-2 horas

# Pausar descarga
# Ctrl+C en la ventana de descarga

# Reanudar después
python scripts/download_imagenet.py --method kaggle
```

### Quiero cancelar todo

```powershell
# Matar proceso de descarga
Get-Process python | Stop-Process

# Limpiar archivos temporales
Remove-Item data\imagenet2012\* -Recurse -Force
```

---

## ✨ Ventajas de este Método

1. **Validación inmediata:** Sabes en 10 minutos si hay errores
2. **Descarga en background:** No bloques tu trabajo
3. **Automático:** Prepara splits solo
4. **Seguro:** Si testing falla, no descarga (ahorra tiempo)
5. **Flexible:** Puedes pausar/reanudar

---

## 📖 Ver También

- [QUICKSTART.md](QUICKSTART.md) - Guía manual paso a paso
- [ROADMAP.md](ROADMAP.md) - Plan completo de revisiones
- [STATUS.md](STATUS.md) - Estado actual del proyecto

---

**¿Listo?**

```powershell
.\quick_setup.ps1
```

🚀 ¡En 10 minutos sabrás si todo funciona!
