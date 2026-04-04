# 🎯 EJECUTA ESTO AHORA

## Opción 1: Setup Automático (UN COMANDO)

```powershell
.\quick_setup.ps1
```

**Si da error de permisos:**

```powershell
# Habilitar scripts PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Luego ejecutar de nuevo
.\quick_setup.ps1
```

---

## ¿Qué va a pasar?

### 1️⃣ Testing Rápido (5-10 minutos)
```
✅ Carga modelos (Encoder, Decoder, Discriminator)
✅ Ejecuta 2 epochs con datos sintéticos
✅ Valida PSNR, SSIM, BER, losses
✅ Guarda checkpoints de prueba
✅ Genera logs

Resultado: Sabrás si el código tiene errores
```

### 2️⃣ Pregunta sobre Descarga
```
¿Descargar ImageNet ahora? [S/n]

Si dices "S":
  → Abre NUEVA VENTANA PowerShell
  → Descarga ImageNet (6.3 GB, 1-2 horas)
  → Prepara splits automáticamente (40K/5K/5K)
  → Puedes cerrar la ventana original
  → La descarga continúa en background
```

### 3️⃣ Resultado Final
```
✅ Código validado (funciona correctamente)
✅ Dataset descargado (50,000 imágenes)
✅ Splits preparados (data/imagenet2012/splits/)
✅ Todo listo para entrenar!

Próximo paso:
python train.py --config configs/base_config.yaml --dataset imagenet
```

---

## Opción 2: Solo Testing (Sin Descarga)

```powershell
# Solo validar código (10 minutos)
python train.py --config configs/test_config.yaml --dataset synthetic
```

**Resultado:** Sabes si hay errores, pero NO puedes replicar paper.

---

## Opción 3: Manual Completo

Ver [QUICKSTART.md](QUICKSTART.md) para pasos detallados.

---

## 🆘 Troubleshooting Rápido

### Error: "no se puede cargar quick_setup.ps1"
```powershell
# Solución: Habilitar scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Error: "ModuleNotFoundError: No module named 'yaml'"
```powershell
# Instalar dependencia faltante
& .venv\Scripts\pip.exe install pyyaml
```

### Error: "No module named 'kaggle'"
```powershell
# Instalar kaggle
& .venv\Scripts\pip.exe install kaggle
```

### Error: "kaggle.json not found"
```
1. Ir a https://www.kaggle.com/account
2. Scroll a "API" → "Create New API Token"
3. Guardar kaggle.json en: C:\Users\Lopez\.kaggle\
```

---

## 📊 Tiempos Estimados

| Paso | Tiempo | Acción |
|------|--------|--------|
| Testing | 5-10 min | ✅ Ejecutar quick_setup.ps1 |
| Descarga | 1-2 horas | ⏳ En background (nueva ventana) |
| Entrenamiento | 2-7 días | 🚀 python train.py ... |

---

## ✨ Ventajas del Setup Automático

✅ **Validación inmediata:** 10 minutos para saber si funciona  
✅ **Descarga en paralelo:** No bloquea tu trabajo  
✅ **Automático:** Prepara splits solo  
✅ **Seguro:** Si testing falla, no descarga (ahorra tiempo)  
✅ **Flexible:** Puedes pausar/reanudar  

---

## 🎯 Recomendación

**Para Revisión 1 (replicar paper):**

```powershell
# Una línea hace todo
.\quick_setup.ps1
```

**Responde "S" cuando pregunte por descarga**

En 10 minutos sabes si hay errores.  
En 2 horas tienes dataset listo.  
En 2-7 días tienes paper replicado.

---

**¿Listo?**

```powershell
.\quick_setup.ps1
```

🚀 ¡Presiona Enter y empieza!
