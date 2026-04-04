# 🚀 Instrucciones para Subir a GitHub

## ✅ Estado Actual
- ✅ Repositorio Git inicializado
- ✅ Commit inicial creado (49 archivos, 14,130+ líneas)
- ✅ .gitignore configurado (no subirá datos ni checkpoints)
- ✅ Usuario Git configurado

**Commit hash:** `3f38008`  
**Archivos incluidos:** 49  
**Total líneas:** 14,130+

---

## 📋 Pasos para Subir a GitHub

### 1️⃣ Crear Repositorio en GitHub

1. Ve a [https://github.com/new](https://github.com/new)
2. Configura el repositorio:
   ```
   Repository name: DCT-GAN-Mobile
   Description: Hybrid DCT-GAN framework for mobile steganography - Implementation of Malik et al. (2025)
   Visibility: ⚪ Public (recomendado) o 🔒 Private
   
   ❌ NO marques:
   - Add a README file
   - Add .gitignore
   - Choose a license
   
   (Ya tenemos estos archivos localmente)
   ```
3. Click **"Create repository"**

---

### 2️⃣ Conectar Repositorio Local con GitHub

GitHub te mostrará instrucciones. **Usa la opción "push an existing repository"**:

```powershell
# Desde PowerShell en la carpeta DCT-GAN-Mobile

# Agregar el remote (reemplaza TU-USUARIO con tu nombre de usuario GitHub)
git remote add origin https://github.com/TU-USUARIO/DCT-GAN-Mobile.git

# Verificar que se agregó correctamente
git remote -v

# Renombrar la rama a 'main' (si es necesario)
git branch -M main

# Subir el código
git push -u origin main
```

**⚠️ IMPORTANTE:** Reemplaza `TU-USUARIO` con tu nombre de usuario real de GitHub.

Si usas autenticación por token (recomendado):
- GitHub pedirá usuario y **Personal Access Token** (no la contraseña)
- [Crear token aquí](https://github.com/settings/tokens) con permisos `repo`

---

### 3️⃣ Verificar que se Subió Correctamente

1. Ve a `https://github.com/TU-USUARIO/DCT-GAN-Mobile`
2. Deberías ver:
   - ✅ 49 archivos
   - ✅ README.md renderizado
   - ✅ Commit: "Initial commit: DCT-GAN Mobile implementation"
   - ✅ Carpetas: `src/`, `configs/`, `scripts/`, etc.
   - ❌ NO debería haber: `data/`, `.venv/`, `checkpoints/`, `logs/`

---

## 🎯 Siguientes Pasos Después de Subir

### Opción A: Entrenar en Google Colab (RECOMENDADO)

1. Ve a [Google Colab](https://colab.research.google.com/)
2. Sube el notebook `train_gpu_cloud.ipynb`
3. O crea uno nuevo y clona el repo:
   ```python
   !git clone https://github.com/TU-USUARIO/DCT-GAN-Mobile.git
   %cd DCT-GAN-Mobile
   !pip install -r requirements_gpu.txt
   ```
4. Cambia runtime a **GPU (T4)**: Runtime → Change runtime type → GPU
5. Ejecuta el entrenamiento

**Ventajas:**
- ✅ GPU T4 gratis
- ✅ 1-2 días vs 19 días en CPU
- ✅ 12 GB RAM
- ⚠️ Límite: ~12 horas continuas

---

### Opción B: Entrenar en Kaggle

1. Ve a [Kaggle Notebooks](https://www.kaggle.com/code)
2. New Notebook → Import from GitHub:
   ```
   https://github.com/TU-USUARIO/DCT-GAN-Mobile
   ```
3. Settings → Accelerator → **GPU P100**
4. Run All

**Ventajas:**
- ✅ GPU P100 (más rápida que T4)
- ✅ 30 horas semanales gratis
- ✅ 16 GB RAM

---

### Opción C: Continuar en Local (CPU)

Solo si tienes paciencia:
```powershell
cd 'C:\Users\Lopez\OneDrive\Documents\MAESTRIA\Semestre 2\Esteganografia y marcas de agua\papers esteganografia DCT\DCT-GAN-Mobile'
.\.venv\Scripts\Activate.ps1
python train.py --epochs 100
```

**Tiempo estimado:** 19 días 😴

---

## 📊 Estructura Subida

```
DCT-GAN-Mobile/
├── src/                    ✅ Código fuente
│   ├── models/            ✅ Arquitecturas
│   ├── training/          ✅ Trainer y losses
│   └── dct/               ✅ Módulo DCT
├── configs/                ✅ YAMLs de configuración
├── scripts/                ✅ Scripts de datasets
├── utils/                  ✅ Utilidades
├── train.py                ✅ Script principal
├── train_test.py           ✅ Test de 2 épocas
├── train_gpu_cloud.ipynb   ✅ Notebook para Colab
├── requirements.txt        ✅ Dependencias
├── README.md               ✅ Documentación
└── .gitignore              ✅ Archivos ignorados

NO SUBIDOS (por .gitignore):
├── .venv/                  ❌ Entorno virtual
├── data/                   ❌ Datasets (10K imágenes)
├── checkpoints/            ❌ Modelos guardados
├── logs/                   ❌ Logs de entrenamiento
└── __pycache__/            ❌ Python cache
```

---

## 🔍 Comandos Útiles Post-GitHub

### Ver historial de commits
```powershell
git log --oneline
```

### Hacer cambios y subir
```powershell
# 1. Hacer cambios en archivos
# 2. Agregar cambios
git add .

# 3. Commit
git commit -m "Descripción de cambios"

# 4. Subir
git push
```

### Clonar en otra máquina (ej. Colab)
```bash
git clone https://github.com/TU-USUARIO/DCT-GAN-Mobile.git
```

### Actualizar desde GitHub (si editas en Colab)
```powershell
git pull origin main
```

---

## 📝 Checklist Final

Antes de entrenar en GPU, verifica:

- [ ] Repositorio subido a GitHub
- [ ] README visible en la página del repo
- [ ] No se subieron archivos grandes (data/, .venv/)
- [ ] Colab/Kaggle puede clonar el repo
- [ ] requirements_gpu.txt instalado
- [ ] GPU activada en Colab (T4) o Kaggle (P100)
- [ ] train_gpu_cloud.ipynb ejecutándose

---

## 🎓 Información del Proyecto

**Autor:** SebasPK01JS  
**Universidad:** MAESTRIA - Semestre 2  
**Materia:** Esteganografía y Marcas de Agua  
**Paper Base:** Malik et al. (2025) - DCT-GAN Steganography  
**Objetivo:** Replicar métricas del paper (PSNR ~58 dB, SSIM ~0.94)

**Estado Actual:**
- ✅ Código: 100% completo
- ✅ Arquitectura: 45,998 parámetros
- ✅ Dataset: 8,000 train + 1,000 val (Tiny ImageNet)
- ⏳ Entrenamiento: Pendiente (100 épocas en GPU)

---

## ❓ Problemas Comunes

### Error: "remote origin already exists"
```powershell
git remote remove origin
git remote add origin https://github.com/TU-USUARIO/DCT-GAN-Mobile.git
```

### Error: "Authentication failed"
- Usa **Personal Access Token**, no tu contraseña
- Crear en: https://github.com/settings/tokens
- Scope necesario: `repo`

### Error: "Updates were rejected"
```powershell
git pull origin main --allow-unrelated-histories
git push -u origin main
```

---

## 🚀 ¡Listo para Entrenar!

Una vez subido, **recomienda usar Google Colab T4**:

1. Tiempo: 1-2 días (vs 19 días en CPU)
2. Gratis
3. Ya tienes el notebook listo (`train_gpu_cloud.ipynb`)

**Siguiente comando después de subir:**
```bash
# En Colab, primera celda:
!git clone https://github.com/TU-USUARIO/DCT-GAN-Mobile.git
%cd DCT-GAN-Mobile
!pip install -r requirements_gpu.txt
```

¡Éxito! 🎉
