# Comandos Listos para Ejecutar

## 🚀 Paso 1: Commit y Push (Local)

```bash
cd "c:\Users\Lopez\OneDrive\Documents\MAESTRIA\Semestre 2\Esteganografia y marcas de agua\papers esteganografia DCT\DCT-GAN-Mobile"

git add -A

git commit -m "CRITICAL FIX: Change D:G update ratio from 4:1 to 5:1

Problem:
- Previous ratio was 4G:1D (4 generator updates per 1 discriminator update)
- Caused Loss_D ≈ 0.0000 (discriminator not learning)
- PSNR stagnated at ~17.95 dB (vs target 58.27 dB)

Solution:
- Changed to 5D:1G (5 discriminator updates per 1 generator update)
- This is the 'typical' ratio according to paper's Training Generator section
- Quote: 'Typically, the discriminator weights are updated five times,
  followed by a single update to the generator weights'

Expected Impact:
- Loss_D should be in range 0.01-0.10 (not near 0)
- PSNR should improve gradually (not stagnate)
- Should reach >30 dB by epoch 30-50

Files Modified:
- src/training/trainer.py: Updated training loop logic
- configs/paper_exact_config.yaml: Added update_strategy section
- configs/base_config.yaml: Updated default ratio
- configs/mobile_config.yaml: Updated mobile ratio

Documentation Created:
- FIXES_DISCRIMINATOR.md: Detailed bug analysis and solution
- DATASETS.md: BOSSBase and alternative datasets info
- RESUMEN_CAMBIOS.md: Quick summary of changes
- COMANDOS.md: Ready-to-run commands

References:
- Paper: Malik et al. 2025, DOI: 10.1038/s41598-025-01054-7
- WGAN: Arjovsky et al. 2017 (recommends 5 critic:1 generator)
- External AI search: Confirmed StegoGAN, RISRANet use similar ratios"

git push origin main
```

---

## 🎯 Paso 2: Entrenar en RunPod

### Opción A: Entrenar 30 Epochs (Validación, ~$1.00)

```bash
# En RunPod terminal
cd /workspace/DCT-GAN-Mobile

# Pull latest changes
git pull

# Activate environment (si usas conda/venv)
# conda activate dcgan  # o source .venv/bin/activate

# Train con early stopping
python train.py \
  --config configs/paper_exact_config.yaml \
  --max_epochs 30 \
  --patience 20 \
  --checkpoint_dir checkpoints_fix_5d1g \
  --log_dir logs_fix_5d1g

# Monitorear progreso
tail -f logs_fix_5d1g/training_*.log
```

### Opción B: Entrenar 100 Epochs (Full, ~$3.00)

```bash
cd /workspace/DCT-GAN-Mobile
git pull

python train.py \
  --config configs/paper_exact_config.yaml \
  --max_epochs 100 \
  --patience 50 \
  --checkpoint_dir checkpoints_fix_5d1g_full \
  --log_dir logs_fix_5d1g_full

# Background mode (para no perder si se desconecta)
nohup python train.py \
  --config configs/paper_exact_config.yaml \
  --max_epochs 100 \
  --patience 50 \
  --checkpoint_dir checkpoints_fix_5d1g_full \
  --log_dir logs_fix_5d1g_full \
  > training_output.log 2>&1 &

# Ver progreso
tail -f training_output.log
```

---

## 📊 Paso 3: Monitorear Métricas Clave

### Durante Entrenamiento

```bash
# Ver últimas 20 líneas del log
tail -n 20 logs_fix_5d1g/training_*.log

# Buscar métricas específicas
grep "Loss_D" logs_fix_5d1g/training_*.log | tail -n 10
grep "PSNR" logs_fix_5d1g/training_*.log | tail -n 10

# Ver evolución de Loss_D (debe estar 0.01-0.10)
grep "Epoch" logs_fix_5d1g/training_*.log | grep "Loss_D"
```

### Indicadores de Éxito

**✅ BUENAS SEÑALES**:
```
Epoch 1:  Loss_D 0.0856, PSNR 19.23 dB   ← D está aprendiendo
Epoch 5:  Loss_D 0.0432, PSNR 24.15 dB   ← Mejorando gradualmente
Epoch 10: Loss_D 0.0289, PSNR 28.67 dB   ← ¡Excelente progreso!
```

**❌ MALAS SEÑALES** (igual que antes):
```
Epoch 1:  Loss_D 0.0006, PSNR 17.91 dB
Epoch 2:  Loss_D 0.0000, PSNR 17.94 dB   ← D no aprende
Epoch 5:  Loss_D 0.0000, PSNR 17.95 dB   ← Estancado
```

---

## 🛑 Paso 4: Decisiones Según Resultados

### Después de 10 Epochs (~$0.30)

**CASO 1: Loss_D en 0.01-0.10, PSNR >20 dB**
```bash
# ✅ ¡FIX FUNCIONA! Continuar entrenamiento
# Dejar correr hasta epoch 50-100
```

**CASO 2: Loss_D ~0.0000, PSNR ~18 dB**
```bash
# ❌ Fix no funcionó → Hay otro bug
# ACCIÓN: Contactar autores del paper inmediatamente
# Ver email template en conversación anterior
```

**CASO 3: Loss_D en 0.01-0.10, pero PSNR <20 dB**
```bash
# ⚠️ D aprende pero no mejora PSNR
# ACCIÓN: Probar con dataset BOSSBase
# Ver DATASETS.md para instrucciones
```

### Después de 30 Epochs (~$1.00)

**CASO A: PSNR >30 dB**
```bash
# ✅ EXCELENTE - Continuar con dataset actual
python train.py --config configs/paper_exact_config.yaml --max_epochs 100
```

**CASO B: PSNR 25-30 dB**
```bash
# ⚠️ ACEPTABLE - Considerar entrenar más o probar BOSSBase
# Decisión: ¿Continuar o cambiar dataset?
```

**CASO C: PSNR <25 dB**
```bash
# ❌ INSUFICIENTE - Cambiar a BOSSBase
# Ver instrucciones en DATASETS.md
```

---

## 📥 Paso 5: Descargar BOSSBase (Si necesario)

### En RunPod Terminal

```bash
cd /workspace
mkdir -p datasets/bossbase
cd datasets/bossbase

# Método 1: wget
wget http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip
unzip BOSSbase_1.01.zip

# Método 2: curl (si wget falla)
curl -O http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip
unzip BOSSbase_1.01.zip

# Verificar descarga
ls -lh BOSSbase/  # Debe tener 10,000 archivos .pgm
```

### Crear Config para BOSSBase

```bash
cd /workspace/DCT-GAN-Mobile/configs
cp paper_exact_config.yaml bossbase_config.yaml

# Editar manualmente o con sed
sed -i 's/dataset_name: "ImageNet2012"/dataset_name: "BOSSBase"/g' bossbase_config.yaml
sed -i 's/image_size: 256/image_size: 512/g' bossbase_config.yaml
sed -i 's/base_path: .*/base_path: "\/workspace\/datasets\/bossbase\/BOSSbase\/"/g' bossbase_config.yaml
```

### Entrenar con BOSSBase

```bash
python train.py \
  --config configs/bossbase_config.yaml \
  --max_epochs 100 \
  --checkpoint_dir checkpoints_bossbase \
  --log_dir logs_bossbase
```

---

## 📤 Paso 6: Recuperar Resultados

### Después del Entrenamiento

```bash
# En RunPod terminal

# Ver mejor modelo
cat checkpoints_fix_5d1g/best_metrics.json

# Comprimir checkpoints para descarga
cd /workspace
tar -czf dcgan_checkpoints_fix5d1g.tar.gz \
  DCT-GAN-Mobile/checkpoints_fix_5d1g/ \
  DCT-GAN-Mobile/logs_fix_5d1g/

# Si RunPod tiene Jupyter, descargar desde:
# http://<runpod-ip>:8888/files/

# O usar scp desde local:
# scp root@<runpod-ip>:/workspace/dcgan_checkpoints_fix5d1g.tar.gz .
```

---

## 🗂️ Paso 7: Análisis de Resultados

### Generar Reporte

```bash
# En RunPod terminal
cd /workspace/DCT-GAN-Mobile

# Script para analizar logs (crear si no existe)
python << 'EOF'
import re

log_file = "logs_fix_5d1g/training_*.log"  # Ajustar nombre real

# Parse log y extraer métricas
epochs = []
psnr = []
ssim = []
loss_d = []

with open(log_file) as f:
    for line in f:
        if "Epoch" in line:
            # Extraer epoch, PSNR, SSIM, Loss_D
            # (Implementar según formato de tu log)
            pass

# Imprimir resumen
print(f"Best PSNR: {max(psnr):.2f} dB at epoch {psnr.index(max(psnr))}")
print(f"Best SSIM: {max(ssim):.4f} at epoch {ssim.index(max(ssim))}")
print(f"Avg Loss_D: {sum(loss_d)/len(loss_d):.4f}")
EOF
```

---

## 📧 Paso 8: Contactar Autores (Si TODO falla)

### Email Template Listo

```
To: ateeq@ksu.edu.sa, seada.hussen@aastu.edu.et
CC: ahalmogren@ksu.edu.sa
Subject: Implementation Details Request - DCT-GAN Steganography (SR 2025)

Dear Dr. Rehman and Dr. Hussen,

I am a master's student attempting to replicate your excellent work:
"A hybrid steganography framework using DCT and GAN for secure data 
communication in the big data era" (Scientific Reports, 2025).

Implementation Status:
- GitHub repo: github.com/jaimelopezm-star/DCT-GAN-Mobile
- Model params: ~115K (target: 50K)
- Best PSNR: 17.95 dB (target: 58.27 dB)
- Loss_D: ~0.0000 (discriminator not learning)

Recent fix applied:
- Changed D:G update ratio from 4:1 (G:D) to 5:1 (D:G) per your 
  "Training Generator" section quote about "typical" ratio

Questions:
1. Dataset: Did you use BOSSBase, BOWS2, or ImageNet2012? What resolution?
2. Update ratio: Paper mentions both 4G:1D (implemented) and 5D:1G (typical). 
   Which did you actually use?
3. Hyperparameters: Any undocumented details (warmup, gradient clipping, etc.)?
4. Code availability: Any chance to access your implementation or checkpoints?

I would greatly appreciate any clarifications to help replicate your results.

Best regards,
[Tu Nombre]
Master's Student, [Tu Universidad]
GitHub: github.com/jaimelopezm-star
```

---

## 📝 Checklist Completo

### Pre-Entrenamiento
- [x] Fix 5D:1G implementado
- [x] Configs actualizados
- [ ] Git commit local
- [ ] Git push a GitHub
- [ ] RunPod instance activo
- [ ] Pull latest code en RunPod

### Durante Entrenamiento (Check cada 10 epochs)
- [ ] Loss_D en rango 0.01-0.10?
- [ ] PSNR mejorando gradualmente?
- [ ] SSIM mejorando?
- [ ] Logs guardándose correctamente?

### Post-Entrenamiento (30 epochs)
- [ ] PSNR >30 dB? → Continuar
- [ ] PSNR <30 dB? → Probar BOSSBase
- [ ] Loss_D ~0? → Contactar autores

### Post-Entrenamiento (100 epochs)
- [ ] PSNR >50 dB? → ¡Éxito!
- [ ] PSNR 40-50 dB? → Aceptable
- [ ] PSNR <40 dB? → Investigar más

---

## 🎯 Objetivo Final

**Meta**: PSNR ≥50 dB (acercarse a 58.27 dB del paper)

**Plan A** (Fix 5D:1G con ImageNet sintético): $1-3  
**Plan B** (BOSSBase dataset): $2-4  
**Plan C** (Contactar autores): $0 (solo tiempo)  
**Plan D** (Pivotar a alternativa): $3-5 (RISRANet/StegoGAN)

**Total máximo invertido hasta ahora**: $0.50  
**Budget disponible**: ~$10-20 para validación completa

---

**Siguiente comando INMEDIATO**:
```bash
cd "c:\Users\Lopez\OneDrive\Documents\MAESTRIA\Semestre 2\Esteganografia y marcas de agua\papers esteganografia DCT\DCT-GAN-Mobile"
git add -A
git commit -m "CRITICAL FIX: Change D:G ratio 4:1→5:1 to fix discriminator"
git push
```

¡DESPUÉS DE PUSH, ENTRENAR EN RUNPOD INMEDIATAMENTE!
