#!/bin/bash
# Script para descargar DIV2K dataset en RunPod
# DIV2K: Dataset de 1000 imágenes 2K de alta calidad
# Uso: bash download_div2k.sh

set -e

echo "=============================================="
echo "DESCARGA DE DIV2K DATASET"
echo "=============================================="
echo ""

# Crear directorio de trabajo
WORK_DIR="/workspace/DIV2K_download"
mkdir -p $WORK_DIR
cd $WORK_DIR

echo "📁 Directorio de trabajo: $WORK_DIR"
echo ""

# URLs de DIV2K
TRAIN_URL="http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
VAL_URL="http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"

# ============================================
# 1. Descargar Training Set
# ============================================
echo "📥 Descargando DIV2K Training Set (800 imágenes)..."
echo "   URL: $TRAIN_URL"
echo "   Tamaño: ~3.5 GB"
echo ""

if [ ! -f "DIV2K_train_HR.zip" ]; then
    wget --progress=bar:force:noscroll $TRAIN_URL
    echo "✅ Training set descargado"
else
    echo "⏭️  Training set ya existe, omitiendo descarga"
fi
echo ""

# ============================================
# 2. Descargar Validation Set
# ============================================
echo "📥 Descargando DIV2K Validation Set (100 imágenes)..."
echo "   URL: $VAL_URL"
echo "   Tamaño: ~450 MB"
echo ""

if [ ! -f "DIV2K_valid_HR.zip" ]; then
    wget --progress=bar:force:noscroll $VAL_URL
    echo "✅ Validation set descargado"
else
    echo "⏭️  Validation set ya existe, omitiendo descarga"
fi
echo ""

# ============================================
# 3. Descomprimir
# ============================================
echo "📦 Descomprimiendo archivos..."
echo ""

if [ ! -d "DIV2K_train_HR" ]; then
    echo "   Extrayendo training set..."
    unzip -q DIV2K_train_HR.zip
    echo "   ✅ Training set extraído"
else
    echo "   ⏭️  Training set ya extraído"
fi

if [ ! -d "DIV2K_valid_HR" ]; then
    echo "   Extrayendo validation set..."
    unzip -q DIV2K_valid_HR.zip
    echo "   ✅ Validation set extraído"
else
    echo "   ⏭️  Validation set ya extraído"
fi
echo ""

# ============================================
# 4. Verificar descarga
# ============================================
echo "🔍 Verificando archivos..."
echo ""

TRAIN_COUNT=$(ls DIV2K_train_HR/*.png 2>/dev/null | wc -l)
VAL_COUNT=$(ls DIV2K_valid_HR/*.png 2>/dev/null | wc -l)

echo "   Training images: $TRAIN_COUNT (esperado: 800)"
echo "   Validation images: $VAL_COUNT (esperado: 100)"
echo ""

if [ $TRAIN_COUNT -eq 800 ] && [ $VAL_COUNT -eq 100 ]; then
    echo "✅ Descarga completa y verificada!"
else
    echo "⚠️  Advertencia: Número de imágenes no coincide"
    echo "   Puede que la descarga esté incompleta"
fi

echo ""
echo "=============================================="
echo "DESCARGA COMPLETADA"
echo "=============================================="
echo ""
echo "📁 Imágenes descargadas en:"
echo "   - Train: $WORK_DIR/DIV2K_train_HR"
echo "   - Val: $WORK_DIR/DIV2K_valid_HR"
echo ""
echo "📝 Siguiente paso:"
echo "   python prepare_div2k.py"
echo ""
