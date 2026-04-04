# Quick Setup para Windows - Testing + Descarga en Paralelo
# Uso: .\quick_setup.ps1

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   DCT-GAN QUICK SETUP - Testing + Descarga" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Activar entorno virtual
Write-Host "[1/4] Activando entorno virtual..." -ForegroundColor Yellow
& .venv\Scripts\Activate.ps1

# Crear directorios
Write-Host "[2/4] Creando directorios..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "logs", "logs_test", "checkpoints_test" | Out-Null

# OPCION 1: Testing rapido PRIMERO
Write-Host ""
Write-Host "[3/4] Ejecutando testing rapido (2 epochs, datos sinteticos)..." -ForegroundColor Yellow
Write-Host "      Tiempo estimado: 5-10 minutos" -ForegroundColor Gray
Write-Host "      Objetivo: Validar que el codigo funciona" -ForegroundColor Gray
Write-Host ""

# Ejecutar test
& python train.py --config configs/test_config.yaml --dataset synthetic --checkpoint_dir checkpoints_test --log_dir logs_test

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "TESTING COMPLETADO EXITOSAMENTE!" -ForegroundColor Green
    Write-Host "El codigo funciona correctamente" -ForegroundColor Green
    Write-Host ""
    
    # OPCION 2: Preguntar si quiere descargar ImageNet
    Write-Host "[4/4] Descargar ImageNet ahora? (6.3 GB, 1-2 horas)" -ForegroundColor Yellow
    $respuesta = Read-Host "      [S/n]"
    
    if ($respuesta -eq "" -or $respuesta -eq "S" -or $respuesta -eq "s") {
        Write-Host ""
        Write-Host "Iniciando descarga de ImageNet en NUEVA VENTANA..." -ForegroundColor Cyan
        Write-Host "Puedes cerrar esta ventana y la descarga continuara" -ForegroundColor Gray
        
        # Abrir nueva ventana PowerShell para descarga
        $downloadCmd = "cd '$PWD'; & .venv\Scripts\Activate.ps1; python scripts/download_imagenet.py --method kaggle; python scripts/prepare_dataset.py; Write-Host 'DESCARGA Y SPLITS COMPLETADOS!' -ForegroundColor Green; Write-Host 'Ahora ejecuta:' -ForegroundColor Yellow; Write-Host 'python train.py --config configs/base_config.yaml --dataset imagenet' -ForegroundColor Cyan; Read-Host 'Presiona Enter para cerrar'"
        Start-Process powershell -ArgumentList "-NoExit", "-Command", $downloadCmd
        
        Write-Host ""
        Write-Host "Descarga iniciada en nueva ventana!" -ForegroundColor Green
        Write-Host "Puedes continuar trabajando en esta ventana" -ForegroundColor Gray
        Write-Host "La descarga se ejecuta independientemente" -ForegroundColor Gray
        Write-Host ""
        
    } else {
        Write-Host ""
        Write-Host "Descarga saltada" -ForegroundColor Yellow
        Write-Host "Para descargar despues:" -ForegroundColor Gray
        Write-Host "python scripts/download_imagenet.py --method kaggle" -ForegroundColor Cyan
    }
    
    Write-Host ""
    Write-Host "=============================================" -ForegroundColor Green
    Write-Host "   QUICK SETUP COMPLETADO!" -ForegroundColor Green
    Write-Host "=============================================" -ForegroundColor Green
    Write-Host ""
    
} else {
    Write-Host ""
    Write-Host "TESTING FALLO" -ForegroundColor Red
    Write-Host "Revisa los errores arriba" -ForegroundColor Red
    Write-Host "Los logs estan en: logs_test\" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Presiona Enter para continuar..."
Read-Host
