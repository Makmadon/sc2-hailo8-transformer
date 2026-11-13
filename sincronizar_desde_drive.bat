@echo off
title Sincronizar Modelos de Google Drive a Local
cd /d "%~dp0"

set "DRIVE_DIR=G:\Mi unidad\hailo_files"
set "LOCAL_MODELS=models"
set "NOTEBOOK_DRIVE=G:\Mi unidad\Colab Notebooks\SC2_Hailo8_Transformer.ipynb"
set "NOTEBOOK_LOCAL=notebooks\SC2_Hailo8_Transformer.ipynb"

echo ============================================================================
echo   Sincronizando Modelos y Binarios (.hef / .pth / .onnx) desde Google Drive
echo ============================================================================
echo.

if not exist "%DRIVE_DIR%" (
    echo [ERROR] No se puede acceder a: "%DRIVE_DIR%"
    echo Verifica que la aplicacion Google Drive este iniciada en tu PC.
    echo.
    pause
    exit /b 1
)

echo [1/3] Sincronizando binarios de NPU y pesos (.hef, .pth, .onnx, .har)...
robocopy "%DRIVE_DIR%" "%LOCAL_MODELS%" sc2_hailo8_agent.hef sc2_hailo8_transformer_best.pth sc2_hailo8_transformer.onnx sc2_value_critic.pth sc2_hailo8_quantized.har sc2_calib_data.npy /XO /R:2 /W:1 /NJH /NJS /NDL /NC /NS

if %ERRORLEVEL% GEQ 8 (
    echo [AVISO] Ocurrio un error al transferir algunos archivos.
) else (
    echo [OK] Modelos y binarios sincronizados correctamente.
)
echo.

echo [2/3] Verificando buffer de experiencia RL (sc2_rl_rollouts.npz)...
robocopy "%DRIVE_DIR%" "%LOCAL_MODELS%" sc2_rl_rollouts.npz /XO /R:2 /W:1 /NJH /NJS /NDL /NC /NS >nul 2>&1
echo [OK] Buffer de partidas actualizado.
echo.

echo [3/3] Sincronizando cuaderno de Google Colab...
if exist "%NOTEBOOK_DRIVE%" (
    copy /Y "%NOTEBOOK_DRIVE%" "%NOTEBOOK_LOCAL%" >nul 2>&1
    echo [OK] Cuaderno actualizado en: %NOTEBOOK_LOCAL%
)
if not exist "%NOTEBOOK_DRIVE%" (
    echo [INFO] Cuaderno en Drive no encontrado en ruta predeterminada.
)

echo.
echo ============================================================================
echo   Sincronizacion completada con exito.
echo.
echo   Archivos locales listos en:
echo     models\sc2_hailo8_agent.hef
echo     models\sc2_hailo8_transformer_best.pth
echo     models\sc2_hailo8_transformer.onnx
echo     models\sc2_rl_rollouts.npz
echo ============================================================================
echo.
pause
