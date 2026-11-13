@echo off
title Subir modelo V2 Hailo-8 a Raspberry Pi 5
cd /d "%~dp0"
echo ============================================================================
echo   Transfiriendo binario sc2_v2_agent.hef y servidor V2 a Raspberry Pi 5
echo   Destino: rp5@192.168.50.7:~/sc2-hailo8-transformer/
echo ============================================================================
echo Ingresa la contrasena de rp5 si se te solicita:
echo.
ssh rp5@192.168.50.7 "mkdir -p ~/sc2-hailo8-transformer/models ~/sc2-hailo8-transformer/src"
if exist "models\sc2_v2_agent.hef" (
    scp "models\sc2_v2_agent.hef" rp5@192.168.50.7:~/sc2-hailo8-transformer/models/sc2_v2_agent.hef
) else (
    echo [Aviso] models\sc2_v2_agent.hef no encontrado aun; omitiendo .hef.
)
if exist "models\sc2_v2_actor.onnx" (
    scp "models\sc2_v2_actor.onnx" rp5@192.168.50.7:~/sc2-hailo8-transformer/models/sc2_v2_actor.onnx
)
scp "src\__init__.py" rp5@192.168.50.7:~/sc2-hailo8-transformer/src/__init__.py
scp "src\config.py" rp5@192.168.50.7:~/sc2-hailo8-transformer/src/config.py
scp "src\rpi_hailo_server.py" rp5@192.168.50.7:~/sc2-hailo8-transformer/src/rpi_hailo_server.py
echo.
echo ============================================================================
echo   Transferencia V2 finalizada.
echo   Inicia en RPi5 con: python3 -m src.rpi_hailo_server --port 5555
echo ============================================================================
pause
