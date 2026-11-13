@echo off
title Subir nuevo modelo Hailo-8 (Ronda 5) a Raspberry Pi 5
cd /d "C:\Users\Stalin\.gemini\antigravity\scratch\sc2-hailo8-transformer"
echo ============================================================================
echo   Transfiriendo binario sc2_hailo8_agent.hef y servidor a Raspberry Pi
echo   Destino: rp5@192.168.50.7:~/sc2-hailo8-transformer/
echo ============================================================================
echo Ingresa la contrasena de rp5 si se te solicita:
echo.
scp "models\sc2_hailo8_agent.hef" rp5@192.168.50.7:~/sc2-hailo8-transformer/models/sc2_hailo8_agent.hef
scp "src\rpi_hailo_server.py" rp5@192.168.50.7:~/sc2-hailo8-transformer/src/rpi_hailo_server.py
echo.
echo ============================================================================
echo   Transferencia finalizada.
echo ============================================================================
pause
