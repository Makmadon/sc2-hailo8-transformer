@echo off
title SC2 Hailo-8 V2 Terran Bot (Raspberry Pi 192.168.50.7:5555)
cd /d "%~dp0"
echo ============================================================================
echo   Iniciando StarCraft II Terran Bot V2 conectado a Raspberry Pi 5 (Hailo-8)
echo ============================================================================
py -3.12 -m src.pc_sc2_client --play --rpi-ip 192.168.50.7 --port 5555 --map AbyssalReefLE --enemy-race Terran --difficulty Medium --realtime
pause
