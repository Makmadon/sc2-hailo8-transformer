@echo off
title SC2 Hailo-8 Transformer Bot (Raspberry Pi 192.168.50.7:5555)
cd /d "C:\Users\Stalin\.gemini\antigravity\scratch\sc2-hailo8-transformer"
echo ============================================================================
echo   Iniciando StarCraft II en Mapa de Torneo (AbyssalReefLE) - Hailo-8 NPU
echo ============================================================================
py -3.12 -m src.pc_sc2_client --play --rpi-ip 192.168.50.7 --port 5555 --map AbyssalReefLE --race Terran --enemy-race Terran --difficulty Medium --realtime
pause
