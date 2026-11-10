@echo off
title SC2 Hailo-8 RL Experience Collector - 30 Partidas Aceleradas (Google Drive)
cd /d "C:\Users\Stalin\.gemini\antigravity\scratch\sc2-hailo8-transformer"
taskkill /F /IM SC2_x64.exe /T >nul 2>&1
echo ============================================================================
echo   Recoleccion Acelerada de Aprendizaje por Refuerzo (RL) - Hailo-8 NPU
echo   Guardando experiencia automaticamente en: G:\Mi unidad\hailo_files\sc2_rl_rollouts.npz
echo ============================================================================
py -3.12 -u -m src.pc_sc2_client --collect-rl --episodes 30 --difficulty Medium --explore-eps 0.16 --rpi-ip 192.168.50.7 --port 5555
pause
