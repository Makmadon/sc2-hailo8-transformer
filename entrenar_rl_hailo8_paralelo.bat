@echo off
title SC2 Hailo-8 RL Parallel Collector (2 Workers) - Google Drive
cd /d "C:\Users\Stalin\.gemini\antigravity\scratch\sc2-hailo8-transformer"
echo ============================================================================
echo   Recoleccion RL Paralela Multiproceso (2 Workers) - Hailo-8 NPU
echo   Guardando experiencia automaticamente en: G:\Mi unidad\hailo_files\sc2_rl_rollouts.npz
echo ============================================================================
py -3.12 -u -m src.pc_sc2_client --collect-rl --episodes 60 --workers 2 --difficulty Easy --explore-eps 0.16 --rpi-ip 192.168.50.7 --port 5555
pause
