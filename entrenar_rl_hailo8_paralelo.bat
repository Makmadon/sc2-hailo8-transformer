@echo off
title SC2 Hailo-8 V2 RL Parallel Collector (2 Workers)
cd /d "%~dp0"
echo ============================================================================
echo   Recoleccion RL PPO V2 en Paralelo (2 Partidas simultaneas) - Hailo-8 NPU
echo   Guardando rollouts en: models\sc2_v2_rollouts.npz
echo ============================================================================
py -3.12 -u -m src.pc_sc2_client --collect-rl --episodes 60 --workers 2 --difficulty Easy --temperature 1.0 --rpi-ip 192.168.50.7 --port 5555
pause
