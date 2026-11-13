@echo off
title SC2 Hailo-8 V2 Teacher BC Collector (2 Workers)
cd /d "%~dp0"
echo ============================================================================
echo   Recoleccion de Demostraciones Teacher Terran para Preentrenamiento BC V2
echo   Guardando dataset en: data\sc2_v2_bc_dataset.npz
echo ============================================================================
py -3.12 -u -m src.pc_sc2_client --collect-teacher --episodes 20 --workers 2 --difficulty Medium --bc-dataset data/sc2_v2_bc_dataset.npz
pause
