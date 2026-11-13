# SC2-Hailo8-Transformer (V2)

Agente **Terran** de Inteligencia Artificial para **StarCraft II** ejecutando inferencia neuronal en tiempo real sobre una **NPU Hailo-8 (Raspberry Pi 5)** conectada por red local TCP binaria a un PC Windows (StarCraft II) y entrenada en **WSL / CPU** (`Intel Core i5-11400`) o GPU.

> La rama `main` conserva la versión V1 original. Todo el rediseño arquitectónico y de RL descrito aquí vive en la rama **`V2`**.

---

## Arquitectura V2: Dual-Branch Spatial-Global + Lazo Cerrado (`K=1`)

V2 reemplaza el cuello de botella horizontal `16x4` (Action-Chunking) de V1 por una política **Actor-Critic** compilable en contexto único INT8 en Hailo-8 y entrenable con **Behavior Cloning + Clipped PPO real**:

1. **Entradas 4D NCHW estáticas (sin rayas escalares dentro del mapa):**
   - `map_tensor [1, 16, 32, 32]`: grid táctico `32x32` del área jugable:
     - `0..3`: trabajadores propios, infantería Bio, unidades Mech, unidades Aéreas.
     - `4`: promedio real de HP aliado por celda.
     - `5..7`: Town Halls propios (CC/Orbital/Planetary), edificios de producción/tech, depósitos/refinerías/defensas.
     - `8..12`: trabajadores enemigos, ejército terrestre enemigo, unidades aéreas enemigas, estructuras enemigas, promedio HP enemigo.
     - `13`: minerales y géiseres de gas vespeno.
     - `14`: terreno transitable + altura normalizada.
     - `15`: visibilidad / niebla de guerra.
   - `global_vec [1, 64, 1, 1]`: vector económico, `supply_used` / `supply_cap` / `supply_left`, conteo de unidades, edificios, add-ons, mejoras Tech/Bio/Mech, cooldowns e historial EMA.

2. **Backbone `SC2HailoV2Actor` (4D puro amigable con Hailo DFC v3.34.0):**
   - Stem convolucional `32x32` (skip connection U-Net) + downsampling a `16x16` con positional embedding 2D aprendible.
   - Encoder global -> `4` tokens (`[B, d_model, 1, 4]`) con **Cross-Attention bidireccional** (`Spatial <-> Global`) implementada con convoluciones `1x1`, `BatchNorm2d`, `ReLU` y `MatMul` sobre dimensiones estáticas `[H, W]`.
   - Decoder espacial con `ConvTranspose2d` `16x16 -> 32x32` fusionado con el skip `32x32`.

3. **Salidas simultáneas en cada tick (`K=1`, sin Chunking):**
   - `macro_logits [1, 28, 1, 1]`: 28 comandos macro Terran (producción SCV/ejército, depósitos, expansiones, add-ons, MULE, upgrades).
   - `micro_logits [1, 12, 1, 1]`: 12 comandos tácticos simultáneos (scout, scanner sweep, ataque/move/kite, stimpack, siege/unsiege, reparación, bajar depósitos).
   - `spatial_logits [1, 2, 32, 32]`: heatmaps espaciales `32x32` independientes para el objetivo macro (canal 0) y micro (canal 1).

4. **Protocolo binario TCP multihilo (`PC <-> RPi5`):**
   - Petición PC -> RPi5: **65,796 bytes** (`magic` + `map_tensor` + `global_vec`).
   - Respuesta RPi5 -> PC: **8,364 bytes** (`magic` + `backend_code` + `npu_ms` + logits brutos `macro`, `micro`, `spatial`).
   - El servidor en Raspberry Pi (`src/rpi_hailo_server.py`) atiende múltiples workers en paralelo con `threading.Thread` y `infer_lock`, devolviendo **logits brutos** para que el PC aplique **Action Masking estricto** y muestreo estocástico con `log_probs` exactos.

---

## Estructura de Módulos (`src/`)

| Archivo | Función |
| :--- | :--- |
| [config.py](src/config.py) | Constantes `32x32`, vocabulario dual (`28` Macro + `12` Micro), protocolo binario y `V2Config`. |
| [features.py](src/features.py) | Extractor unificado de estado (`16x32x32` + `64`) y **Action Masking** Terran compartido entre vivo y replays. |
| [actions.py](src/actions.py) | Ejecutor `TerranActionExecutorV2` para las 28 acciones macro + 12 micro en `python-sc2`. |
| [model.py](src/model.py) | `SC2HailoV2Actor`, `SC2HailoV2Critic`, utilidades PPO/KL enmascaradas y exportación ONNX Opset 16. |
| [replay_dataset.py](src/replay_dataset.py) | Extractor `.SC2Replay` (`ObserverAI`), respaldo offline `SC2-Dynamics-50K` y `SC2V2BCDataset`. |
| [train_bc.py](src/train_bc.py) | Entrenamiento BC (genera actor, ancla KL fija, calibración INT8 y `.onnx`). |
| [train_rl.py](src/train_rl.py) | Crítico `V_phi(s)` + **GAE(λ)** + **Clipped PPO** con ancla KL fija al BC inicial y entropía. |
| [export_onnx.py](src/export_onnx.py) | Exportador CLI de checkpoints `.pth` a `models/sc2_v2_actor.onnx`. |
| [compile_hef.py](src/compile_hef.py) | Compilador Hailo DFC v3.34.0 (ONNX -> HAR cuantizado INT8 -> `models/sc2_v2_agent.hef`). |
| [rpi_hailo_server.py](src/rpi_hailo_server.py) | Servidor TCP binario multihilo (Hailo-8 NPU o fallback ONNX Runtime). |
| [pc_sc2_client.py](src/pc_sc2_client.py) | Cliente SC2 en Windows (`--play`, `--collect-teacher`, `--collect-rl`, `--benchmark`, multi-worker). |
| [verify_v2.py](src/verify_v2.py) | Suite local de verificación sin SC2 ni NPU (paridad ONNX, BC, PPO/GAE, roundtrip TCP). |

---

## Flujo Recomendado (Windows + WSL + Raspberry Pi 5)

El diseño actual permite dividir limpiamente el flujo sin depender de GPU ni notebooks externos:

1. **Verificación rápida (Windows o WSL):**
   ```bash
   python -m src.verify_v2
   ```

2. **Generación de dataset inicial BC Terran (elige una opción):**
   - **Opción A — Windows (Recomendada, cero domain-shift):** ejecuta demostraciones del teacher Terran integrado o extrae `.SC2Replay`:
     ```bat
     recolectar_teacher_bc.bat
     ```
     O con carpeta de replays:
     ```bat
     py -3.12 -m src.replay_dataset --extract-replays "C:\ruta\a\replays" --out data/sc2_v2_bc_dataset.npz
     ```
   - **Opción B — WSL (sin motor SC2):** conversión Terran de respaldo desde `SC2-Dynamics-50K`:
     ```bash
     python3 -m src.replay_dataset --from-hf --max-samples 15000 --out data/sc2_v2_bc_dataset.npz
     ```

3. **Preentrenamiento Behavior Cloning + Exportación ONNX (WSL CPU):**
   ```bash
   python3 -m src.train_bc --dataset data/sc2_v2_bc_dataset.npz --epochs 20 --batch-size 64 --device cpu --threads 6
   ```
   Esto genera:
   - `models/sc2_v2_actor.pth`
   - `models/sc2_v2_bc_anchor.pth` (referencia congelada para el KL de PPO)
   - `models/sc2_v2_calib.npz`
   - `models/sc2_v2_actor.onnx`

4. **Compilación `.hef` en WSL (Hailo Dataflow Compiler v3.34.0) y despliegue en RPi5:**
   ```bash
   ./compilar_hef_wsl.sh
   ./subir_a_raspberry_wsl.sh
   ```
   En la Raspberry Pi 5:
   ```bash
   cd ~/sc2-hailo8-transformer && python3 -m src.rpi_hailo_server --port 5555
   ```

5. **Ciclo Iterativo de Aprendizaje por Refuerzo (Clipped PPO + GAE):**
   - En **Windows** (recolectar 60 partidas con curriculum automático usando la RPi5):
     ```bat
     entrenar_rl_hailo8_paralelo.bat
     ```
   - En **WSL** (actualizar Actor/Critic, calibración y re-exportar ONNX):
     ```bash
     python3 -m src.train_rl --rollouts models/sc2_v2_rollouts.npz --epochs 6 --critic-epochs 8 --device cpu --threads 6
     ./compilar_hef_wsl.sh
     ./subir_a_raspberry_wsl.sh
     ```

6. **Jugar partida en vivo (Windows):**
   ```bat
   jugar_starcraft_hailo8.bat
   ```
