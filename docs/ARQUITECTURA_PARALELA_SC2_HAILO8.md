# ⚡ Arquitectura de Recolección y Entrenamiento en Paralelo (Hailo-8 + SC2)

Este documento detalla la implementación y optimización para ejecutar **múltiples partidas simultáneas de StarCraft II en paralelo** utilizando la NPU **Hailo-8 (26 TOPS)** en la Raspberry Pi 5 y entrenamiento acelerado en **Google Colab**.

---

## 1. El Reto: StarCraft II y el Cuello de Botella Mononúcleo

StarCraft II (`SC2_x64.exe`) está programado en C++ con un bucle de simulación síncrono estrictamente **mononúcleo (single-threaded lockstep)**. 
- Por más núcleos que tenga tu CPU (ej. 8, 12 o 16 núcleos), una sola partida únicamente consume 1 núcleo al ~100%.
- El acelerador NPU **Hailo-8** conectado vía PCIe en la Raspberry Pi 5 procesa cada inferencia de nuestro Transformer en apenas **4.8 ms**.
- Con Action Chunking ($K=4$), cada partida de SC2 solo pide **1 inferencia cada 2.24 segundos** (4 pasos de juego a ~0.56s por paso en modo acelerado).
- **Capacidad ociosa:** La NPU Hailo-8 puede ejecutar más de **200 inferencias por segundo**, lo que significa que con 1 sola partida la NPU pasa el **99% del tiempo esperando**.

---

## 2. Solución: Paralelismo Multi-Instancia Asíncrono

En lugar de intentar forzar multihilo dentro de una sola partida, ejecutamos **múltiples partidas independientes en paralelo**, cada una en su propio núcleo de CPU y con su propia ventana en pantalla, compartiendo la misma NPU Hailo-8 vía TCP.

```
                  ┌─── [Núcleo CPU 0] SC2 Partida #1 (Ventana 1) ───┐
                  │                                                  │
PC Windows (i7/i9)├─── [Núcleo CPU 1] SC2 Partida #2 (Ventana 2) ───┼── TCP (5555) ──► Raspberry Pi 5 + Hailo-8
                  │                                                  │                 (Multi-Threaded Server)
                  └─── [Núcleo CPU 2] SC2 Partida #3 (Opcional) ─────┘                 infer_lock + PCIe Engine
```

### Rendimiento Estimado:
| Configuración | Partidas Simultáneas | Tiempo para 60 Partidas | Carga Hailo-8 | Carga CPU PC |
|---|:---:|:---:|:---:|:---:|
| **Secuencial (Original)** | 1 | ~120 - 150 min | ~0.5% | 1 núcleo al 100% |
| **Paralelo (2 Workers)** | **2** | **~60 - 75 min (2x)** | **~1.0%** | **2 núcleos al 100%** |
| **Paralelo (3 Workers)** | **3** | **~40 - 50 min (3x)** | **~1.5%** | **3 núcleos al 100%** |

---

## 3. Componentes Implementados

### A. Servidor Multi-Cliente en Raspberry Pi (`src/rpi_hailo_server.py`)
- **Concurrencia TCP:** Soporta hasta 16 clientes simultáneos mediante `threading.Thread(target=handle_client)`.
- **Exclusión Mutua NPU (`infer_lock`):** Protege la cola PCIe de la NPU con un mutex ligero. Dado que cada inferencia toma 4.8 ms, el tiempo de espera máximo entre dos workers es prácticamente cero ($< 5\text{ ms}$).

### B. Cliente SC2 con Aislamiento y Mosaico de Ventanas (`src/pc_sc2_client.py`)
- **Mosaico Automático (Window Tiling):** Las ventanas de StarCraft II se posicionan automáticamente en cuadrícula sin solaparse:
  - Worker 0: `(X=40, Y=40, W=640, H=480)`
  - Worker 1: `(X=700, Y=40, W=640, H=480)`
  - Worker 2: `(X=1360, Y=40, W=640, H=480)`
- **Aislamiento de Procesos y Timeout Seguro:** Cada partida corre en un proceso independiente. Si un juego se congela tras 240s, el supervisor solo elimina el PID de ese worker (`taskkill /F /PID <pid>`), protegiendo a los demás workers activos.
- **Bloqueo Atómico de Archivo (`InterprocessFileLock`):** Para evitar que dos workers escriban simultáneamente en `sc2_rl_rollouts.npz` y corrompan los datos, se utiliza un bloqueo atómico a nivel de sistema operativo (`os.O_CREAT | os.O_EXCL`).
- **Currículum Adaptativo Compartido:** Las victorias se agregan en un registro protegido por mutex (`stats_lock`). Si el WinRate reciente supera el 60%, el nivel de dificultad se eleva automáticamente para todos los workers (Easy $\to$ Medium $\to$ Hard $\to$ VeryHard).

### C. Script Ejecutable de 1 Clic (`entrenar_rl_hailo8_paralelo.bat`)
Configurado por defecto para **2 workers concurrentes**, 60 partidas acumuladas y rotación de mapas y razas.

### D. Entrenamiento Paralelo en Google Colab (`SC2_Hailo8_Transformer.ipynb`)
- **DataLoaders Multiproceso:** `critic_loader` y `rl_loader` configurados con `num_workers=2` y `pin_memory=True`.
- Los tensores se transfieren a la GPU en segundo plano en hilos asíncronos mientras los Tensor Cores ejecutan la optimización de política PPO.

---

## 4. Instrucciones de Despliegue (Cuando termine la recolección actual)

> [!IMPORTANT]
> **No interrumpir la recolección en curso:** La recolección actual en Easy sigue corriendo de forma normal y segura. Las siguientes instrucciones aplican una vez que finalice tu sesión actual o cuando desees iniciar el modo paralelo.

### Paso 1: Actualizar el servidor en la Raspberry Pi 5
Desde una terminal en tu PC (PowerShell o CMD):
```bash
scp "C:\Users\Stalin\.gemini\antigravity\scratch\sc2-hailo8-transformer\src\rpi_hailo_server.py" rp5@192.168.50.7:~/sc2-hailo8-transformer/rpi_hailo_server.py
```
*(O reinicia el servicio en la RPi para que cargue la versión multihilo).*

### Paso 2: Lanzar la recolección paralela en tu PC
Doble clic en:
```
entrenar_rl_hailo8_paralelo.bat
```
Verás abrirse dos ventanas de StarCraft II lado a lado jugando simultáneamente.

### Paso 3: Fine-Tuning en Google Colab
Abre `G:\Mi unidad\Colab Notebooks\SC2_Hailo8_Transformer.ipynb` y ejecuta la celda final (**1 Solo Botón**) para compilar la Ronda 5 a `.hef`.
