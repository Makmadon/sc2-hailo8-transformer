# Auditoría de Ruido, Eliminación de Muletillas y Reestructuración de Recompensas RL

**Fecha:** 10 de Noviembre de 2026  
**Proyecto:** SC2 Hailo-8 Transformer (StarCraft II Agent 100% Neural)  
**Objetivo:** Garantizar un aprendizaje por refuerzo (Actor-Critic PPO / AWR) matemáticamente riguroso, libre de muletillas algorítmicas, sin contaminación por señales sintéticas y con alineación estricta de la atención espacial.

---

## 1. Diagnóstico y Problemas Observados

Durante las pruebas de inferencia y recolección de las primeras rondas se detectó un patrón subóptimo:
- **Inercia espacial:** El ejército aliado marchaba hacia la posición original de la base enemiga ya demolida, ignorando al ejército rival que maniobraba en otra zona del mapa.
- **Baja probabilidad de descubrimiento por azar:** Con una malla $16 \times 16 = 256$ celdas, la exploración $\varepsilon$-greedy puramente uniforme ($0.16 \times \frac{1}{256} = 0.06\%$) casi nunca acertaba a pulsar sobre la coordenada exacta donde aparecía el ejército enemigo.
- **Interferencia de gradientes:** Acciones económicas (como entrenar un trabajador o recolectar minerales) forzaban al cabezal de coordenadas espaciales a predecir `(8, 8)` o `(bgx, bgy)`, contaminando la atención táctica de ataque y movimiento.

---

## 2. Auditoría Exhaustiva de Fuentes de Ruido Eliminadas

Se realizó una auditoría completa del cliente (`src/pc_sc2_client.py`) y del cuaderno de entrenamiento (`SC2_Hailo8_Transformer.ipynb`). Se identificaron y eliminaron cuatro fuentes críticas de ruido:

### A. Señales Sintéticas en `build_obs_tensor`
- **Minerales falseados:** Cuando el bot tenía menos de 22 trabajadores y $>180$ minerales, se inyectaba una señal cíclica artificial (`65.0 + minerals % 115.0`) para asemejarlo al dataset estático. Esto alteraba la percepción real del estado financiero.
- **Reloj forzado:** Se forzaba un tiempo mínimo de 25 segundos al inicio de la partida.
- **Tasas mínimas inventadas:** Se sobreescribía la tasa de recolección de minerales con una cota inferior teórica (`supply_workers * 45.0`).
- **Cola de órdenes falseada:** El Canal 5 se forzaba con `max(orders, 6)`, reportando actividad de producción constante aun con edificios inactivos.
- **Marcadores de unidades fantasma:** En los Canales 0, 1 y 2 existían tuplas de coordenadas fijas heredadas (`(6, 7), (7, 7), (8, 15)...`).
- **Estado Actual:** Todos los 12 canales ahora reflejan con fidelidad matemática exacta las unidades, estructuras, recursos y tiempos reales del juego.

### B. Muestreo de Coordenadas en la Exploración (`apply_rl_exploration`)
- Anteriormente, el explorador muestreaba celdas uniformes sin correlación con el juego.
- Ahora, [`sample_combat_cell`](file:///C:/Users/Stalin/.gemini/antigravity/scratch/sc2-hailo8-transformer/src/pc_sc2_client.py#L750-L760) consulta directamente el tensor de observación:
  1. Si hay presencia en el **Canal 3** (enemigos visibles), el 70% de las exploraciones de combate se dirigen hacia esas celdas.
  2. Si no hay enemigos visibles, prioriza celdas aún no visitadas en la cuadrícula de exploración para promover el barrido del mapa.

### C. Máscara de Pérdida Espacial en el Entrenamiento (Puntos Críticos en Notebook)
- **Problema previo:** Se calculaba `ce_tx` y `ce_ty` para todas las acciones del vocabulario indiscriminadamente. Al entrenar un SCV o minar gas, el target asignado en los datos era un valor por defecto que obligaba a la red a predecir el centro del mapa.
- **Solución implementada:** Se integró una máscara booleana `is_spatial` que restringe el cálculo de la pérdida de coordenadas espaciales exclusivamente a las 18 acciones que genuinamente usan un objetivo en el mapa:
  $$\mathcal{L}_{\text{spatial}} = ( \text{CE}_{tx} + \text{CE}_{ty} ) \cdot \mathbb{I}(a \in \mathcal{A}_{\text{spatial}})$$
  Donde $\mathcal{A}_{\text{spatial}} = \{4, 6, 7, 8, 9, 10, 11, 18, 20, 21, 22, 23, 24, 26, 28, 29, 30, 31\}$.

### D. Corrección de Pesos de Ventaja Negativa (AWR / MARWIL)
- **Problema previo:** Para $A_t < 0$, la fórmula anterior usaba `clamp(0.15 + 0.25 * badv, -0.25, 0.15)`. Cuando una jugada tenía ventaja ligeramente negativa (ej. $A_t = -0.5$), el peso resultante era $+0.025 > 0$, entrenando a la red por imitación supervisada a repetir acciones peores que el promedio.
- **Solución implementada:** Se adoptó la ponderación exponencial monótona estándar de Advantage-Weighted Regression (AWR):
  $$w(A_t) = \text{clamp}\left(\exp\left(\frac{A_t}{1.0}\right), 0.02, 5.0\right)$$
  - Si $A_t = +2.0$: peso $= 5.0$ (refuerzo máximo).
  - Si $A_t = 0.0$: peso $= 1.0$ (neutral).
  - Si $A_t = -2.0$: peso $= 0.13$ (supresión severa).
  - Si $A_t = -3.0$: peso $= 0.05$ (ignorado casi en su totalidad).
  - Nunca se generan pesos negativos que puedan provocar divergencia de la entropía cruzada.

---

## 3. Eliminación Completa de Muletillas Algorítmicas

En [`src/pc_sc2_client.py`](file:///C:/Users/Stalin/.gemini/antigravity/scratch/sc2-hailo8-transformer/src/pc_sc2_client.py#L309-L720) se removieron todas las dependencias cruzadas y decisiones secundarias automáticas que sustituían las decisiones de la red:

1. **`[03] TRAIN_WORKER`:** Ya no revisa si falta suministro ni manda a construir Depósitos de Suministro por su cuenta. Si la red necesita suministro, debe predecir `[04] BUILD_SUPPLY`.
2. **`[07] BUILD_BARRACKS`:** Se eliminó la construcción automática de Depósitos de Suministro previos.
3. **`[11] BUILD_DEFENSE`:** Se eliminó la verificación y construcción encadenada de Depósitos y Cuarteles para habilitar el búnker.
4. **`[13] TRAIN_RANGED_CORE`:** Se eliminó la orden que construía automáticamente un Tech Lab en el cuartel si no existía.
5. **`[15] TRAIN_SIEGE_HEAVY`:** Se eliminó la orden que construía automáticamente un Tech Lab en la fábrica.
6. **`[18] RESEARCH_WEAPONS_ARMOR`:** Se eliminó la orden que construía automáticamente la Bahía de Ingeniería.
7. **Exploración:** Se eliminó la acción espuria que bajaba depósitos sin motivo táctico y el uso de Stim/Siege cuando no existían unidades militares en el campo.

---

## 4. Reestructuración de la Función de Recompensa

En [`compute_chunk_reward`](file:///C:/Users/Stalin/.gemini/antigravity/scratch/sc2-hailo8-transformer/src/pc_sc2_client.py#L825-L935) se reformuló el cálculo denso $r_t$ (acotado en $[-2.5, +2.5]$):

### Premios ($+$):
- **Daño a unidades enemigas:** $+\frac{\Delta \text{killed\_units}}{200.0}$
- **Daño a estructuras enemigas:** $+1.5 \times \frac{\Delta \text{killed\_structures}}{200.0}$
- **Crecimiento de ejército:** $+\frac{\Delta \text{army\_value}}{400.0}$
- **Producción de obreros:** $+0.15 \times \Delta \text{SCVs}$
- **Expansión económica:** $+0.80$ por nuevo Centro de Mando / base.
- **Atención a enemigos visibles:** $+0.30$ cuando una acción de combate apunta a $\le 1$ celda de donde el Canal 3 muestra hostiles.
- **Exploración territorial (Count-Based):** $+0.12$ (una sola vez por celda) cuando el ejército recorre una casilla de la malla $16 \times 16$ nunca antes visitada en la partida.
- **Terminal:** Victoria real $+10.0$ (o $+2.5$ por ventaja notable en timeout).

### Penalizaciones ($-$):
- **Estructuras aliadas destruidas:** Se calcula mediante el hook [`on_unit_destroyed`](file:///C:/Users/Stalin/.gemini/antigravity/scratch/sc2-hailo8-transformer/src/pc_sc2_client.py#L227-L245) que registra el costo real en minerales y gas de cada estructura derribada:
  $$\Delta r_{\text{lost}} = -1.5 \times \frac{\text{valor}}{200.0} - 0.80 \times \text{townhalls\_perdidos}$$
  *(Las cancelaciones voluntarias antes del colapso no reciben penalización).*
- **Ignorar enemigos visibles:** $-0.25$ cuando hay hostiles en el Canal 3 pero la red emite ataque hacia celdas vacías distantes.
- **Estancamiento en ruinas:** $-0.20$ si no hay enemigos visibles y el ejército insiste en atacar la misma celda vacía donde ya está situado.
- **Bloqueo de suministro (*Supply Block*):** $-0.15$ si `supply_left <= 0` con minerales suficientes.
- **Acumulación ociosa (*Floating Minerals*):** $-0.10$ si acumula $>600$ minerales con suministro $<185$.
- **Terminal:** Derrota $-8.0$ (o $-2.5$ por desventaja en timeout).

---

## 5. Respaldo y Versionado del Buffer de Experiencia

Las 25,000 muestras recolectadas en las Rondas 1 a 4 fueron generadas bajo el esquema anterior (con señales falseadas y sin penalizaciones de estructuras). Mezclar esas muestras con datos limpios causaría un desajuste severo en la distribución (*distribution shift*) y en los objetivos del Crítico $V_\phi(s_t)$.

Por ello, se procedió a su respaldo y aislamiento:
- **Respaldo en Google Drive:**  
  `G:\Mi unidad\hailo_files\backup_v1_legacy_rollouts\sc2_rl_rollouts_round4_legacy.npz`
- **Respaldo Local:**  
  `models/backup_v1_legacy_rollouts/sc2_rl_rollouts_round4_legacy.npz`
- **Estado del Buffer Activo:**  
  El archivo activo `sc2_rl_rollouts.npz` se reinicializó para que la próxima ejecución de `entrenar_rl_hailo8.bat` acumule experiencia 100% limpia bajo la nueva función de recompensa y observaciones no contaminadas.
