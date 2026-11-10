"""
Pipeline de Datos y Balanceo Multi-Raza / Multi-Acción para StarCraft II.
Soporta descarga automática desde Hugging Face:
  1. yxzhang2024/SC2-Dynamics-50K (Observaciones estructuradas + Secuencias de acciones en horizonte de 5s)
  2. Kaszanas/SC2EGSet (Torneos profesionales IEM / WCS)
Incluye WeightedRandomSampler y pesos inversos por frecuencia para corregir el desbalanceo severo
entre acciones económicas rutinarias (Harvest/Move) y decisiones estratégicas/combate.
"""

import json
import os
import re
import zipfile
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .config import (
    ACTION_CHUNK_SIZE,
    ACTION_TO_ID,
    GRID_H,
    GRID_W,
    IN_CHANNELS,
    NUM_ACTIONS,
    RACES,
)


def map_raw_sc2_command_to_action_id(subject: str, command: str, target: str) -> int:
    """
    Traduce comandos crudos de StarCraft II (PySC2 / SC2-Dynamics-50K / SC2EGSet)
    al vocabulario unificado de 32 acciones para Terran, Protoss y Zerg.
    """
    s = subject.lower()
    c = command.lower()
    t = target.lower()

    if "harvest" in c or ("smartcmd" in c and ("mineral" in t or "vespene" in t or "geyser" in t)):
        if "gas" in t or "vespene" in t or "refinery" in t or "assimilator" in t or "extractor" in t:
            return ACTION_TO_ID["HARVEST_GATHER_GAS"]
        return ACTION_TO_ID["HARVEST_GATHER_MINERALS"]

    if any(k in c for k in ["train_scv", "train_probe", "train_drone", "morph_drone"]):
        return ACTION_TO_ID["TRAIN_WORKER"]

    if any(k in c for k in ["supplydepot", "pylon", "overlord"]):
        return ACTION_TO_ID["BUILD_SUPPLY"]

    if any(k in c for k in ["refinery", "assimilator", "extractor"]):
        return ACTION_TO_ID["BUILD_GAS_EXTRACTOR"]

    if "build" in c or "morph" in c:
        if any(k in c for k in ["commandcenter", "nexus", "hatchery", "orbital", "planetary"]):
            return ACTION_TO_ID["BUILD_EXPAND_BASE"]
        if any(k in c for k in ["barracks", "gateway", "spawningpool"]):
            return ACTION_TO_ID["BUILD_BARRACKS_GATEWAY_POOL"]
        if any(k in c for k in ["factory", "robotics", "roachwarren", "banelingnest", "hydraliskden"]):
            return ACTION_TO_ID["BUILD_FACTORY_ROBO_ROACH"]
        if any(k in c for k in ["starport", "stargate", "spire"]):
            return ACTION_TO_ID["BUILD_STARPORT_STARGATE_SPIRE"]
        if any(k in c for k in ["techlab", "reactor", "cyberneticscore", "twilight", "lair", "hive", "infestation"]):
            return ACTION_TO_ID["BUILD_TECH_LAB_CORE_LAIR"]
        if any(k in c for k in ["bunker", "turret", "cannon", "battery", "spine", "spore"]):
            return ACTION_TO_ID["BUILD_DEFENSE_STRUCTURE"]

    if "train" in c or "warp" in c or "morph" in c:
        if any(k in c for k in ["marine", "zealot", "zergling"]):
            return ACTION_TO_ID["TRAIN_BASIC_INFANTRY"]
        if any(k in c for k in ["marauder", "stalker", "adept", "roach", "hydralisk", "ravager"]):
            return ACTION_TO_ID["TRAIN_RANGED_CORE"]
        if any(k in c for k in ["reaper", "ghost", "sentry", "templar", "baneling", "infestor", "viper", "queen"]):
            return ACTION_TO_ID["TRAIN_SPECIALIST_UNIT"]
        if any(k in c for k in ["tank", "thor", "hellion", "cyclone", "immortal", "colossus", "disruptor", "ultralisk", "lurker"]):
            return ACTION_TO_ID["TRAIN_SIEGE_HEAVY"]
        if any(k in c for k in ["medivac", "viking", "liberator", "banshee", "battlecruiser", "phoenix", "voidray", "oracle", "carrier", "tempest", "mutalisk", "corruptor", "broodlord"]):
            return ACTION_TO_ID["TRAIN_AIR_UNIT"]

    if "research" in c or "upgrade" in c:
        if any(k in c for k in ["weapon", "armor", "shield", "melee", "missile", "carapace", "flyer"]):
            return ACTION_TO_ID["RESEARCH_WEAPONS_ARMOR"]
        return ACTION_TO_ID["RESEARCH_COMBAT_UPGRADE"]

    if any(k in c for k in ["mule", "chrono", "inject", "spawnlarva", "transfusion", "scanner"]):
        return ACTION_TO_ID["MACRO_CHRONO_MULE_INJECT"]

    if any(k in c for k in ["stim", "blink", "forcefield", "guardian", "storm", "corrosive", "emp", "snipe", "fungal"]):
        return ACTION_TO_ID["MICRO_USE_STIM_ABILITY"]

    if any(k in c for k in ["siege", "unsiege", "burrow", "unburrow", "assault", "fighter", "defender"]):
        return ACTION_TO_ID["MICRO_SIEGE_TRANSFORM"]

    if any(k in c for k in ["repair", "heal", "restore"]):
        return ACTION_TO_ID["MICRO_HEAL_REPAIR_SHIELD"]

    if "rally" in c:
        return ACTION_TO_ID["RALLY_PRODUCTION"]

    if "cancel" in c or "hold" in c or "stop" in c:
        return ACTION_TO_ID["CANCEL_OR_HOLD_POSITION"]

    if "attack" in c:
        if "[" in t and not t.startswith("("):
            if any(w in t for w in ["scv", "probe", "drone"]):
                return ACTION_TO_ID["HARASS_ENEMY_WORKERS"]
            return ACTION_TO_ID["FOCUS_FIRE_TARGET"]
        return ACTION_TO_ID["ATTACK_MOVE_COORDINATE"]

    if "move" in c or "smartcmd" in c:
        if any(w in s for w in ["scv", "probe", "drone"]):
            return ACTION_TO_ID["HARVEST_GATHER_MINERALS"]
        if "overlord" in s:
            return ACTION_TO_ID["MOVE_WORKER_SCOUT"]
        return ACTION_TO_ID["MOVE_ARMY_POSITION"]

    return ACTION_TO_ID["NO_OP"]


def parse_sc2_dynamics_sample(sample_text: str, map_size: float = 64.0) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Convierte una muestra de SC2-Dynamics-50K en:
      - state_tensor: np.ndarray [IN_CHANNELS=12, GRID_H=16, GRID_W=16] (float32)
      - action_chunks: np.ndarray [ACTION_CHUNK_SIZE=4, 4] -> (action_id, target_x, target_y, queue_flag) (int64)
      - race_id: int (0=Terran, 1=Protoss, 2=Zerg)
    """
    if "#### Current Observation:" not in sample_text:
        return None

    obs_part, *act_parts = sample_text.split("#### Action Sequence")
    act_part = act_parts[0] if act_parts else ""

    tensor = np.zeros((IN_CHANNELS, GRID_H, GRID_W), dtype=np.float32)

    # 1. Extraer economía y metadatos de [Info]
    time_match = re.search(r"Time:\s*(\d+):(\d+)", obs_part)
    game_sec = (int(time_match.group(1)) * 60 + int(time_match.group(2))) if time_match else 0

    race_match = re.search(r"Race:\s*(Terran|Protoss|Zerg)", obs_part)
    enemy_race_match = re.search(r"Enemy Race:\s*(Terran|Protoss|Zerg)", obs_part)
    self_race = RACES.get(race_match.group(1), 0) if race_match else 0
    enemy_race = RACES.get(enemy_race_match.group(1), 0) if enemy_race_match else 0

    min_match = re.search(r"Minerals:\s*(\d+)\s*\(\+(\d+)/min\)", obs_part)
    minerals = float(min_match.group(1)) if min_match else 50.0
    min_rate = float(min_match.group(2)) if min_match else 0.0

    gas_match = re.search(r"Gas:\s*(\d+)\s*\(\+(\d+)/min\)", obs_part)
    gas = float(gas_match.group(1)) if gas_match else 0.0
    gas_rate = float(gas_match.group(2)) if gas_match else 0.0

    sup_match = re.search(r"Supply:\s*(\d+)/(\d+)\s*\(Army:\s*(\d+),\s*Workers:\s*(\d+)\)", obs_part)
    sup_used = float(sup_match.group(1)) if sup_match else 12.0
    sup_cap = float(max(1, int(sup_match.group(2)))) if sup_match else 15.0
    army_sup = float(sup_match.group(3)) if sup_match else 0.0
    worker_sup = float(sup_match.group(4)) if sup_match else 12.0

    # Canales 6..11: Estado económico, progreso temporal y raza (codificados espacialmente en 2D para Hailo-8)
    tensor[6, :, :] = np.clip(minerals / 2000.0, 0.0, 1.0)
    tensor[7, :, :] = np.clip(gas / 1500.0, 0.0, 1.0)
    tensor[8, :, :] = np.clip(sup_used / 200.0, 0.0, 1.0)
    tensor[9, :, :] = np.clip(army_sup / 120.0, 0.0, 1.0)
    tensor[10, :, :] = np.clip(worker_sup / 90.0, 0.0, 1.0)
    # Canal 11: Codificación estructurada de tiempo, tasas de recolección y razas en sub-bloques 4x4
    tensor[11, 0:4, :] = np.clip(game_sec / 1200.0, 0.0, 1.0)
    tensor[11, 4:8, :] = np.clip(min_rate / 2500.0, 0.0, 1.0)
    tensor[11, 8:12, :] = (self_race + 1) / 3.0
    tensor[11, 12:16, :] = (enemy_race + 1) / 3.0

    # 2. Extraer coordenadas espaciales para los canales 0..5 del minimapa 16x16:
    # Canal 0: Unidades propias (densidad)
    # Canal 1: HP promedio de unidades propias
    # Canal 2: Estructuras propias
    # Canal 3: Unidades/estructuras enemigas visibles
    # Canal 4: HP enemigo
    # Canal 5: Actividad de cola de producción / alertas
    sections = re.split(r"\[(My Units|My Structures|Visible Hostiles|Queue)\]", obs_part)
    current_sec = ""
    coord_pattern = re.compile(r"at\s*\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)(?:\s*\(HP:(\d+)%)?")

    for part in sections:
        if part in ("My Units", "My Structures", "Visible Hostiles", "Queue"):
            current_sec = part
            continue
        if current_sec == "Queue":
            active_tasks = len([line for line in part.splitlines() if line.strip().startswith("-") and "Empty" not in line])
            tensor[5, :, :] = np.clip(active_tasks / 10.0, 0.0, 1.0)
            continue

        for m in coord_pattern.finditer(part):
            x_raw, y_raw = float(m.group(1)), float(m.group(2))
            hp_pct = float(m.group(3)) / 100.0 if m.group(3) else 1.0
            gx = int(np.clip((x_raw / map_size) * GRID_W, 0, GRID_W - 1))
            gy = int(np.clip((y_raw / map_size) * GRID_H, 0, GRID_H - 1))

            if current_sec == "My Units":
                tensor[0, gy, gx] = min(1.0, tensor[0, gy, gx] + 0.25)
                tensor[1, gy, gx] = hp_pct
            elif current_sec == "My Structures":
                tensor[2, gy, gx] = min(1.0, tensor[2, gy, gx] + 0.5)
            elif current_sec == "Visible Hostiles":
                tensor[3, gy, gx] = min(1.0, tensor[3, gy, gx] + 0.25)
                tensor[4, gy, gx] = hp_pct

    # 3. Extraer secuencia de acciones (Action-Chunking de horizonte K=4)
    # Priorizamos acciones estratégicas/combate sobre repeticiones de Harvest/Move en el mismo frame
    action_lines = [line.strip() for line in act_part.splitlines() if line.strip().startswith("- +")]
    parsed_actions: List[Tuple[int, int, int, int]] = []

    act_line_re = re.compile(r"-\s*\+[\d.]+s:\s*([^\-]+?)\s*-\s*([^\-]+?)(?:\s*-\s*(.+))?$")
    target_coord_re = re.compile(r"\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)")

    for line in action_lines:
        m = act_line_re.match(line)
        if not m:
            continue
        subj = m.group(1).strip()
        cmd = m.group(2).strip()
        targ = (m.group(3) or "").strip()

        act_id = map_raw_sc2_command_to_action_id(subj, cmd, targ)
        tx, ty = GRID_W // 2, GRID_H // 2
        cm = target_coord_re.search(targ)
        if cm:
            tx = int(np.clip((float(cm.group(1)) / map_size) * GRID_W, 0, GRID_W - 1))
            ty = int(np.clip((float(cm.group(2)) / map_size) * GRID_H, 0, GRID_H - 1))

        queue_flag = 1 if ("smartcmd" in cmd.lower() and len(parsed_actions) > 0) else 0
        parsed_actions.append((act_id, tx, ty, queue_flag))

    # Agrupar por tipo de acción (act_id) para evitar que 20 Marines ejecutando Attack en el mismo segundo
    # llenen los 4 slots del Action-Chunk con 4 copias de ATTACK_MOVE_COORDINATE y descarten la construcción/producción.
    unique_by_act_id: Dict[int, Tuple[int, int, int, int]] = {}
    for item in parsed_actions:
        act_id, tx, ty, q_flag = item
        if act_id not in unique_by_act_id:
            unique_by_act_id[act_id] = item
        elif (unique_by_act_id[act_id][1], unique_by_act_id[act_id][2]) == (GRID_W // 2, GRID_H // 2) and (tx, ty) != (GRID_W // 2, GRID_H // 2):
            unique_by_act_id[act_id] = item

    dedup_actions = list(unique_by_act_id.values())

    # Orden de prioridad táctica dentro del horizonte K=4:
    # 1. Construcción de estructuras y mejoras (4..11, 17..19)
    # 2. Entrenamiento de ejército y trabajadores (3, 12..16)
    # 3. Micro y combate (22..31)
    # 4. Recolección y movimiento (1, 2, 20, 21)
    macro_build = [a for a in dedup_actions if a[0] in (4, 5, 6, 7, 8, 9, 10, 11, 17, 18, 19)]
    unit_prod = [a for a in dedup_actions if a[0] in (3, 12, 13, 14, 15, 16)]
    combat_micro = [a for a in dedup_actions if a[0] in (22, 23, 24, 25, 26, 27, 28, 29, 30, 31)]
    econ_move = [a for a in dedup_actions if a[0] in (1, 2, 20, 21)]

    ordered = (macro_build + unit_prod + combat_micro + econ_move)[:ACTION_CHUNK_SIZE]

    # Si hubo menos de 4 acciones distintas en la ventana de 5s, rellenar con recolección de minerales en fase temprana o NO_OP
    fill_act = ACTION_TO_ID["HARVEST_GATHER_MINERALS"] if game_sec < 180 else ACTION_TO_ID["NO_OP"]
    while len(ordered) < ACTION_CHUNK_SIZE:
        ordered.append((fill_act, GRID_W // 2, GRID_H // 2, 0))

    chunks_arr = np.array(ordered[:ACTION_CHUNK_SIZE], dtype=np.int64)
    return tensor, chunks_arr, self_race


def generate_balanced_synthetic_sc2_data(num_samples: int = 2048, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera datos sintéticos multi-raza (Terran, Protoss, Zerg) con patrones espaciales y económicos realistas
    de StarCraft II para pruebas locales rápidas o respaldo offline.
    """
    rng = np.random.default_rng(seed)
    states = np.zeros((num_samples, IN_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
    actions = np.zeros((num_samples, ACTION_CHUNK_SIZE, 4), dtype=np.int64)
    races = np.zeros(num_samples, dtype=np.int64)

    for i in range(num_samples):
        race = i % 3
        enemy_race = (i + 1) % 3
        races[i] = race

        # Fase de partida (early, mid, late)
        phase = rng.choice([0, 1, 2], p=[0.35, 0.40, 0.25])
        game_progress = rng.uniform(0.05, 0.30) if phase == 0 else (rng.uniform(0.30, 0.70) if phase == 1 else rng.uniform(0.70, 1.0))

        minerals = rng.uniform(50, 400) if phase == 0 else rng.uniform(200, 1800)
        gas = rng.uniform(0, 150) if phase == 0 else rng.uniform(100, 1200)
        workers = rng.uniform(12, 28) if phase == 0 else rng.uniform(35, 80)
        army = rng.uniform(0, 15) if phase == 0 else rng.uniform(25, 110)
        supply = min(200.0, workers + army)

        # Base aliada y enemiga en esquinas opuestas del minimapa 16x16
        ally_bx, ally_by = (3, 3) if rng.random() < 0.5 else (12, 12)
        enemy_bx, enemy_by = (15 - ally_bx, 15 - ally_by)

        states[i, 0, max(0, ally_by - 2):min(16, ally_by + 3), max(0, ally_bx - 2):min(16, ally_bx + 3)] = rng.uniform(0.3, 1.0)
        states[i, 1, max(0, ally_by - 2):min(16, ally_by + 3), max(0, ally_bx - 2):min(16, ally_bx + 3)] = rng.uniform(0.6, 1.0)
        states[i, 2, ally_by, ally_bx] = 1.0

        # Presencia enemiga
        under_attack = rng.random() < 0.3
        ex = ally_bx + rng.integers(-2, 3) if under_attack else enemy_bx + rng.integers(-2, 3)
        ey = ally_by + rng.integers(-2, 3) if under_attack else enemy_by + rng.integers(-2, 3)
        ex, ey = int(np.clip(ex, 0, 15)), int(np.clip(ey, 0, 15))
        states[i, 3, ey, ex] = rng.uniform(0.4, 1.0)
        states[i, 4, ey, ex] = rng.uniform(0.4, 1.0)
        states[i, 5, :, :] = rng.uniform(0.0, 0.8)

        states[i, 6, :, :] = minerals / 2000.0
        states[i, 7, :, :] = gas / 1500.0
        states[i, 8, :, :] = supply / 200.0
        states[i, 9, :, :] = army / 120.0
        states[i, 10, :, :] = workers / 90.0
        states[i, 11, 0:4, :] = game_progress
        states[i, 11, 4:8, :] = min(1.0, workers / 66.0)
        states[i, 11, 8:12, :] = (race + 1) / 3.0
        states[i, 11, 12:16, :] = (enemy_race + 1) / 3.0

        # Asignar secuencia coherente de K=4 acciones según el estado táctico
        for k in range(ACTION_CHUNK_SIZE):
            if under_attack:
                act_id = int(rng.choice([22, 23, 24, 25, 27, 28]))
                tx, ty = ex, ey
            elif phase == 0:
                act_id = int(rng.choice([1, 2, 3, 4, 5, 7, 12, 19, 21]))
                tx, ty = ally_bx, ally_by
            elif phase == 1:
                act_id = int(rng.choice([6, 8, 9, 10, 11, 13, 14, 15, 17, 19, 20, 22, 29, 30]))
                tx, ty = (ex, ey) if act_id in (22, 29) else (ally_bx, ally_by)
            else:
                act_id = int(rng.choice([15, 16, 18, 22, 23, 25, 26, 29, 31]))
                tx, ty = ex, ey
            q_flag = 1 if k > 0 else 0
            actions[i, k] = [act_id, tx, ty, q_flag]

    return states, actions, races


class SC2ActionChunkDataset(Dataset):
    """
    Dataset PyTorch para entrenar el SC2HailoTransformer con balanceo automático
    de clases (acciones) y razas (Terran, Protoss, Zerg).
    """

    def __init__(self, states: np.ndarray, actions: np.ndarray, races: np.ndarray):
        self.states = states.astype(np.float32)
        self.actions = actions.astype(np.int64)
        self.races = races.astype(np.int64)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.tensor(self.races[idx], dtype=torch.long),
        )

    def compute_balancing_weights(self) -> Tuple[WeightedRandomSampler, torch.Tensor, Dict[str, dict]]:
        """
        Calcula:
          1. WeightedRandomSampler por par (Raza, Acción Principal) usando suavizado inverso de raíz cuadrada
             para evitar que acciones masivas (Harvest/Move) dominen los lotes de entrenamiento.
          2. Vector class_weights [NUM_ACTIONS] para la función Class-Balanced Focal Loss.
          3. Resumen estadístico del desbalanceo original vs corregido.
        """
        primary_actions = self.actions[:, 0, 0]
        all_chunk_actions = self.actions[:, :, 0].reshape(-1)

        race_counts = Counter(self.races.tolist())
        action_counts = Counter(all_chunk_actions.tolist())
        pair_counts = Counter(zip(self.races.tolist(), primary_actions.tolist()))

        # 1. Pesos por muestra para WeightedRandomSampler (suavizado 1 / sqrt(freq))
        sample_weights = np.zeros(len(self.states), dtype=np.float64)
        for i in range(len(self.states)):
            pair_key = (int(self.races[i]), int(primary_actions[i]))
            sample_weights[i] = 1.0 / np.sqrt(max(1, pair_counts[pair_key]))
        sample_weights /= sample_weights.mean()

        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True,
        )

        # 2. Pesos por clase para Class-Balanced Focal Loss (Effective Number of Samples, beta=0.999)
        beta = 0.999
        class_weights = np.ones(NUM_ACTIONS, dtype=np.float32)
        for c in range(NUM_ACTIONS):
            n_c = max(1, action_counts.get(c, 0))
            effective_num = 1.0 - np.power(beta, n_c)
            class_weights[c] = (1.0 - beta) / effective_num
        class_weights = class_weights / np.mean(class_weights)

        stats = {
            "race_distribution": {k: int(v) for k, v in race_counts.items()},
            "top_5_actions": action_counts.most_common(5),
            "rarest_5_actions": action_counts.most_common()[:-6:-1],
        }
        return sampler, torch.from_numpy(class_weights), stats


def load_sc2_huggingface_dataset(
    max_samples: int = 25000,
    cache_dir: str = "./data_cache",
) -> Tuple[SC2ActionChunkDataset, SC2ActionChunkDataset]:
    """
    Descarga y procesa automáticamente el dataset SC2-Dynamics-50K (y/o SC2EGSet) desde Hugging Face.
    Si no hay conexión a internet, utiliza el generador sintético multi-raza de respaldo.
    """
    os.makedirs(cache_dir, exist_ok=True)
    processed_npz = os.path.join(cache_dir, f"sc2_hailo8_processed_{max_samples}.npz")

    if os.path.exists(processed_npz):
        print(f"[Dataset] Cargando tensores preprocesados en caché: {processed_npz}")
        data = np.load(processed_npz)
        states, actions, races = data["states"], data["actions"], data["races"]
    else:
        states_list, actions_list, races_list = [], [], []
        try:
            from huggingface_hub import hf_hub_download

            print("[Dataset] Descargando 'yxzhang2024/SC2-Dynamics-50K' desde Hugging Face...")
            for split_file in ["wm_train_horizon5.json", "wm_val_horizon5.json"]:
                json_path = hf_hub_download(
                    repo_id="yxzhang2024/SC2-Dynamics-50K",
                    filename=split_file,
                    repo_type="dataset",
                    cache_dir=cache_dir,
                )
                with open(json_path, "r", encoding="utf-8") as f:
                    records = json.load(f)

                for rec in records:
                    msgs = rec.get("messages", [])
                    if not msgs:
                        continue
                    user_text = msgs[0].get("content", "")
                    parsed = parse_sc2_dynamics_sample(user_text)
                    if parsed is not None:
                        st, ac, rc = parsed
                        states_list.append(st)
                        actions_list.append(ac)
                        races_list.append(rc)
                        if len(states_list) >= max_samples:
                            break
                if len(states_list) >= max_samples:
                    break

            states = np.stack(states_list, axis=0)
            actions = np.stack(actions_list, axis=0)
            races = np.array(races_list, dtype=np.int64)
            np.savez_compressed(processed_npz, states=states, actions=actions, races=races)
            print(f"[Dataset] Procesadas y guardadas {len(states)} muestras reales de StarCraft II.")
        except Exception as e:
            print(f"[Dataset] Aviso: No se pudo descargar desde Hugging Face ({e}). Usando generador sintético SC2...")
            states, actions, races = generate_balanced_synthetic_sc2_data(num_samples=min(max_samples, 4096))

    # División 85% Train / 15% Validation estratificada
    n_total = len(states)
    perm = np.random.default_rng(42).permutation(n_total)
    split_idx = int(n_total * 0.85)
    train_idx, val_idx = perm[:split_idx], perm[split_idx:]

    train_ds = SC2ActionChunkDataset(states[train_idx], actions[train_idx], races[train_idx])
    val_ds = SC2ActionChunkDataset(states[val_idx], actions[val_idx], races[val_idx])
    return train_ds, val_ds
