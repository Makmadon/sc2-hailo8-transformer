"""
Pipeline de Datos V2 para Terran:
  1. Extractor real de replays `.SC2Replay` usando `ObserverAI` + `extract_state_and_masks`
     (cero domain-shift respecto al bot en vivo).
  2. Conversor Terran de `yxzhang2024/SC2-Dynamics-50K` (útil como respaldo sin motor SC2).
  3. Datasets PyTorch (`SC2V2BCDataset`, `SC2V2RLDataset`) optimizados para CPU/GPU.
"""

import argparse
import asyncio
import glob
import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, WeightedRandomSampler

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Dataset = object
    WeightedRandomSampler = None
    _TORCH_AVAILABLE = False

from .config import (
    ENEMY_RACES,
    GLOBAL_DIM,
    GRID_H,
    GRID_W,
    MACRO_TO_ID,
    MAP_CHANNELS,
    MICRO_TO_ID,
    NUM_MACRO,
    NUM_MICRO,
)
from .features import EpisodeTrackerV2, extract_state_and_masks, get_playable_bounds, world_to_grid


# ============================================================================
# 1. Mapeo de AbilityId de StarCraft II -> Acciones V2 (Macro y Micro)
# ============================================================================

def map_ability_name_to_v2(ab_name: str, target_is_worker: bool = False) -> Tuple[Optional[int], Optional[int]]:
    """
    Mapea el nombre de un AbilityId crudo de StarCraft II a (macro_id, micro_id).
    Retorna None en el cabezal que no corresponda.
    """
    n = ab_name.upper()

    # --- MACRO ---
    if "HARVEST_GATHER" in n:
        return MACRO_TO_ID["ASSIGN_MINERALS"], None
    if "TRAIN_SCV" in n:
        return MACRO_TO_ID["TRAIN_SCV"], None
    if "BUILD_SUPPLYDEPOT" in n:
        return MACRO_TO_ID["BUILD_SUPPLY_DEPOT"], None
    if "BUILD_REFINERY" in n:
        return MACRO_TO_ID["BUILD_REFINERY"], None
    if "BUILD_COMMANDCENTER" in n:
        return MACRO_TO_ID["BUILD_COMMAND_CENTER"], None
    if "BUILD_BARRACKS" in n:
        return MACRO_TO_ID["BUILD_BARRACKS"], None
    if "BUILD_FACTORY" in n:
        return MACRO_TO_ID["BUILD_FACTORY"], None
    if "BUILD_STARPORT" in n:
        return MACRO_TO_ID["BUILD_STARPORT"], None
    if "BUILD_ENGINEERINGBAY" in n:
        return MACRO_TO_ID["BUILD_ENGINEERING_BAY"], None
    if "BUILD_ARMORY" in n:
        return MACRO_TO_ID["BUILD_ARMORY"], None
    if "BUILD_BUNKER" in n:
        return MACRO_TO_ID["BUILD_BUNKER"], None
    if "BUILD_MISSILETURRET" in n:
        return MACRO_TO_ID["BUILD_MISSILE_TURRET"], None
    if "UPGRADETOORBITAL" in n:
        return MACRO_TO_ID["UPGRADE_ORBITAL_COMMAND"], None
    if "TECHLAB" in n:
        return MACRO_TO_ID["BUILD_ADDON_TECHLAB"], None
    if "REACTOR" in n:
        return MACRO_TO_ID["BUILD_ADDON_REACTOR"], None
    if "CALLDOWNMULE" in n:
        return MACRO_TO_ID["CALLDOWN_MULE"], None
    if "TRAIN_MARINE" in n:
        return MACRO_TO_ID["TRAIN_MARINE"], None
    if "TRAIN_MARAUDER" in n:
        return MACRO_TO_ID["TRAIN_MARAUDER"], None
    if "TRAIN_REAPER" in n:
        return MACRO_TO_ID["TRAIN_REAPER"], None
    if "TRAIN_HELLION" in n or "TRAIN_HELLBAT" in n:
        return MACRO_TO_ID["TRAIN_HELLION"], None
    if "TRAIN_SIEGETANK" in n:
        return MACRO_TO_ID["TRAIN_SIEGE_TANK"], None
    if "TRAIN_THOR" in n or "TRAIN_CYCLONE" in n:
        return MACRO_TO_ID["TRAIN_THOR_OR_CYCLONE"], None
    if "TRAIN_MEDIVAC" in n:
        return MACRO_TO_ID["TRAIN_MEDIVAC"], None
    if "TRAIN_VIKING" in n or "TRAIN_LIBERATOR" in n:
        return MACRO_TO_ID["TRAIN_VIKING_OR_LIBERATOR"], None
    if "RESEARCH_STIMPACK" in n or "RESEARCH_COMBATSHIELD" in n or "RESEARCH_CONCUSSIVESHELLS" in n:
        return MACRO_TO_ID["RESEARCH_BIO_TECH"], None
    if "RESEARCH_TERRANINFANTRY" in n or "RESEARCH_TERRANVEHICLE" in n or "RESEARCH_TERRANSHIP" in n:
        return MACRO_TO_ID["RESEARCH_UPGRADES"], None

    # --- MICRO ---
    if "SCANNERSWEEP" in n:
        return None, MICRO_TO_ID["SCANNER_SWEEP"]
    if "EFFECT_STIM" in n:
        return None, MICRO_TO_ID["USE_STIMPACK"]
    if "SIEGEMODE" in n:
        return None, MICRO_TO_ID["SIEGE_TANKS_AT_AREA"]
    if "UNSIEGE" in n:
        return None, MICRO_TO_ID["UNSIEGE_TANKS_TO_MOVE"]
    if "EFFECT_REPAIR" in n:
        return None, MICRO_TO_ID["REPAIR_WITH_SCV"]
    if "MORPH_SUPPLYDEPOT" in n:
        return None, MICRO_TO_ID["LOWER_OR_RAISE_DEPOTS"]
    if "ATTACK_ATTACK" in n or "ATTACK" == n:
        return None, MICRO_TO_ID["ATTACK_MOVE_ARMY"]
    if "MOVE_MOVE" in n or "SMART" in n:
        if target_is_worker:
            return None, MICRO_TO_ID["SCOUT_WITH_SCV_OR_REAPER"]
        return None, MICRO_TO_ID["MOVE_ARMY"]

    return None, None


# ============================================================================
# 2. Extractor de Replays Reales (.SC2Replay) vía python-sc2 ObserverAI
# ============================================================================

def extract_single_replay(
    replay_path: str,
    observed_player_id: int = 1,
    step_interval: int = 8,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Reproduce un archivo .SC2Replay en segundo plano con el motor SC2 (en Windows)
    y extrae transiciones idénticas a la inferencia en vivo V2.
    """
    from sc2.data import Race
    from sc2.ids.ability_id import AbilityId
    from sc2.ids.unit_typeid import UnitTypeId
    from sc2.observer_ai import ObserverAI
    from sc2.position import Point2
    import sc2.main as sc2_main

    if not hasattr(sc2_main, "run_replay"):
        raise RuntimeError(
            "Tu versión instalada de burnysc2/python-sc2 no expone sc2.main.run_replay. "
            "Actualiza burnysc2 o usa --collect-teacher / --from-hf."
        )

    collected = {
        "map_obs": [],
        "vec_obs": [],
        "macro_mask": [],
        "micro_mask": [],
        "macro_id": [],
        "micro_id": [],
        "macro_xy": [],
        "micro_xy": [],
    }

    class TerranReplayExtractor(ObserverAI):
        def __init__(self):
            super().__init__()
            self.tracker = EpisodeTrackerV2()
            self.terrain_cache = None
            self.pending_sample = None

        async def on_start(self):
            self.client.game_step = step_interval

        async def on_step(self, iteration: int):
            if getattr(self, "race", None) != Race.Terran:
                return

            bounds = get_playable_bounds(self.game_info)
            map_obs, vec_obs, m_mask, mi_mask, self.terrain_cache = extract_state_and_masks(
                self, tracker=self.tracker, terrain_cache=self.terrain_cache
            )

            # Leer acciones crudas ejecutadas por el jugador en este bloque de pasos
            macro_id = 0
            micro_id = 0
            macro_xy = GRID_W * (GRID_H // 2) + (GRID_W // 2)
            micro_xy = macro_xy

            worker_tags = {u.tag for u in self.units(UnitTypeId.SCV)}
            for raw_act in getattr(self.state, "actions", []):
                uc = getattr(raw_act, "unit_command", None)
                if uc is None:
                    continue
                try:
                    ab_name = AbilityId(uc.ability_id).name
                except Exception:
                    continue
                is_w = any(t in worker_tags for t in getattr(uc, "unit_tags", []))
                m_id, mi_id = map_ability_name_to_v2(ab_name, target_is_worker=is_w)

                gx, gy = GRID_W // 2, GRID_H // 2
                if uc.HasField("target_world_space_pos"):
                    pos = uc.target_world_space_pos
                    gx, gy = world_to_grid(Point2((pos.x, pos.y)), bounds)
                elif uc.HasField("target_unit_tag"):
                    tu = self.units.find_by_tag(uc.target_unit_tag) or (self.enemy_units + self.enemy_structures).find_by_tag(uc.target_unit_tag)
                    if tu is not None:
                        gx, gy = world_to_grid(tu.position, bounds)

                flat_xy = int(gy * GRID_W + gx)
                if m_id is not None and m_mask[m_id] and macro_id == 0:
                    macro_id = m_id
                    macro_xy = flat_xy
                if mi_id is not None and mi_mask[mi_id] and micro_id == 0:
                    micro_id = mi_id
                    micro_xy = flat_xy

            # Descartar pasos completamente ociosos para evitar sesgo hacia NO_OP
            if macro_id != 0 or micro_id != 0 or (iteration % (step_interval * 4) == 0):
                collected["map_obs"].append(map_obs[0])
                collected["vec_obs"].append(vec_obs[0])
                collected["macro_mask"].append(m_mask)
                collected["micro_mask"].append(mi_mask)
                collected["macro_id"].append(macro_id)
                collected["micro_id"].append(micro_id)
                collected["macro_xy"].append(macro_xy)
                collected["micro_xy"].append(micro_xy)

            self.tracker.update_action(macro_id, micro_id)

    sc2_main.run_replay(TerranReplayExtractor(), replay_path=replay_path, realtime=False, observed_id=observed_player_id)

    if not collected["map_obs"]:
        return None

    return {
        "map_obs": np.stack(collected["map_obs"], axis=0).astype(np.float32),
        "vec_obs": np.stack(collected["vec_obs"], axis=0).astype(np.float32),
        "macro_mask": np.stack(collected["macro_mask"], axis=0).astype(bool),
        "micro_mask": np.stack(collected["micro_mask"], axis=0).astype(bool),
        "macro_id": np.array(collected["macro_id"], dtype=np.int64),
        "micro_id": np.array(collected["micro_id"], dtype=np.int64),
        "macro_xy": np.array(collected["macro_xy"], dtype=np.int64),
        "micro_xy": np.array(collected["micro_xy"], dtype=np.int64),
    }


def extract_replays_from_folder(
    replays_dir: str,
    out_npz: str = "data/sc2_v2_bc_dataset.npz",
    max_replays: int = 100,
    step_interval: int = 8,
) -> str:
    replay_files = sorted(glob.glob(os.path.join(replays_dir, "**", "*.SC2Replay"), recursive=True))[:max_replays]
    if not replay_files:
        raise FileNotFoundError(f"No se encontraron archivos .SC2Replay en: {replays_dir}")

    shards = []
    for idx, rp in enumerate(replay_files, 1):
        print(f"[Replay {idx}/{len(replay_files)}] Extrayendo: {os.path.basename(rp)}")
        for pid in (1, 2):
            try:
                data = extract_single_replay(rp, observed_player_id=pid, step_interval=step_interval)
                if data is not None and len(data["macro_id"]) > 10:
                    shards.append(data)
                    print(f"  -> Player {pid}: +{len(data['macro_id'])} muestras Terran")
            except Exception as exc:
                print(f"  -> Player {pid} omitido ({exc})")

    if not shards:
        raise RuntimeError("No se extrajeron muestras Terran válidas de los replays.")

    merged = {
        key: np.concatenate([s[key] for s in shards], axis=0)
        for key in shards[0].keys()
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    np.savez_compressed(out_npz, **merged)
    print(f"[Dataset V2] Guardadas {len(merged['macro_id'])} muestras reales de replays en: {out_npz}")
    return out_npz


# ============================================================================
# 3. Conversor Terran de SC2-Dynamics-50K (Respaldo offline/WSL sin motor SC2)
# ============================================================================

def parse_sc2_dynamics_terran_v2(sample_text: str, map_size: float = 64.0) -> Optional[Dict[str, np.ndarray]]:
    """Convierte una muestra Terran de SC2-Dynamics-50K al formato V2 (32x32 + global 64)."""
    if "#### Current Observation:" not in sample_text:
        return None

    obs_part, *act_parts = sample_text.split("#### Action Sequence")
    act_part = act_parts[0] if act_parts else ""

    race_match = re.search(r"Race:\s*(Terran|Protoss|Zerg)", obs_part)
    if not race_match or race_match.group(1) != "Terran":
        return None

    enemy_match = re.search(r"Enemy Race:\s*(Terran|Protoss|Zerg)", obs_part)
    erace = ENEMY_RACES.get(enemy_match.group(1), 0) if enemy_match else 0

    time_match = re.search(r"Time:\s*(\d+):(\d+)", obs_part)
    game_sec = (int(time_match.group(1)) * 60 + int(time_match.group(2))) if time_match else 0

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

    map_obs = np.zeros((MAP_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
    map_obs[14, :, :] = 0.65  # terreno caminable por defecto
    map_obs[15, :, :] = 0.50  # explorado parcial

    vec_obs = np.zeros((GLOBAL_DIM, 1, 1), dtype=np.float32)
    gv = vec_obs[:, 0, 0]
    gv[0] = np.clip(minerals / 2000.0, 0.0, 1.0)
    gv[1] = np.clip(gas / 1500.0, 0.0, 1.0)
    gv[2] = np.clip(sup_used / 200.0, 0.0, 1.0)
    gv[3] = np.clip(sup_cap / 200.0, 0.0, 1.0)
    gv[4] = np.clip(max(0.0, sup_cap - sup_used) / 20.0, 0.0, 1.0)
    gv[5] = np.clip(worker_sup / 80.0, 0.0, 1.0)
    gv[6] = np.clip(army_sup / 150.0, 0.0, 1.0)
    gv[8] = np.clip(game_sec / 1800.0, 0.0, 1.0)
    gv[9] = np.clip(min_rate / 2500.0, 0.0, 1.0)
    gv[10] = np.clip(gas_rate / 1200.0, 0.0, 1.0)
    if erace in (0, 1, 2):
        gv[11 + erace] = 1.0

    ally_hp_sum = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    ally_cnt = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    enemy_hp_sum = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    enemy_cnt = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    sections = re.split(r"\[(My Units|My Structures|Visible Hostiles|Queue)\]", obs_part)
    current_sec = ""
    item_re = re.compile(r"-\s*([^:]+?):\s*.*?at\s*\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)(?:\s*\(HP:(\d+)%)?")

    # Detectar escala real del mapa observando la máxima coordenada si supera 64
    all_coords = re.findall(r"\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)", obs_part)
    max_coord = max([max(float(a), float(b)) for a, b in all_coords], default=map_size)
    eff_map = max(map_size, max_coord + 2.0)

    has_rax = has_fac = has_sp = has_ebay = has_armory = False
    has_tl_rax = has_tl_fac = has_cc = False

    for part in sections:
        if part in ("My Units", "My Structures", "Visible Hostiles", "Queue"):
            current_sec = part
            continue
        for m in item_re.finditer(part):
            name = m.group(1).lower()
            gx = int(np.clip((float(m.group(2)) / eff_map) * GRID_W, 0, GRID_W - 1))
            gy = int(np.clip((float(m.group(3)) / eff_map) * GRID_H, 0, GRID_H - 1))
            hp = float(m.group(4)) / 100.0 if m.group(4) else 1.0

            if current_sec == "My Units":
                ally_hp_sum[gy, gx] += hp
                ally_cnt[gy, gx] += 1.0
                if "scv" in name or "mule" in name:
                    map_obs[0, gy, gx] = min(1.0, map_obs[0, gy, gx] + 1.0 / 12.0)
                elif any(k in name for k in ("tank", "hellion", "thor", "cyclone")):
                    map_obs[2, gy, gx] = min(1.0, map_obs[2, gy, gx] + 1.0 / 6.0)
                elif any(k in name for k in ("medivac", "viking", "liberator", "banshee", "raven", "battlecruiser")):
                    map_obs[3, gy, gx] = min(1.0, map_obs[3, gy, gx] + 1.0 / 6.0)
                else:
                    map_obs[1, gy, gx] = min(1.0, map_obs[1, gy, gx] + 1.0 / 12.0)
            elif current_sec == "My Structures":
                if any(k in name for k in ("commandcenter", "orbital", "planetary")):
                    map_obs[5, gy, gx] = min(1.0, map_obs[5, gy, gx] + 0.5)
                    has_cc = True
                    gv[14] = min(1.0, gv[14] + 0.2)
                elif any(k in name for k in ("supplydepot", "refinery", "bunker", "turret")):
                    map_obs[7, gy, gx] = min(1.0, map_obs[7, gy, gx] + 0.25)
                else:
                    map_obs[6, gy, gx] = min(1.0, map_obs[6, gy, gx] + 0.25)
                    if "barracks" in name:
                        has_rax = True
                        gv[18] = min(1.0, gv[18] + 0.1)
                    if "barrackstechlab" in name or "techlab" in name:
                        has_tl_rax = True
                        has_tl_fac = True
                    if "factory" in name:
                        has_fac = True
                        gv[21] = min(1.0, gv[21] + 0.2)
                    if "starport" in name:
                        has_sp = True
                        gv[23] = min(1.0, gv[23] + 0.25)
                    if "engineeringbay" in name:
                        has_ebay = True
                    if "armory" in name:
                        has_armory = True
            elif current_sec == "Visible Hostiles":
                enemy_hp_sum[gy, gx] += hp
                enemy_cnt[gy, gx] += 1.0
                if any(k in name for k in ("scv", "probe", "drone")):
                    map_obs[8, gy, gx] = min(1.0, map_obs[8, gy, gx] + 1.0 / 12.0)
                elif any(k in name for k in ("command", "nexus", "hatchery", "barracks", "gateway", "pool", "pylon", "cannon")):
                    map_obs[11, gy, gx] = min(1.0, map_obs[11, gy, gx] + 0.25)
                else:
                    map_obs[9, gy, gx] = min(1.0, map_obs[9, gy, gx] + 1.0 / 10.0)
                map_obs[15, gy, gx] = 1.0

    map_obs[4][ally_cnt > 0] = ally_hp_sum[ally_cnt > 0] / ally_cnt[ally_cnt > 0]
    map_obs[12][enemy_cnt > 0] = enemy_hp_sum[enemy_cnt > 0] / enemy_cnt[enemy_cnt > 0]

    # Extraer primera acción macro y primera acción micro en la ventana
    macro_id = 0
    micro_id = 0
    macro_xy = GRID_W * (GRID_H // 2) + (GRID_W // 2)
    micro_xy = macro_xy

    act_line_re = re.compile(r"-\s*\+[\d.]+s:\s*([^\-]+?)\s*-\s*([^\-]+?)(?:\s*-?\s*(.*))?$")
    coord_re = re.compile(r"\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)")

    for line in act_part.splitlines():
        line = line.strip()
        if not line.startswith("- +"):
            continue
        m = act_line_re.match(line)
        if not m:
            continue
        subj, cmd, targ = m.group(1).lower(), m.group(2).strip(), (m.group(3) or "")
        is_w = "scv" in subj
        m_id, mi_id = map_ability_name_to_v2(cmd, target_is_worker=is_w)
        cm = coord_re.search(targ)
        gx, gy = GRID_W // 2, GRID_H // 2
        if cm:
            gx = int(np.clip((float(cm.group(1)) / eff_map) * GRID_W, 0, GRID_W - 1))
            gy = int(np.clip((float(cm.group(2)) / eff_map) * GRID_H, 0, GRID_H - 1))
        flat_xy = int(gy * GRID_W + gx)
        if m_id is not None and macro_id == 0:
            macro_id = m_id
            macro_xy = flat_xy
        if mi_id is not None and micro_id == 0:
            micro_id = mi_id
            micro_xy = flat_xy
        if macro_id != 0 and micro_id != 0:
            break

    # Máscara coherente con el estado observado + acción etiquetada garantizada válida
    macro_mask = np.zeros(NUM_MACRO, dtype=bool)
    micro_mask = np.zeros(NUM_MICRO, dtype=bool)
    macro_mask[0] = True
    micro_mask[0] = True
    macro_mask[1:4] = True
    if minerals >= 100:
        macro_mask[4] = True
    if minerals >= 75:
        macro_mask[5] = True
    if minerals >= 400:
        macro_mask[6] = True
    if minerals >= 150:
        macro_mask[7] = True
    if has_rax and minerals >= 150 and gas >= 100:
        macro_mask[8] = True
    if has_fac and minerals >= 150 and gas >= 100:
        macro_mask[9] = True
    if has_cc and minerals >= 125:
        macro_mask[10] = True
    if has_fac and minerals >= 150 and gas >= 100:
        macro_mask[11] = True
    if has_rax:
        macro_mask[14:21] = True
    if has_fac:
        macro_mask[21:24] = True
    if has_sp:
        macro_mask[24:26] = True
    if has_tl_rax or has_ebay or has_armory or has_tl_fac:
        macro_mask[26:28] = True
    macro_mask[macro_id] = True

    micro_mask[1] = True
    if army_sup > 0:
        micro_mask[3:7] = True
    micro_mask[micro_id] = True

    return {
        "map_obs": map_obs,
        "vec_obs": vec_obs,
        "macro_mask": macro_mask,
        "micro_mask": micro_mask,
        "macro_id": np.int64(macro_id),
        "micro_id": np.int64(micro_id),
        "macro_xy": np.int64(macro_xy),
        "micro_xy": np.int64(micro_xy),
    }


def download_sc2_dynamics_terran_v2(
    out_npz: str = "data/sc2_v2_bc_dataset.npz",
    max_samples: int = 15000,
    cache_dir: str = "./data_cache",
) -> str:
    from huggingface_hub import hf_hub_download

    os.makedirs(cache_dir, exist_ok=True)
    items = []
    for split_file in ("wm_train_horizon5.json", "wm_val_horizon5.json"):
        path = hf_hub_download(
            repo_id="yxzhang2024/SC2-Dynamics-50K",
            filename=split_file,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        for rec in records:
            msgs = rec.get("messages", [])
            if not msgs:
                continue
            parsed = parse_sc2_dynamics_terran_v2(msgs[0].get("content", ""))
            if parsed is not None:
                items.append(parsed)
                if len(items) >= max_samples:
                    break
        if len(items) >= max_samples:
            break

    merged = {
        "map_obs": np.stack([x["map_obs"] for x in items], axis=0).astype(np.float32),
        "vec_obs": np.stack([x["vec_obs"] for x in items], axis=0).astype(np.float32),
        "macro_mask": np.stack([x["macro_mask"] for x in items], axis=0).astype(bool),
        "micro_mask": np.stack([x["micro_mask"] for x in items], axis=0).astype(bool),
        "macro_id": np.array([x["macro_id"] for x in items], dtype=np.int64),
        "micro_id": np.array([x["micro_id"] for x in items], dtype=np.int64),
        "macro_xy": np.array([x["macro_xy"] for x in items], dtype=np.int64),
        "micro_xy": np.array([x["micro_xy"] for x in items], dtype=np.int64),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    np.savez_compressed(out_npz, **merged)
    print(f"[HF -> V2] Guardadas {len(items)} muestras Terran en: {out_npz}")
    return out_npz


# ============================================================================
# 4. Datasets PyTorch V2 (Imitación BC y Rollouts PPO)
# ============================================================================

class SC2V2BCDataset(Dataset):
    def __init__(self, data_dict: Dict[str, np.ndarray]):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch es obligatorio para usar SC2V2BCDataset.")
        self.map_obs = data_dict["map_obs"].astype(np.float32)
        self.vec_obs = data_dict["vec_obs"].astype(np.float32)
        self.macro_mask = data_dict["macro_mask"].astype(bool)
        self.micro_mask = data_dict["micro_mask"].astype(bool)
        self.macro_id = data_dict["macro_id"].astype(np.int64)
        self.micro_id = data_dict["micro_id"].astype(np.int64)
        self.macro_xy = data_dict["macro_xy"].astype(np.int64)
        self.micro_xy = data_dict["micro_xy"].astype(np.int64)

    def __len__(self) -> int:
        return len(self.macro_id)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.map_obs[idx]),
            torch.from_numpy(self.vec_obs[idx]),
            torch.from_numpy(self.macro_mask[idx]),
            torch.from_numpy(self.micro_mask[idx]),
            torch.tensor(self.macro_id[idx], dtype=torch.long),
            torch.tensor(self.micro_id[idx], dtype=torch.long),
            torch.tensor(self.macro_xy[idx], dtype=torch.long),
            torch.tensor(self.micro_xy[idx], dtype=torch.long),
        )

    def build_sampler(self) -> WeightedRandomSampler:
        pair_counts = Counter(zip(self.macro_id.tolist(), self.micro_id.tolist()))
        weights = np.array(
            [1.0 / np.sqrt(max(1, pair_counts[(int(a), int(b))])) for a, b in zip(self.macro_id, self.micro_id)],
            dtype=np.float64,
        )
        weights /= weights.mean()
        return WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)


def load_bc_splits(npz_path: str, val_ratio: float = 0.15, seed: int = 42) -> Tuple[SC2V2BCDataset, SC2V2BCDataset]:
    raw = np.load(npz_path)
    n = len(raw["macro_id"])
    perm = np.random.default_rng(seed).permutation(n)
    split = int(n * (1.0 - val_ratio))
    tr_idx, vl_idx = perm[:split], perm[split:]
    keys = ["map_obs", "vec_obs", "macro_mask", "micro_mask", "macro_id", "micro_id", "macro_xy", "micro_xy"]
    tr_dict = {k: raw[k][tr_idx] for k in keys}
    vl_dict = {k: raw[k][vl_idx] for k in keys}
    return SC2V2BCDataset(tr_dict), SC2V2BCDataset(vl_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extractor y conversor de dataset V2 Terran")
    parser.add_argument("--extract-replays", type=str, default="", help="Carpeta con archivos .SC2Replay")
    parser.add_argument("--from-hf", action="store_true", help="Descargar y convertir Terran de SC2-Dynamics-50K")
    parser.add_argument("--out", type=str, default="data/sc2_v2_bc_dataset.npz")
    parser.add_argument("--max-samples", type=int, default=15000)
    args = parser.parse_args()

    if args.extract_replays:
        extract_replays_from_folder(args.extract_replays, out_npz=args.out)
    elif args.from_hf:
        download_sc2_dynamics_terran_v2(out_npz=args.out, max_samples=args.max_samples)
    else:
        parser.print_help()
