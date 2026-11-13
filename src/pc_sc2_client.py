"""
Cliente StarCraft II V2 (Terran, Lazo Cerrado K=1, Dual-Head Macro+Micro, Action Masking y PPO).
Modos soportados:
  1. Benchmark LAN (--benchmark): Verifica conectividad y latencia binaria con RPi5 + Hailo-8.
  2. Jugar en vivo (--play [--human]): Control 100% neuronal por el Hailo-8 (o fallback ONNX).
  3. Recolección BC Teacher (--collect-teacher): Genera demostraciones limpias Terran sin domain-shift.
  4. Recolección RL PPO (--collect-rl): Muestreo estocástico con Action Masking y log_probs exactos.
"""

import argparse
import asyncio
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .actions import TerranActionExecutorV2
from .config import (
    GLOBAL_DIM,
    GRID_H,
    GRID_W,
    ID_TO_MACRO,
    ID_TO_MICRO,
    MACRO_TO_ID,
    MAP_CHANNELS,
    MICRO_TO_ID,
    NUM_MACRO,
    NUM_MICRO,
    PROTOCOL_MAGIC,
    SPATIAL_CELLS,
    SPATIAL_MACRO_IDS,
    SPATIAL_MICRO_IDS,
    V2Config,
)
from .features import EpisodeTrackerV2, extract_state_and_masks, get_playable_bounds, world_to_grid

RESP_HEADER_BYTES = 12  # !IIf
RESP_BODY_FLOATS = NUM_MACRO + NUM_MICRO + 2 * SPATIAL_CELLS  # 28 + 12 + 2048 = 2088 floats
RESP_TOTAL_BYTES = RESP_HEADER_BYTES + RESP_BODY_FLOATS * 4   # 8,364 bytes


class SC2V2LANClient:
    """Cliente TCP binario de ultra-baja latencia hacia el servidor RPi5 + Hailo-8."""

    def __init__(self, rpi_ip: str = "192.168.50.7", port: int = 5555, timeout: float = 15.0):
        self.rpi_ip = rpi_ip
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self._connect()

    def _connect(self):
        self.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.rpi_ip, self.port))
        print(f"[PC V2] Conectado a servidor Hailo-8 en {self.rpi_ip}:{self.port}")

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Conexión cerrada por el servidor RPi.")
            buf.extend(chunk)
        return bytes(buf)

    def request_logits(
        self,
        map_obs: np.ndarray,
        vec_obs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        assert map_obs.shape == (1, MAP_CHANNELS, GRID_H, GRID_W)
        assert vec_obs.shape == (1, GLOBAL_DIM, 1, 1)

        payload = (
            struct.pack("!I", PROTOCOL_MAGIC)
            + np.ascontiguousarray(map_obs, dtype=np.float32).tobytes()
            + np.ascontiguousarray(vec_obs, dtype=np.float32).tobytes()
        )

        for attempt in range(2):
            try:
                if self.sock is None:
                    self._connect()
                self.sock.sendall(payload)
                resp = self._recv_exact(RESP_TOTAL_BYTES)
                magic, backend_code, npu_ms = struct.unpack("!IIf", resp[:RESP_HEADER_BYTES])
                if magic != PROTOCOL_MAGIC:
                    raise ValueError(f"Magic inválido en respuesta: {hex(magic)}")
                floats = np.frombuffer(resp, dtype=np.float32, offset=RESP_HEADER_BYTES)
                macro_l = floats[:NUM_MACRO].copy()
                micro_l = floats[NUM_MACRO : NUM_MACRO + NUM_MICRO].copy()
                spatial_l = floats[NUM_MACRO + NUM_MICRO :].reshape(2, GRID_H, GRID_W).copy()
                return macro_l, micro_l, spatial_l, float(npu_ms), int(backend_code)
            except (socket.timeout, TimeoutError, ConnectionError, OSError, ValueError) as exc:
                print(f"[LAN Retry {attempt+1}/2] Error de red ({exc}), reconectando...")
                try:
                    self._connect()
                except Exception:
                    time.sleep(0.4)

        return (
            np.zeros(NUM_MACRO, dtype=np.float32),
            np.zeros(NUM_MICRO, dtype=np.float32),
            np.zeros((2, GRID_H, GRID_W), dtype=np.float32),
            0.0,
            0,
        )

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None


def _masked_softmax_sample(
    logits: np.ndarray,
    mask: Optional[np.ndarray],
    rng: np.random.Generator,
    temperature: float = 1.0,
    greedy: bool = False,
) -> Tuple[int, float]:
    """Aplica máscara, softmax estable y retorna (acción_elegida, log_prob bajo temp=1.0 para PPO)."""
    masked = logits.astype(np.float64).copy()
    if mask is not None:
        masked[~mask] = -1e9

    # log_prob exacto bajo la política sin temperatura (pi_theta) para que PPO sea insesgado
    shifted_1 = masked - np.max(masked)
    log_z_1 = np.log(np.sum(np.exp(shifted_1)))
    log_probs_1 = shifted_1 - log_z_1

    if greedy or temperature <= 1e-4:
        act = int(np.argmax(masked))
        return act, float(log_probs_1[act])

    scaled = masked / float(temperature)
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    probs /= np.sum(probs)
    act = int(rng.choice(len(logits), p=probs))
    return act, float(log_probs_1[act])


# ============================================================================
# Bloqueo atómico de archivos para recolección paralela multi-worker
# ============================================================================

class InterprocessFileLock:
    def __init__(self, lock_path: str, timeout: float = 60.0):
        self.lock_path = lock_path
        self.timeout = timeout
        self.fd = None

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                self.fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                return self
            except FileExistsError:
                if time.time() - t0 > self.timeout:
                    try:
                        if time.time() - os.path.getmtime(self.lock_path) > 120.0:
                            os.remove(self.lock_path)
                    except OSError:
                        pass
                time.sleep(0.15)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
        try:
            os.remove(self.lock_path)
        except OSError:
            pass


def append_v2_rollouts(ep_data: Dict[str, np.ndarray], ep_won: int, rollout_path: str, max_steps: int = 30000) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(rollout_path)), exist_ok=True)
    lock_file = rollout_path + ".lock"
    keys = [
        "map_obs", "vec_obs", "macro_mask", "micro_mask",
        "macro_id", "micro_id", "macro_xy", "micro_xy",
        "old_lp_macro", "old_lp_micro", "old_lp_s_macro", "old_lp_s_micro",
        "rewards", "dones",
    ]
    with InterprocessFileLock(lock_file):
        if os.path.exists(rollout_path):
            old = np.load(rollout_path)
            merged = {k: np.concatenate([old[k], ep_data[k]], axis=0) for k in keys}
            ep_wins = np.concatenate([old["ep_wins"], np.array([ep_won], dtype=np.int64)], axis=0)
        else:
            merged = {k: ep_data[k] for k in keys}
            ep_wins = np.array([ep_won], dtype=np.int64)

        if len(merged["rewards"]) > max_steps:
            merged = {k: v[-max_steps:] for k, v in merged.items()}
        if len(ep_wins) > 500:
            ep_wins = ep_wins[-500:]

        np.savez_compressed(rollout_path, ep_wins=ep_wins, **merged)
        return int(len(merged["rewards"]))


def append_bc_teacher_data(ep_data: Dict[str, np.ndarray], dataset_path: str, max_samples: int = 40000) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(dataset_path)), exist_ok=True)
    lock_file = dataset_path + ".lock"
    keys = ["map_obs", "vec_obs", "macro_mask", "micro_mask", "macro_id", "micro_id", "macro_xy", "micro_xy"]
    with InterprocessFileLock(lock_file):
        if os.path.exists(dataset_path):
            old = np.load(dataset_path)
            merged = {k: np.concatenate([old[k], ep_data[k]], axis=0) for k in keys}
        else:
            merged = {k: ep_data[k] for k in keys}
        if len(merged["macro_id"]) > max_samples:
            merged = {k: v[-max_samples:] for k, v in merged.items()}
        np.savez_compressed(dataset_path, **merged)
        return int(len(merged["macro_id"]))


# ============================================================================
# Política Teacher Terran para generar datos BC sin domain-shift (--collect-teacher)
# ============================================================================

def choose_terran_teacher_actions(
    bot,
    macro_mask: np.ndarray,
    micro_mask: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[int, int, int, int]:
    """
    Maestro heurístico limpio sobre el espacio de acciones V2 para preentrenar (BC)
    un build order sólido de Bio-Tank-Medivac sin domain-shift.
    """
    from sc2.ids.unit_typeid import UnitTypeId
    from sc2.ids.upgrade_id import UpgradeId

    bounds = get_playable_bounds(bot.game_info)
    main_pos = bot.townhalls.first.position if bot.townhalls else bot.start_location
    mgx, mgy = world_to_grid(main_pos, bounds)
    nat_pos = min(bot.expansion_locations_list, key=lambda p: p.distance_to(bot.start_location)) if bot.expansion_locations_list else main_pos
    ngx, ngy = world_to_grid(nat_pos, bounds)

    def near_cell(cx: int, cy: int, rad: int = 2) -> Tuple[int, int]:
        return (
            int(np.clip(cx + int(rng.integers(-rad, rad + 1)), 1, GRID_W - 2)),
            int(np.clip(cy + int(rng.integers(-rad, rad + 1)), 1, GRID_H - 2)),
        )

    macro_id = MACRO_TO_ID["MACRO_NO_OP"]
    macro_gx, macro_gy = near_cell(mgx, mgy, 1)

    n_bases = bot.townhalls.amount
    n_rax = bot.structures(UnitTypeId.BARRACKS).amount + bot.already_pending(UnitTypeId.BARRACKS)
    n_fac = bot.structures(UnitTypeId.FACTORY).amount + bot.already_pending(UnitTypeId.FACTORY)
    n_sp = bot.structures(UnitTypeId.STARPORT).amount + bot.already_pending(UnitTypeId.STARPORT)
    n_ebay = bot.structures(UnitTypeId.ENGINEERINGBAY).amount + bot.already_pending(UnitTypeId.ENGINEERINGBAY)
    n_gas = bot.gas_buildings.amount + bot.already_pending(UnitTypeId.REFINERY)

    if macro_mask[MACRO_TO_ID["ASSIGN_MINERALS"]] and bot.idle_worker_count > 0:
        macro_id = MACRO_TO_ID["ASSIGN_MINERALS"]
    elif macro_mask[MACRO_TO_ID["BUILD_SUPPLY_DEPOT"]] and bot.supply_left <= (4 if bot.supply_cap < 40 else 7) and not bot.already_pending(UnitTypeId.SUPPLYDEPOT):
        macro_id = MACRO_TO_ID["BUILD_SUPPLY_DEPOT"]
        macro_gx, macro_gy = near_cell(mgx, mgy, 2)
    elif macro_mask[MACRO_TO_ID["UPGRADE_ORBITAL_COMMAND"]]:
        macro_id = MACRO_TO_ID["UPGRADE_ORBITAL_COMMAND"]
        macro_gx, macro_gy = mgx, mgy
    elif macro_mask[MACRO_TO_ID["CALLDOWN_MULE"]]:
        macro_id = MACRO_TO_ID["CALLDOWN_MULE"]
        macro_gx, macro_gy = mgx, mgy
    elif macro_mask[MACRO_TO_ID["ASSIGN_GAS"]]:
        macro_id = MACRO_TO_ID["ASSIGN_GAS"]
    elif macro_mask[MACRO_TO_ID["TRAIN_SCV"]] and bot.supply_workers < min(66, n_bases * 22):
        macro_id = MACRO_TO_ID["TRAIN_SCV"]
    elif macro_mask[MACRO_TO_ID["BUILD_COMMAND_CENTER"]] and n_bases < (2 if bot.time < 420 else 3) and not bot.already_pending(UnitTypeId.COMMANDCENTER):
        macro_id = MACRO_TO_ID["BUILD_COMMAND_CENTER"]
        macro_gx, macro_gy = ngx, ngy
    elif macro_mask[MACRO_TO_ID["BUILD_BARRACKS"]] and n_rax < min(5, 1 + n_bases * 2):
        macro_id = MACRO_TO_ID["BUILD_BARRACKS"]
        macro_gx, macro_gy = near_cell(mgx, mgy, 2)
    elif macro_mask[MACRO_TO_ID["BUILD_REFINERY"]] and n_gas < n_bases * 2 and n_rax >= 1:
        macro_id = MACRO_TO_ID["BUILD_REFINERY"]
        macro_gx, macro_gy = mgx, mgy
    elif macro_mask[MACRO_TO_ID["RESEARCH_BIO_TECH"]]:
        macro_id = MACRO_TO_ID["RESEARCH_BIO_TECH"]
    elif macro_mask[MACRO_TO_ID["RESEARCH_UPGRADES"]]:
        macro_id = MACRO_TO_ID["RESEARCH_UPGRADES"]
    elif macro_mask[MACRO_TO_ID["BUILD_ADDON_TECHLAB"]] and bot.structures(UnitTypeId.BARRACKSTECHLAB).amount < 2:
        macro_id = MACRO_TO_ID["BUILD_ADDON_TECHLAB"]
        macro_gx, macro_gy = mgx, mgy
    elif macro_mask[MACRO_TO_ID["BUILD_ADDON_REACTOR"]] and bot.structures(UnitTypeId.BARRACKSREACTOR).amount < 2:
        macro_id = MACRO_TO_ID["BUILD_ADDON_REACTOR"]
        macro_gx, macro_gy = mgx, mgy
    elif macro_mask[MACRO_TO_ID["BUILD_FACTORY"]] and n_fac < 1:
        macro_id = MACRO_TO_ID["BUILD_FACTORY"]
        macro_gx, macro_gy = near_cell(mgx, mgy, 2)
    elif macro_mask[MACRO_TO_ID["BUILD_STARPORT"]] and n_sp < 1:
        macro_id = MACRO_TO_ID["BUILD_STARPORT"]
        macro_gx, macro_gy = near_cell(mgx, mgy, 2)
    elif macro_mask[MACRO_TO_ID["BUILD_ENGINEERING_BAY"]] and n_ebay < 1 and n_bases >= 2:
        macro_id = MACRO_TO_ID["BUILD_ENGINEERING_BAY"]
        macro_gx, macro_gy = near_cell(mgx, mgy, 2)
    elif macro_mask[MACRO_TO_ID["TRAIN_MEDIVAC"]] and bot.units(UnitTypeId.MEDIVAC).amount < 5:
        macro_id = MACRO_TO_ID["TRAIN_MEDIVAC"]
    elif macro_mask[MACRO_TO_ID["TRAIN_SIEGE_TANK"]] and bot.units({UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED}).amount < 6:
        macro_id = MACRO_TO_ID["TRAIN_SIEGE_TANK"]
    elif macro_mask[MACRO_TO_ID["TRAIN_MARAUDER"]] and bot.units(UnitTypeId.MARAUDER).amount * 2 < bot.units(UnitTypeId.MARINE).amount:
        macro_id = MACRO_TO_ID["TRAIN_MARAUDER"]
    elif macro_mask[MACRO_TO_ID["TRAIN_MARINE"]]:
        macro_id = MACRO_TO_ID["TRAIN_MARINE"]

    # Micro Teacher
    micro_id = MICRO_TO_ID["MICRO_NO_OP"]
    micro_gx, micro_gy = mgx, mgy
    enemy_spawn = bot.enemy_start_locations[0] if bot.enemy_start_locations else main_pos
    egx, egy = world_to_grid(enemy_spawn, bounds)

    base_threats = [
        eu for eu in bot.enemy_units
        if any(s.distance_to(eu) <= 22.0 for s in bot.structures)
    ] if bot.enemy_units else []

    if micro_mask[MICRO_TO_ID["USE_STIMPACK"]]:
        micro_id = MICRO_TO_ID["USE_STIMPACK"]
    elif base_threats and micro_mask[MICRO_TO_ID["ATTACK_MOVE_ARMY"]]:
        micro_id = MICRO_TO_ID["ATTACK_MOVE_ARMY"]
        micro_gx, micro_gy = world_to_grid(base_threats[0].position, bounds)
    elif micro_mask[MICRO_TO_ID["SIEGE_TANKS_AT_AREA"]] and bot.enemy_units:
        tanks = bot.units(UnitTypeId.SIEGETANK)
        if tanks and any(bot.enemy_units.closer_than(13.0, t) for t in tanks):
            micro_id = MICRO_TO_ID["SIEGE_TANKS_AT_AREA"]
            micro_gx, micro_gy = world_to_grid(tanks.first.position, bounds)
    elif micro_mask[MICRO_TO_ID["UNSIEGE_TANKS_TO_MOVE"]]:
        sieged = bot.units(UnitTypeId.SIEGETANKSIEGED)
        if sieged and (not bot.enemy_units or all(not bot.enemy_units.closer_than(15.0, s) for s in sieged)):
            micro_id = MICRO_TO_ID["UNSIEGE_TANKS_TO_MOVE"]
            micro_gx, micro_gy = world_to_grid(sieged.first.position, bounds)
    elif micro_mask[MICRO_TO_ID["ATTACK_MOVE_ARMY"]] and (bot.supply_army >= 26 or (bot.enemy_units and bot.supply_army >= 12)):
        targets = bot.enemy_units + bot.enemy_structures
        if targets:
            t_pos = targets.closest_to(main_pos).position
            micro_gx, micro_gy = world_to_grid(t_pos, bounds)
        else:
            micro_gx, micro_gy = egx, egy
        micro_id = MICRO_TO_ID["ATTACK_MOVE_ARMY"]
    elif micro_mask[MICRO_TO_ID["SCOUT_WITH_SCV_OR_REAPER"]] and 85.0 < bot.time < 125.0:
        micro_id = MICRO_TO_ID["SCOUT_WITH_SCV_OR_REAPER"]
        micro_gx, micro_gy = egx, egy
    elif micro_mask[MICRO_TO_ID["REPAIR_WITH_SCV"]] and rng.random() < 0.25:
        micro_id = MICRO_TO_ID["REPAIR_WITH_SCV"]
        micro_gx, micro_gy = mgx, mgy
    elif micro_mask[MICRO_TO_ID["MOVE_ARMY"]] and bot.supply_army >= 4:
        micro_id = MICRO_TO_ID["MOVE_ARMY"]
        micro_gx, micro_gy = ngx, ngy

    return (
        macro_id,
        int(macro_gy * GRID_W + macro_gx),
        micro_id,
        int(micro_gy * GRID_W + micro_gx),
    )


# ============================================================================
# Bot en vivo V2 y Bucle de Partida
# ============================================================================

def launch_live_sc2_match(
    rpi_ip: str = "192.168.50.7",
    port: int = 5555,
    map_name: str = "Simple64",
    enemy_race: str = "Terran",
    difficulty: str = "Easy",
    realtime: bool = False,
    max_game_steps: int = 0,
    human_vs_ai: bool = False,
    collect_rl: bool = False,
    collect_teacher: bool = False,
    temperature: float = 1.0,
    greedy: bool = False,
    rollout_path: str = "models/sc2_v2_rollouts.npz",
    bc_dataset_path: str = "data/sc2_v2_bc_dataset.npz",
    window_x: int = 50,
    window_y: int = 50,
    window_w: int = 960,
    window_h: int = 720,
) -> Dict[str, object]:
    from sc2 import maps
    from sc2.bot_ai import BotAI
    from sc2.data import Difficulty, Race, Result
    from sc2.ids.unit_typeid import UnitTypeId
    from sc2.main import run_game
    from sc2.player import Bot, Computer, Human

    race_map = {"Terran": Race.Terran, "Protoss": Race.Protoss, "Zerg": Race.Zerg, "Random": Race.Random}
    diff_map = {
        "VeryEasy": Difficulty.VeryEasy,
        "Easy": Difficulty.Easy,
        "Medium": Difficulty.Medium,
        "Hard": Difficulty.Hard,
        "VeryHard": Difficulty.VeryHard,
    }

    summary: Dict[str, object] = {"won": 0, "result_str": "Unknown", "steps_recorded": 0, "total_buffer": 0}

    class Hailo8TerranBotV2(BotAI):
        def __init__(self):
            super().__init__()
            self.lan_client: Optional[SC2V2LANClient] = None
            self.executor = TerranActionExecutorV2()
            self.tracker = EpisodeTrackerV2()
            self.terrain_cache = None
            self.rng = np.random.default_rng()
            self.reached_step_limit = False
            self.last_npu_ms = 0.0
            self.last_rtt_ms = 0.0
            self.last_macro_name = "INIT"
            self.last_micro_name = "INIT"

            # Buffers de episodio
            self.buf_map: List[np.ndarray] = []
            self.buf_vec: List[np.ndarray] = []
            self.buf_mmask: List[np.ndarray] = []
            self.buf_mimask: List[np.ndarray] = []
            self.buf_mid: List[int] = []
            self.buf_miid: List[int] = []
            self.buf_mxy: List[int] = []
            self.buf_mixy: List[int] = []
            self.buf_lpm: List[float] = []
            self.buf_lpmi: List[float] = []
            self.buf_lpsm: List[float] = []
            self.buf_lpsmi: List[float] = []
            self.buf_rew: List[float] = []

        async def on_unit_destroyed(self, unit_tag: int):
            lost = self._structures_previous_map.get(unit_tag) or self._units_previous_map.get(unit_tag)
            if lost is None:
                return
            try:
                val = self.calculate_unit_value(lost.type_id)
                self.tracker.lost_value_total += float(val.minerals + val.vespene)
            except Exception:
                self.tracker.lost_value_total += 50.0

        async def on_start(self):
            self.client.game_step = 8 if ((collect_rl or collect_teacher) and not realtime) else 4
            if not collect_teacher:
                self.lan_client = SC2V2LANClient(rpi_ip=rpi_ip, port=port)

        def _compute_pbrs_potential(self) -> float:
            score = self.state.score
            killed = float(score.killed_value_units + score.killed_value_structures)
            lost = self.tracker.lost_value_total
            army_val = float(score.total_value_units)
            workers = min(66.0, float(self.supply_workers))
            bases = float(self.townhalls.amount)
            upgs = float(len(self.state.upgrades))
            return (
                (killed - lost) / 400.0
                + army_val / 600.0
                + 0.05 * workers
                + 0.60 * bases
                + 0.35 * upgs
            )

        def _compute_step_reward(self) -> float:
            phi = self._compute_pbrs_potential()
            if self.tracker.prev_phi is None:
                self.tracker.prev_phi = phi
                return 0.0
            r = phi - self.tracker.prev_phi
            self.tracker.prev_phi = phi
            if self.supply_left <= 0 and self.supply_cap < 200 and self.minerals >= 100:
                r -= 0.02
            if self.minerals > 900 and self.supply_used < 185:
                r -= 0.015
            return float(np.clip(r, -1.5, 1.5))

        async def on_step(self, iteration: int):
            if max_game_steps > 0 and iteration >= max_game_steps:
                self.reached_step_limit = True
                await self.client.leave()
                return

            if (collect_rl or collect_teacher) and self.buf_map:
                self.buf_rew.append(self._compute_step_reward())

            map_obs, vec_obs, m_mask, mi_mask, self.terrain_cache = extract_state_and_masks(
                self, tracker=self.tracker, terrain_cache=self.terrain_cache
            )

            if collect_teacher:
                m_id, m_xy, mi_id, mi_xy = choose_terran_teacher_actions(self, m_mask, mi_mask, self.rng)
                lp_m = lp_mi = lp_sm = lp_smi = 0.0
            else:
                t0 = time.perf_counter()
                m_log, mi_log, sp_log, npu_ms, _ = await asyncio.to_thread(
                    self.lan_client.request_logits, map_obs, vec_obs
                )
                self.last_rtt_ms = (time.perf_counter() - t0) * 1000.0
                self.last_npu_ms = npu_ms

                eff_temp = temperature if collect_rl else (0.0 if greedy else 0.25)
                use_greedy = greedy or (not collect_rl and eff_temp <= 0.05)

                m_id, lp_m = _masked_softmax_sample(m_log, m_mask, self.rng, temperature=eff_temp, greedy=use_greedy)
                mi_id, lp_mi = _masked_softmax_sample(mi_log, mi_mask, self.rng, temperature=eff_temp, greedy=use_greedy)
                m_xy, lp_sm = _masked_softmax_sample(sp_log[0].reshape(-1), None, self.rng, temperature=eff_temp, greedy=use_greedy)
                mi_xy, lp_smi = _masked_softmax_sample(sp_log[1].reshape(-1), None, self.rng, temperature=eff_temp, greedy=use_greedy)

            mgx, mgy = m_xy % GRID_W, m_xy // GRID_W
            migx, migy = mi_xy % GRID_W, mi_xy // GRID_W
            self.last_macro_name = ID_TO_MACRO.get(m_id, "MACRO_NO_OP")
            self.last_micro_name = ID_TO_MICRO.get(mi_id, "MICRO_NO_OP")

            if collect_rl or collect_teacher:
                self.buf_map.append(map_obs[0])
                self.buf_vec.append(vec_obs[0])
                self.buf_mmask.append(m_mask)
                self.buf_mimask.append(mi_mask)
                self.buf_mid.append(m_id)
                self.buf_miid.append(mi_id)
                self.buf_mxy.append(m_xy)
                self.buf_mixy.append(mi_xy)
                self.buf_lpm.append(lp_m)
                self.buf_lpmi.append(lp_mi)
                self.buf_lpsm.append(lp_sm if m_id in SPATIAL_MACRO_IDS else 0.0)
                self.buf_lpsmi.append(lp_smi if mi_id in SPATIAL_MICRO_IDS else 0.0)

            try:
                await self.executor.execute_step(self, m_id, mgx, mgy, mi_id, migx, migy)
            except Exception as exc:
                print(f"[Aviso V2] Error ejecutando ({self.last_macro_name}, {self.last_micro_name}): {exc}")

            self.tracker.update_action(m_id, mi_id)

            if not human_vs_ai and not collect_rl and not collect_teacher:
                hud = (
                    f"Hailo-8 V2 (K=1 Dual) | NPU: {self.last_npu_ms:.2f}ms | RTT: {self.last_rtt_ms:.1f}ms\n"
                    f"Macro: {self.last_macro_name} ({mgx},{mgy}) | Micro: {self.last_micro_name} ({migx},{migy})\n"
                    f"Min: {self.minerals} | Gas: {self.vespene} | Sup: {self.supply_used}/{self.supply_cap}"
                )
                self.client.debug_text_screen(hud, pos=(0.01, 0.02), size=14)

            if iteration % 80 == 0:
                print(
                    f"[SC2 V2 t={self.time_formatted}] NPU={self.last_npu_ms:.2f}ms | "
                    f"Sup={self.supply_used}/{self.supply_cap} | Mac={self.last_macro_name} | Mic={self.last_micro_name}"
                )

        async def on_end(self, game_result):
            won = 1 if game_result == Result.Victory else 0
            result_tag = "Result.Timeout" if (self.reached_step_limit and not won) else str(game_result)
            print(f"[SC2 V2 Fin] Resultado: {result_tag} | Tiempo: {self.time_formatted}")

            if collect_rl and self.buf_map:
                while len(self.buf_rew) < len(self.buf_map):
                    self.buf_rew.append(self._compute_step_reward())
                term = 2.0 if won else (-0.3 if self.reached_step_limit else -2.0)
                self.buf_rew[-1] += term

                dones = np.zeros(len(self.buf_map), dtype=bool)
                dones[-1] = True
                ep_data = {
                    "map_obs": np.stack(self.buf_map, axis=0).astype(np.float32),
                    "vec_obs": np.stack(self.buf_vec, axis=0).astype(np.float32),
                    "macro_mask": np.stack(self.buf_mmask, axis=0).astype(bool),
                    "micro_mask": np.stack(self.buf_mimask, axis=0).astype(bool),
                    "macro_id": np.array(self.buf_mid, dtype=np.int64),
                    "micro_id": np.array(self.buf_miid, dtype=np.int64),
                    "macro_xy": np.array(self.buf_mxy, dtype=np.int64),
                    "micro_xy": np.array(self.buf_mixy, dtype=np.int64),
                    "old_lp_macro": np.array(self.buf_lpm, dtype=np.float32),
                    "old_lp_micro": np.array(self.buf_lpmi, dtype=np.float32),
                    "old_lp_s_macro": np.array(self.buf_lpsm, dtype=np.float32),
                    "old_lp_s_micro": np.array(self.buf_lpsmi, dtype=np.float32),
                    "rewards": np.array(self.buf_rew, dtype=np.float32),
                    "dones": dones,
                }
                total_buf = append_v2_rollouts(ep_data, ep_won=won, rollout_path=rollout_path)
                summary["total_buffer"] = total_buf
                summary["steps_recorded"] = len(self.buf_map)

            elif collect_teacher and self.buf_map and (won == 1 or len(self.buf_map) > 40):
                ep_data = {
                    "map_obs": np.stack(self.buf_map, axis=0).astype(np.float32),
                    "vec_obs": np.stack(self.buf_vec, axis=0).astype(np.float32),
                    "macro_mask": np.stack(self.buf_mmask, axis=0).astype(bool),
                    "micro_mask": np.stack(self.buf_mimask, axis=0).astype(bool),
                    "macro_id": np.array(self.buf_mid, dtype=np.int64),
                    "micro_id": np.array(self.buf_miid, dtype=np.int64),
                    "macro_xy": np.array(self.buf_mxy, dtype=np.int64),
                    "micro_xy": np.array(self.buf_mixy, dtype=np.int64),
                }
                total_buf = append_bc_teacher_data(ep_data, dataset_path=bc_dataset_path)
                summary["total_buffer"] = total_buf
                summary["steps_recorded"] = len(self.buf_map)

            summary["won"] = won
            summary["result_str"] = result_tag
            if self.lan_client is not None:
                self.lan_client.close()

    import sc2.main as sc2_main
    orig_proc = sc2_main.SC2Process
    cnt = {"n": 0}

    def patched_proc(*args, **kwargs):
        cnt["n"] += 1
        if human_vs_ai and cnt["n"] == 1:
            kwargs.setdefault("resolution", (1280, 720))
            kwargs.setdefault("placement", (40, 40))
        else:
            kwargs.setdefault("resolution", (window_w, window_h))
            kwargs.setdefault("placement", (window_x, window_y))
        return orig_proc(*args, **kwargs)

    sc2_main.SC2Process = patched_proc
    players = (
        [Human(race_map.get(enemy_race, Race.Zerg), name="Human"), Bot(Race.Terran, Hailo8TerranBotV2(), name="Hailo8_V2")]
        if human_vs_ai
        else [Bot(Race.Terran, Hailo8TerranBotV2(), name="Hailo8_V2"), Computer(race_map.get(enemy_race, Race.Terran), diff_map.get(difficulty, Difficulty.Easy))]
    )
    run_game(maps.get(map_name), players, realtime=True if human_vs_ai else realtime)
    return summary


# ============================================================================
# Orquestador Multi-Worker (Windows SC2) con Supervisor Anti-Hang
# ============================================================================

def _cleanup_sc2(pid: Optional[int] = None):
    if os.name == "nt":
        cmd = ["taskkill", "/F", "/T", "/PID", str(pid)] if pid else ["taskkill", "/F", "/IM", "SC2_x64.exe", "/T"]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def run_collection_loop(
    mode: str,
    rpi_ip: str,
    port: int,
    difficulty: str,
    episodes: int,
    workers: int,
    max_steps: int,
    temperature: float,
    rollout_path: str,
    bc_dataset_path: str,
):
    import concurrent.futures
    import queue
    import threading

    _cleanup_sc2()
    time.sleep(1.0)
    diff_ladder = ["Easy", "Medium", "Hard", "VeryHard"]
    diff_idx = [diff_ladder.index(difficulty) if difficulty in diff_ladder else 0]
    maps_pool = ["Simple64", "AbyssalReefLE", "BelShirVestigeLE", "CactusValleyLE"]
    enemies = ["Terran", "Zerg", "Protoss"]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ep_q = queue.Queue()
    for ep in range(1, episodes + 1):
        ep_q.put(ep)

    lock = threading.Lock()
    stats = {"done": 0, "wins": 0, "recent": []}

    def worker_fn(wid: int):
        wx = 40 + (wid % 3) * 650
        wy = 40 + (wid // 3) * 500
        while not ep_q.empty():
            try:
                ep = ep_q.get_nowait()
            except queue.Empty:
                break
            with lock:
                c_diff = diff_ladder[diff_idx[0]]
            c_map = maps_pool[(ep - 1) % len(maps_pool)]
            c_enemy = enemies[(ep - 1) % len(enemies)]

            fd, sjson = tempfile.mkstemp(prefix=f"sc2v2_w{wid}_", suffix=".json")
            os.close(fd)
            cmd = [
                sys.executable, "-u", "-m", "src.pc_sc2_client",
                "--single-ep", mode,
                "--rpi-ip", rpi_ip, "--port", str(port),
                "--map", c_map, "--enemy-race", c_enemy, "--difficulty", c_diff,
                "--max-steps", str(max_steps), "--temperature", str(temperature),
                "--rollout-path", rollout_path, "--bc-dataset", bc_dataset_path,
                "--summary-json", sjson,
                "--window-x", str(wx), "--window-y", str(wy), "--window-w", "640", "--window-h", "480",
            ]
            proc = subprocess.Popen(cmd, cwd=project_root)
            res = {}
            try:
                proc.wait(timeout=240)
                if os.path.exists(sjson) and os.path.getsize(sjson) > 0:
                    with open(sjson, "r", encoding="utf-8") as f:
                        res = json.load(f)
            except subprocess.TimeoutExpired:
                _cleanup_sc2(pid=proc.pid)
            finally:
                if os.path.exists(sjson):
                    try:
                        os.remove(sjson)
                    except OSError:
                        pass
                _cleanup_sc2(pid=proc.pid)

            if res.get("steps_recorded", 0) > 0:
                with lock:
                    stats["done"] += 1
                    w = int(res.get("won", 0))
                    stats["wins"] += w
                    stats["recent"].append(w)
                    if len(stats["recent"]) > 10:
                        stats["recent"].pop(0)
                    r_wr = 100.0 * sum(stats["recent"]) / len(stats["recent"])
                    print(
                        f"[W{wid} Ep {ep:02d}/{episodes:02d}] {c_map} vs {c_enemy} ({c_diff}) | "
                        f"Win={w} | WR Reciente={r_wr:.1f}% | Muestras={res.get('total_buffer', 0)}"
                    )
                    if mode == "rl" and len(stats["recent"]) >= 6 and r_wr >= 60.0 and diff_idx[0] < len(diff_ladder) - 1:
                        diff_idx[0] += 1
                        stats["recent"].clear()
                        print(f"[Curriculum V2] Subiendo dificultad a: {diff_ladder[diff_idx[0]]}")
            ep_q.task_done()

    eff_w = max(1, min(workers, episodes))
    with concurrent.futures.ThreadPoolExecutor(max_workers=eff_w) as ex:
        futs = [ex.submit(worker_fn, i) for i in range(eff_w)]
        concurrent.futures.wait(futs)


def run_lan_benchmark(rpi_ip: str, port: int, steps: int = 10):
    client = SC2V2LANClient(rpi_ip=rpi_ip, port=port)
    rng = np.random.default_rng(42)
    for i in range(steps):
        m = rng.uniform(0.0, 1.0, size=(1, MAP_CHANNELS, GRID_H, GRID_W)).astype(np.float32)
        v = rng.uniform(0.0, 1.0, size=(1, GLOBAL_DIM, 1, 1)).astype(np.float32)
        t0 = time.perf_counter()
        ml, mil, spl, npu_ms, bcode = client.request_logits(m, v)
        rtt = (time.perf_counter() - t0) * 1000.0
        print(
            f"[Bench {i+1:02d}] RTT={rtt:.2f}ms | NPU={npu_ms:.2f}ms | Backend={bcode} | "
            f"Macro={ID_TO_MACRO[int(np.argmax(ml))]} | Micro={ID_TO_MICRO[int(np.argmax(mil))]}"
        )
    client.close()


if __name__ == "__main__":
    cfg = V2Config()
    parser = argparse.ArgumentParser(description="Cliente SC2 V2 Terran (Windows SC2 <-> WSL/RPi5)")
    parser.add_argument("--rpi-ip", default="192.168.50.7")
    parser.add_argument("--port", type=int, default=cfg.lan_port)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--human", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--collect-rl", action="store_true")
    parser.add_argument("--collect-teacher", action="store_true")
    parser.add_argument("--single-ep", default="", choices=["", "rl", "teacher"])
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=2400)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--map", default="Simple64")
    parser.add_argument("--enemy-race", default="Terran", choices=["Terran", "Protoss", "Zerg", "Random"])
    parser.add_argument("--difficulty", default="Easy", choices=["VeryEasy", "Easy", "Medium", "Hard", "VeryHard"])
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--rollout-path", default=cfg.rollout_path)
    parser.add_argument("--bc-dataset", default=cfg.bc_dataset_path)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--window-x", type=int, default=50)
    parser.add_argument("--window-y", type=int, default=50)
    parser.add_argument("--window-w", type=int, default=960)
    parser.add_argument("--window-h", type=int, default=720)
    args = parser.parse_args()

    if args.single_ep:
        res = launch_live_sc2_match(
            rpi_ip=args.rpi_ip, port=args.port, map_name=args.map, enemy_race=args.enemy_race,
            difficulty=args.difficulty, realtime=False, max_game_steps=args.max_steps,
            collect_rl=(args.single_ep == "rl"), collect_teacher=(args.single_ep == "teacher"),
            temperature=args.temperature, rollout_path=args.rollout_path, bc_dataset_path=args.bc_dataset,
            window_x=args.window_x, window_y=args.window_y, window_w=args.window_w, window_h=args.window_h,
        )
        if args.summary_json:
            with open(args.summary_json, "w", encoding="utf-8") as f:
                json.dump(res, f)
    elif args.collect_rl or args.collect_teacher:
        run_collection_loop(
            mode="rl" if args.collect_rl else "teacher",
            rpi_ip=args.rpi_ip, port=args.port, difficulty=args.difficulty,
            episodes=args.episodes, workers=args.workers, max_steps=args.max_steps,
            temperature=args.temperature, rollout_path=args.rollout_path, bc_dataset_path=args.bc_dataset,
        )
    elif args.play:
        launch_live_sc2_match(
            rpi_ip=args.rpi_ip, port=args.port, map_name=args.map, enemy_race=args.enemy_race,
            difficulty=args.difficulty, realtime=args.realtime, max_game_steps=0,
            human_vs_ai=args.human, greedy=args.greedy, temperature=args.temperature,
        )
    else:
        run_lan_benchmark(args.rpi_ip, args.port, steps=args.steps)
