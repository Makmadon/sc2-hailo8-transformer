"""
Cliente puente y Bot 100% Controlado por Red Neuronal (Sin reglas asistidas) para StarCraft II.
Soporta dos modos:
  1. Benchmark LAN: Prueba de latencia e inferencia pura hacia la Raspberry Pi + Hailo-8.
  2. Partida Real de StarCraft II (--play): Abre StarCraft II (SC2_x64.exe), extrae el tensor de observación
     [1, 12, 16, 16] en tiempo real, consulta al HAT Hailo-8 en la Raspberry Pi (192.168.50.7:5555)
     y ejecuta exclusivamente las 32 acciones del vocabulario decididas por el Transformer.
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

from .config import ACTION_CHUNK_SIZE, ACTION_TO_ID, GRID_H, GRID_W, ID_TO_ACTION, IN_CHANNELS, RACES


class SC2HailoLANClient:
    """Cliente TCP de baja latencia (TCP_NODELAY) hacia la Raspberry Pi + Hailo-8."""

    def __init__(self, rpi_ip: str = "192.168.50.7", port: int = 5555):
        self.rpi_ip = rpi_ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(5.0)
        self.sock.connect((self.rpi_ip, self.port))
        print(f"[PC Cliente SC2] Conectado al cerebro Raspberry Pi + Hailo-8 en {self.rpi_ip}:{self.port}")

    def request_action_chunk(self, obs_tensor: np.ndarray) -> Dict[str, object]:
        assert obs_tensor.shape == (1, IN_CHANNELS, GRID_H, GRID_W)
        self.sock.sendall(obs_tensor.astype(np.float32).tobytes())

        header = self._recv_exact(4)
        msg_len = struct.unpack("!I", header)[0]
        body = self._recv_exact(msg_len)
        return json.loads(body.decode("utf-8"))

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Conexión perdida con la Raspberry Pi.")
            buf += chunk
        return buf

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def run_lan_benchmark(rpi_ip: str = "192.168.50.7", port: int = 5555, steps: int = 10):
    client = SC2HailoLANClient(rpi_ip=rpi_ip, port=port)
    rng = np.random.default_rng(42)
    for i in range(steps):
        dummy_obs = rng.uniform(0.0, 1.0, size=(1, IN_CHANNELS, GRID_H, GRID_W)).astype(np.float32)
        t0 = time.perf_counter()
        response = client.request_action_chunk(dummy_obs)
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        first_act = response["chunks"][0]
        print(
            f"[Paso {i+1:02d}] RTT LAN: {rtt_ms:.2f} ms | Inferencia NPU: {response['latency_ms']} ms | "
            f"Acción[0]: {first_act['action_name']} -> Coord {first_act['target_sc2_coord']}"
        )
    client.close()


DEFAULT_DRIVE_ROLLOUT_PATH = r"G:\Mi unidad\hailo_files\sc2_rl_rollouts.npz"
LOCAL_BACKUP_ROLLOUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "sc2_rl_rollouts.npz",
)


def append_rl_rollout_to_drive(
    ep_states: np.ndarray,
    ep_actions: np.ndarray,
    ep_returns: np.ndarray,
    ep_races: np.ndarray,
    ep_won: int,
    rollout_path: str = DEFAULT_DRIVE_ROLLOUT_PATH,
) -> int:
    """
    Acumula las trayectorias (states, actions, returns, races, wins) jugadas por el bot
    directamente en Google Drive (G:\\Mi unidad\\hailo_files\\sc2_rl_rollouts.npz) y en respaldo local.
    Retorna el total acumulado de muestras RL.
    """
    target_paths = [rollout_path, LOCAL_BACKUP_ROLLOUT_PATH]
    total_samples = len(ep_states)

    for path in target_paths:
        try:
            parent = os.path.dirname(path)
            if parent and not os.path.exists(parent):
                if path == rollout_path:
                    continue
                os.makedirs(parent, exist_ok=True)

            if os.path.exists(path):
                old = np.load(path)
                states = np.concatenate([old["states"], ep_states], axis=0)
                actions = np.concatenate([old["actions"], ep_actions], axis=0)
                returns = np.concatenate([old["returns"], ep_returns], axis=0)
                races = np.concatenate([old["races"], ep_races], axis=0)
                wins = np.concatenate([old["wins"], np.array([ep_won], dtype=np.int64)], axis=0)
            else:
                states = ep_states
                actions = ep_actions
                returns = ep_returns
                races = ep_races
                wins = np.array([ep_won], dtype=np.int64)

            # Conservar las últimas 25,000 muestras más recientes (Replay Buffer deslizante)
            if len(states) > 25000:
                states = states[-25000:]
                actions = actions[-25000:]
                returns = returns[-25000:]
                races = races[-25000:]

            np.savez_compressed(
                path,
                states=states.astype(np.float32),
                actions=actions.astype(np.int64),
                returns=returns.astype(np.float32),
                races=races.astype(np.int64),
                wins=wins.astype(np.int64),
            )
            total_samples = len(states)
        except Exception as exc:
            print(f"[Aviso RL] No se pudo guardar en {path}: {exc}")

    return total_samples


def launch_live_sc2_match(
    rpi_ip: str = "192.168.50.7",
    port: int = 5555,
    map_name: str = "AbyssalReefLE",
    player_race: str = "Terran",
    enemy_race: str = "Terran",
    difficulty: str = "Medium",
    realtime: bool = False,
    max_game_steps: int = 0,
    human_vs_ai: bool = False,
    collect_rl: bool = False,
    explore_eps: float = 0.20,
    rollout_path: str = DEFAULT_DRIVE_ROLLOUT_PATH,
) -> Dict[str, object]:
    """
    Abre StarCraft II y ejecuta una partida 100% controlada por las 32 acciones del modelo en la Raspberry Pi + Hailo-8.
    Si collect_rl=True, registra estados, chunks de acciones y recompensas densas + victoria/derrota para RL (PPO).
    """
    from sc2 import maps
    from sc2.bot_ai import BotAI
    from sc2.data import Difficulty, Race, Result
    from sc2.ids.ability_id import AbilityId
    from sc2.ids.unit_typeid import UnitTypeId
    from sc2.ids.upgrade_id import UpgradeId
    from sc2.main import run_game
    from sc2.player import Bot, Computer, Human
    from sc2.position import Point2

    race_map = {
        "Terran": Race.Terran,
        "Protoss": Race.Protoss,
        "Zerg": Race.Zerg,
        "Random": Race.Random,
    }
    diff_map = {
        "VeryEasy": Difficulty.VeryEasy,
        "Easy": Difficulty.Easy,
        "Medium": Difficulty.Medium,
        "Hard": Difficulty.Hard,
        "VeryHard": Difficulty.VeryHard,
    }

    match_summary: Dict[str, object] = {
        "won": 0,
        "result_str": "Unknown",
        "steps_recorded": 0,
        "total_buffer_samples": 0,
        "episode_return": 0.0,
    }

    class Hailo8StarCraftBot(BotAI):
        def __init__(self):
            super().__init__()
            self.lan_client: Optional[SC2HailoLANClient] = None
            self.chunk_queue: List[Dict[str, object]] = []
            self.last_npu_ms: float = 0.0
            self.last_rtt_ms: float = 0.0
            self.last_action_name: str = "INIT"
            self.total_inferences: int = 0
            self.scout_worker_tag: Optional[int] = None
            self.cleared_enemy_spawns: set = set()
            self.reached_step_limit: bool = False

            # Buffers de Aprendizaje por Refuerzo (RL)
            self.rng = np.random.default_rng()
            self.rl_states: List[np.ndarray] = []
            self.rl_actions: List[np.ndarray] = []
            self.rl_rewards: List[float] = []
            self.prev_killed_val: float = 0.0
            self.prev_army_val: float = 0.0
            self.prev_workers: float = 12.0
            self.prev_bases: int = 1
            # Estructuras aliadas destruidas (acumuladas entre chunks vía on_unit_destroyed)
            self.pending_lost_structure_value: float = 0.0
            self.pending_lost_townhalls: int = 0
            self.cancelled_structure_tags: set = set()
            # Celdas 16x16 ya recorridas por el ejército en esta partida (bonus de exploración count-based)
            self.army_visited_cells = np.zeros((GRID_H, GRID_W), dtype=bool)

        async def on_unit_destroyed(self, unit_tag: int):
            """Registra el valor (minerales + gas) de cada estructura aliada destruida por el enemigo."""
            lost = self._structures_previous_map.get(unit_tag)
            if lost is None or unit_tag in self.cancelled_structure_tags:
                return
            try:
                val = self.calculate_unit_value(lost.type_id)
                value = float(val.minerals + val.vespene)
            except Exception:
                value = 100.0
            # Las estructuras en construcción solo pierden la fracción invertida
            value *= max(0.25, float(getattr(lost, "build_progress", 1.0)))
            self.pending_lost_structure_value += value
            if lost.type_id in (
                UnitTypeId.COMMANDCENTER, UnitTypeId.ORBITALCOMMAND, UnitTypeId.PLANETARYFORTRESS,
                UnitTypeId.COMMANDCENTERFLYING, UnitTypeId.ORBITALCOMMANDFLYING,
                UnitTypeId.NEXUS, UnitTypeId.HATCHERY, UnitTypeId.LAIR, UnitTypeId.HIVE,
            ):
                self.pending_lost_townhalls += 1

        async def on_start(self):
            self.client.game_step = 8 if (collect_rl and not realtime) else 4
            self.lan_client = SC2HailoLANClient(rpi_ip=rpi_ip, port=port)
            mode_tag = "RL Acelerado" if collect_rl else "100% IA Pura"
            print(
                f"🎮 [StarCraft II - {mode_tag}] Mapa '{map_name}' ({player_race} vs {enemy_race} [{difficulty}]). "
                f"Conectado a Raspberry Pi + Hailo-8 ({rpi_ip}:{port})"
            )

        def build_obs_tensor(self) -> np.ndarray:
            """
            Construye el tensor [1, 12, 16, 16] con los valores reales de la partida
            (misma codificación de canales que SC2-Dynamics-50K, sin señales sintéticas).
            """
            tensor = np.zeros((1, IN_CHANNELS, GRID_H, GRID_W), dtype=np.float32)

            # Canales 6..11: Estado económico, suministro, raza y progreso temporal (valores reales, sin reescalados ficticios)
            tensor[0, 6, :, :] = np.clip(float(self.minerals) / 2000.0, 0.0, 1.0)
            tensor[0, 7, :, :] = np.clip(float(self.vespene) / 1500.0, 0.0, 1.0)
            tensor[0, 8, :, :] = np.clip(float(self.supply_used) / 200.0, 0.0, 1.0)
            tensor[0, 9, :, :] = np.clip(float(self.supply_army) / 120.0, 0.0, 1.0)
            tensor[0, 10, :, :] = np.clip(float(self.supply_workers) / 90.0, 0.0, 1.0)

            self_race_id = RACES.get(self.race.name, 0)
            enemy_race_id = RACES.get(self.enemy_race.name, 0)
            tensor[0, 11, 0:4, :] = np.clip(float(self.time) / 1200.0, 0.0, 1.0)
            tensor[0, 11, 4:8, :] = np.clip(float(self.state.score.collection_rate_minerals) / 2500.0, 0.0, 1.0)
            tensor[0, 11, 8:12, :] = (self_race_id + 1) / 3.0
            tensor[0, 11, 12:16, :] = (enemy_race_id + 1) / 3.0

            # Proyección espacial 16x16 proporcional al tamaño total del mapa (sin anclajes ni recortes artificiales)
            map_w = max(1.0, float(self.game_info.map_size.width))
            map_h = max(1.0, float(self.game_info.map_size.height))

            def to_training_grid(pos) -> tuple[int, int]:
                gx = int(np.clip((float(pos.x) / map_w) * GRID_W, 0, GRID_W - 1))
                gy = int(np.clip((float(pos.y) / map_h) * GRID_H, 0, GRID_H - 1))
                return gx, gy

            # Canal 0 y 1: Densidad real y HP promedio de todas las unidades propias en la malla 16x16
            for u in self.units:
                gx, gy = to_training_grid(u.position)
                weight = 0.08 if u.type_id in (UnitTypeId.SCV, UnitTypeId.PROBE, UnitTypeId.DRONE, UnitTypeId.MULE) else 0.20
                tensor[0, 0, gy, gx] = min(1.0, tensor[0, 0, gy, gx] + weight)
                tensor[0, 1, gy, gx] = float(u.health_percentage)

            # Canal 2 y 5: Estructuras propias reales y cola de producción
            active_orders = 0
            for s in self.structures:
                gx, gy = to_training_grid(s.position)
                tensor[0, 2, gy, gx] = min(1.0, tensor[0, 2, gy, gx] + 0.5)
                active_orders += len(s.orders)

            tensor[0, 5, :, :] = np.clip(active_orders / 10.0, 0.0, 1.0)

            # Canal 3 y 4: Unidades y estructuras enemigas visibles reales en el mapa
            for eu in self.enemy_units + self.enemy_structures:
                gx, gy = to_training_grid(eu.position)
                tensor[0, 3, gy, gx] = min(1.0, tensor[0, 3, gy, gx] + 0.25)
                tensor[0, 4, gy, gx] = float(eu.health_percentage)

            return tensor

        def grid_to_point2(self, gx: int, gy: int) -> Point2:
            map_w = float(self.game_info.map_size.width)
            map_h = float(self.game_info.map_size.height)
            px = np.clip(((gx + 0.5) / GRID_W) * map_w, 4.0, map_w - 4.0)
            py = np.clip(((gy + 0.5) / GRID_H) * map_h, 4.0, map_h - 4.0)
            return Point2((float(px), float(py)))

        def resolve_build_location(self, gx: int, gy: int, distance_from_base: float = 8.0) -> Point2:
            """
            Usa directamente la coordenada espacial (gx, gy) predicha por la red neuronal
            escalada sobre las dimensiones completas del mapa.
            """
            return self.grid_to_point2(gx, gy)

        async def execute_decoded_action(self, act_info: Dict[str, object]):
            """
            Traductor directo 1-a-1 de las 32 acciones del vocabulario de SC2HailoTransformer.
            Sin reglas automáticas ni correcciones de destino: todas las coordenadas provienen de (gx, gy).
            """
            act_name = str(act_info["action_name"])
            gx, gy = int(act_info["target_grid"][0]), int(act_info["target_grid"][1])
            target_pt = self.grid_to_point2(gx, gy)
            queue = bool(act_info.get("queue", False))
            army = self.units.exclude_type([UnitTypeId.SCV, UnitTypeId.PROBE, UnitTypeId.DRONE, UnitTypeId.MULE, UnitTypeId.OVERLORD])

            # [00] NO_OP
            if act_name == "NO_OP":
                return

            # [01] HARVEST_GATHER_MINERALS
            elif act_name == "HARVEST_GATHER_MINERALS":
                await self.distribute_workers()

            # [02] HARVEST_GATHER_GAS
            elif act_name == "HARVEST_GATHER_GAS":
                for gas_b in self.gas_buildings.ready:
                    if gas_b.assigned_harvesters < gas_b.ideal_harvesters and self.workers:
                        w = self.workers.closest_to(gas_b)
                        w.gather(gas_b)
                        break

            # [03] TRAIN_WORKER (SCV / Probe / Drone)
            elif act_name == "TRAIN_WORKER":
                for th in self.townhalls.ready:
                    if len(th.orders) < 2:
                        if self.race == Race.Terran and self.can_afford(UnitTypeId.SCV):
                            th.train(UnitTypeId.SCV)
                            break
                        elif self.race == Race.Protoss and self.can_afford(UnitTypeId.PROBE):
                            th.train(UnitTypeId.PROBE)
                            break
                if self.race == Race.Zerg and self.larva and self.can_afford(UnitTypeId.DRONE):
                    self.larva.random.train(UnitTypeId.DRONE)

            # [04] BUILD_SUPPLY (Supply Depot / Pylon / Overlord)
            elif act_name == "BUILD_SUPPLY":
                if self.supply_cap < 200:
                    loc = self.resolve_build_location(gx, gy, distance_from_base=6.0)
                    if self.race == Race.Terran and self.can_afford(UnitTypeId.SUPPLYDEPOT):
                        await self.build(UnitTypeId.SUPPLYDEPOT, near=loc)
                    elif self.race == Race.Protoss and self.can_afford(UnitTypeId.PYLON):
                        await self.build(UnitTypeId.PYLON, near=loc)
                    elif self.race == Race.Zerg and self.larva and self.can_afford(UnitTypeId.OVERLORD):
                        self.larva.random.train(UnitTypeId.OVERLORD)

            # [05] BUILD_GAS_EXTRACTOR (Refinery / Assimilator / Extractor)
            elif act_name == "BUILD_GAS_EXTRACTOR":
                gas_unit = (
                    UnitTypeId.REFINERY if self.race == Race.Terran
                    else (UnitTypeId.ASSIMILATOR if self.race == Race.Protoss else UnitTypeId.EXTRACTOR)
                )
                if self.can_afford(gas_unit) and self.townhalls.ready:
                    for th in self.townhalls.ready:
                        for v in self.vespene_geyser.closer_than(10.0, th):
                            if not self.gas_buildings.closer_than(1.0, v):
                                w = self.select_build_worker(v.position)
                                if w:
                                    w.build_gas(v)
                                    return

            # [06] BUILD_EXPAND_BASE (Command Center / Nexus / Hatchery)
            elif act_name == "BUILD_EXPAND_BASE":
                th_unit = (
                    UnitTypeId.COMMANDCENTER if self.race == Race.Terran
                    else (UnitTypeId.NEXUS if self.race == Race.Protoss else UnitTypeId.HATCHERY)
                )
                if self.can_afford(th_unit) and not self.already_pending(th_unit):
                    all_structures = self.structures + self.enemy_structures
                    free_exps = [
                        loc for loc in self.expansion_locations_list
                        if not all_structures.closer_than(6.0, loc)
                    ]
                    if free_exps:
                        chosen_exp = min(free_exps, key=lambda loc: loc.distance_to(target_pt))
                        await self.build(th_unit, near=chosen_exp, max_distance=6, random_alternative=False, placement_step=1)

            # [07] BUILD_BARRACKS_GATEWAY_POOL
            elif act_name == "BUILD_BARRACKS_GATEWAY_POOL":
                loc = self.resolve_build_location(gx, gy, distance_from_base=8.5)
                if self.race == Race.Terran:
                    if self.can_afford(UnitTypeId.BARRACKS):
                        await self.build(UnitTypeId.BARRACKS, near=loc)
                elif self.race == Race.Protoss and self.can_afford(UnitTypeId.GATEWAY):
                    await self.build(UnitTypeId.GATEWAY, near=loc)
                elif self.race == Race.Zerg and self.can_afford(UnitTypeId.SPAWNINGPOOL):
                    await self.build(UnitTypeId.SPAWNINGPOOL, near=loc)

            # [08] BUILD_FACTORY_ROBO_ROACH
            elif act_name == "BUILD_FACTORY_ROBO_ROACH":
                loc = self.resolve_build_location(gx, gy, distance_from_base=10.5)
                if self.race == Race.Terran and self.can_afford(UnitTypeId.FACTORY):
                    await self.build(UnitTypeId.FACTORY, near=loc)
                elif self.race == Race.Protoss and self.can_afford(UnitTypeId.ROBOTICSFACILITY):
                    await self.build(UnitTypeId.ROBOTICSFACILITY, near=loc)
                elif self.race == Race.Zerg and self.can_afford(UnitTypeId.ROACHWARREN):
                    await self.build(UnitTypeId.ROACHWARREN, near=loc)

            # [09] BUILD_STARPORT_STARGATE_SPIRE
            elif act_name == "BUILD_STARPORT_STARGATE_SPIRE":
                loc = self.resolve_build_location(gx, gy, distance_from_base=12.0)
                if self.race == Race.Terran and self.can_afford(UnitTypeId.STARPORT):
                    await self.build(UnitTypeId.STARPORT, near=loc)
                elif self.race == Race.Protoss and self.can_afford(UnitTypeId.STARGATE):
                    await self.build(UnitTypeId.STARGATE, near=loc)
                elif self.race == Race.Zerg and self.can_afford(UnitTypeId.SPIRE):
                    await self.build(UnitTypeId.SPIRE, near=loc)

            # [10] BUILD_TECH_LAB_CORE_LAIR
            elif act_name == "BUILD_TECH_LAB_CORE_LAIR":
                if self.race == Race.Terran:
                    for fac in self.structures(UnitTypeId.FACTORY).ready.idle:
                        if not fac.has_add_on and self.can_afford(UnitTypeId.FACTORYTECHLAB):
                            fac.build(UnitTypeId.FACTORYTECHLAB)
                            return
                    for rax in self.structures(UnitTypeId.BARRACKS).ready.idle:
                        if not rax.has_add_on and self.can_afford(UnitTypeId.BARRACKSTECHLAB):
                            rax.build(UnitTypeId.BARRACKSTECHLAB)
                            return
                    for sp in self.structures(UnitTypeId.STARPORT).ready.idle:
                        if not sp.has_add_on and self.can_afford(UnitTypeId.STARPORTREACTOR):
                            sp.build(UnitTypeId.STARPORTREACTOR)
                            return
                elif self.race == Race.Protoss and self.can_afford(UnitTypeId.CYBERNETICSCORE):
                    await self.build(UnitTypeId.CYBERNETICSCORE, near=self.resolve_build_location(gx, gy, 7.0))
                elif self.race == Race.Zerg and self.townhalls(UnitTypeId.HATCHERY).ready.idle and self.can_afford(UnitTypeId.LAIR):
                    self.townhalls(UnitTypeId.HATCHERY).ready.idle.first.build(UnitTypeId.LAIR)

            # [11] BUILD_DEFENSE_STRUCTURE (Bunker / Missile Turret / Cannon / Spine)
            elif act_name == "BUILD_DEFENSE_STRUCTURE":
                loc = self.resolve_build_location(gx, gy, distance_from_base=9.5)
                if self.race == Race.Terran:
                    if self.structures(UnitTypeId.BUNKER).amount < 2 and self.can_afford(UnitTypeId.BUNKER):
                        await self.build(UnitTypeId.BUNKER, near=loc)
                    elif self.can_afford(UnitTypeId.MISSILETURRET):
                        await self.build(UnitTypeId.MISSILETURRET, near=loc)
                elif self.race == Race.Protoss and self.can_afford(UnitTypeId.PHOTONCANNON):
                    await self.build(UnitTypeId.PHOTONCANNON, near=loc)
                elif self.race == Race.Zerg and self.can_afford(UnitTypeId.SPINECRAWLER):
                    await self.build(UnitTypeId.SPINECRAWLER, near=loc)

            # [12] TRAIN_BASIC_INFANTRY (Marine / Zealot / Zergling)
            elif act_name == "TRAIN_BASIC_INFANTRY":
                if self.race == Race.Terran:
                    for rax in self.structures(UnitTypeId.BARRACKS).ready:
                        if len(rax.orders) < 2 and self.can_afford(UnitTypeId.MARINE):
                            rax.train(UnitTypeId.MARINE)
                            break
                elif self.race == Race.Protoss:
                    for gw in self.structures(UnitTypeId.GATEWAY).ready:
                        if len(gw.orders) < 2 and self.can_afford(UnitTypeId.ZEALOT):
                            gw.train(UnitTypeId.ZEALOT)
                            break
                elif self.race == Race.Zerg and self.larva and self.can_afford(UnitTypeId.ZERGLING):
                    self.larva.random.train(UnitTypeId.ZERGLING)

            # [13] TRAIN_RANGED_CORE (Marauder / Stalker / Roach)
            elif act_name == "TRAIN_RANGED_CORE":
                if self.race == Race.Terran:
                    for rax in self.structures(UnitTypeId.BARRACKS).ready:
                        if getattr(rax, "has_techlab", False) and len(rax.orders) < 2 and self.can_afford(UnitTypeId.MARAUDER):
                            rax.train(UnitTypeId.MARAUDER)
                            return
                elif self.race == Race.Protoss:
                    for gw in self.structures(UnitTypeId.GATEWAY).ready:
                        if len(gw.orders) < 2 and self.can_afford(UnitTypeId.STALKER):
                            gw.train(UnitTypeId.STALKER)
                            break
                elif self.race == Race.Zerg and self.larva and self.can_afford(UnitTypeId.ROACH):
                    self.larva.random.train(UnitTypeId.ROACH)

            # [14] TRAIN_SPECIALIST_UNIT (Reaper / Ghost / Sentry / Queen)
            elif act_name == "TRAIN_SPECIALIST_UNIT":
                if self.race == Race.Terran:
                    for rax in self.structures(UnitTypeId.BARRACKS).ready:
                        if len(rax.orders) < 2 and self.can_afford(UnitTypeId.REAPER):
                            rax.train(UnitTypeId.REAPER)
                            break
                elif self.race == Race.Protoss:
                    for gw in self.structures(UnitTypeId.GATEWAY).ready:
                        if len(gw.orders) < 2 and self.can_afford(UnitTypeId.SENTRY):
                            gw.train(UnitTypeId.SENTRY)
                            break
                elif self.race == Race.Zerg:
                    for th in self.townhalls.ready:
                        if len(th.orders) < 2 and self.can_afford(UnitTypeId.QUEEN):
                            th.train(UnitTypeId.QUEEN)
                            break

            # [15] TRAIN_SIEGE_HEAVY (Siege Tank / Hellion / Thor / Immortal / Ultralisk)
            elif act_name == "TRAIN_SIEGE_HEAVY":
                if self.race == Race.Terran:
                    for fac in self.structures(UnitTypeId.FACTORY).ready:
                        if getattr(fac, "has_techlab", False) and len(fac.orders) < 2 and self.can_afford(UnitTypeId.SIEGETANK):
                            fac.train(UnitTypeId.SIEGETANK)
                            return
                        elif len(fac.orders) < 2 and self.can_afford(UnitTypeId.HELLION):
                            fac.train(UnitTypeId.HELLION)
                            return
                elif self.race == Race.Protoss:
                    for robo in self.structures(UnitTypeId.ROBOTICSFACILITY).ready:
                        if len(robo.orders) < 2 and self.can_afford(UnitTypeId.IMMORTAL):
                            robo.train(UnitTypeId.IMMORTAL)
                            break
                elif self.race == Race.Zerg and self.larva and self.can_afford(UnitTypeId.ULTRALISK):
                    self.larva.random.train(UnitTypeId.ULTRALISK)

            # [16] TRAIN_AIR_UNIT (Medivac / Viking / Liberator / VoidRay / Mutalisk)
            elif act_name == "TRAIN_AIR_UNIT":
                if self.race == Race.Terran:
                    for sp in self.structures(UnitTypeId.STARPORT).ready:
                        if len(sp.orders) < 2:
                            if self.units(UnitTypeId.MEDIVAC).amount < 4 and self.can_afford(UnitTypeId.MEDIVAC):
                                sp.train(UnitTypeId.MEDIVAC)
                                break
                            elif self.can_afford(UnitTypeId.VIKINGFIGHTER):
                                sp.train(UnitTypeId.VIKINGFIGHTER)
                                break
                elif self.race == Race.Protoss:
                    for sg in self.structures(UnitTypeId.STARGATE).ready:
                        if len(sg.orders) < 2 and self.can_afford(UnitTypeId.VOIDRAY):
                            sg.train(UnitTypeId.VOIDRAY)
                            break
                elif self.race == Race.Zerg and self.larva and self.can_afford(UnitTypeId.MUTALISK):
                    self.larva.random.train(UnitTypeId.MUTALISK)

            # [17] RESEARCH_COMBAT_UPGRADE (Stimpack / Combat Shield / Concussive Shells)
            elif act_name == "RESEARCH_COMBAT_UPGRADE":
                for lab in self.structures(UnitTypeId.BARRACKSTECHLAB).ready.idle:
                    if self.can_afford(UpgradeId.STIMPACK):
                        lab.research(UpgradeId.STIMPACK)
                        return
                    if self.can_afford(UpgradeId.SHIELDWALL):
                        lab.research(UpgradeId.SHIELDWALL)
                        return
                    if self.can_afford(UpgradeId.PUNISHERGRENADES):
                        lab.research(UpgradeId.PUNISHERGRENADES)
                        return

            # [18] RESEARCH_WEAPONS_ARMOR (Engineering Bay +1/+2/+3 Infantry Weapons & Armor)
            elif act_name == "RESEARCH_WEAPONS_ARMOR":
                if self.race == Race.Terran:
                    for eb in self.structures(UnitTypeId.ENGINEERINGBAY).ready.idle:
                        for upg in (
                            UpgradeId.TERRANINFANTRYWEAPONSLEVEL1,
                            UpgradeId.TERRANINFANTRYARMORSLEVEL1,
                            UpgradeId.TERRANINFANTRYWEAPONSLEVEL2,
                            UpgradeId.TERRANINFANTRYARMORSLEVEL2,
                        ):
                            if self.can_afford(upg):
                                eb.research(upg)
                                return

            # [19] MACRO_CHRONO_MULE_INJECT (Orbital Command + Calldown MULE / Chrono / Larva Inject)
            elif act_name == "MACRO_CHRONO_MULE_INJECT":
                if self.race == Race.Terran:
                    for cc in self.townhalls(UnitTypeId.COMMANDCENTER).ready.idle:
                        if self.structures(UnitTypeId.BARRACKS).ready and self.can_afford(UnitTypeId.ORBITALCOMMAND):
                            cc(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND)
                            return
                    for oc in self.townhalls(UnitTypeId.ORBITALCOMMAND).ready:
                        if oc.energy >= 50 and self.mineral_field:
                            oc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, self.mineral_field.closest_to(oc))
                            return

            # [20] MOVE_ARMY_POSITION
            elif act_name == "MOVE_ARMY_POSITION":
                for u in army:
                    u.move(target_pt, queue=queue)

            # [21] MOVE_WORKER_SCOUT
            elif act_name == "MOVE_WORKER_SCOUT":
                if self.workers:
                    scout = self.workers.find_by_tag(self.scout_worker_tag) if self.scout_worker_tag else None
                    if scout is None and self.supply_workers >= 14:
                        scout = self.workers.first
                        self.scout_worker_tag = scout.tag
                    if scout is not None:
                        scout.move(target_pt, queue=queue)

            # [22] ATTACK_MOVE_COORDINATE
            elif act_name == "ATTACK_MOVE_COORDINATE":
                for u in army:
                    if u.type_id == UnitTypeId.MEDIVAC and army.exclude_type([UnitTypeId.MEDIVAC]):
                        u.attack(army.exclude_type([UnitTypeId.MEDIVAC]).center, queue=queue)
                    else:
                        u.attack(target_pt, queue=queue)

            # [23] FOCUS_FIRE_TARGET
            elif act_name == "FOCUS_FIRE_TARGET":
                nearby_enemies = self.enemy_units.closer_than(15.0, target_pt) if self.enemy_units else []
                if nearby_enemies and army:
                    weakest = min(nearby_enemies, key=lambda e: e.health + e.shield)
                    for u in army:
                        u.attack(weakest, queue=queue)
                else:
                    for u in army:
                        u.attack(target_pt, queue=queue)

            # [24] MICRO_KITE_RETREAT
            elif act_name == "MICRO_KITE_RETREAT":
                for u in army:
                    if u.health_percentage < 0.65 or (self.enemy_units and self.enemy_units.closer_than(6.0, u)):
                        u.move(u.position.towards(target_pt, 5.0), queue=queue)

            # [25] MICRO_USE_STIM_ABILITY
            elif act_name == "MICRO_USE_STIM_ABILITY":
                for u in self.units([UnitTypeId.MARINE, UnitTypeId.MARAUDER]):
                    if u.health_percentage > 0.55 and not u.has_buff(
                        getattr(__import__("sc2.ids.buff_id", fromlist=["BuffId"]).BuffId, "STIMPACK", None)
                    ):
                        if self.enemy_units and self.enemy_units.closer_than(14.0, u):
                            stim_ab = (
                                AbilityId.EFFECT_STIM_MARINE
                                if u.type_id == UnitTypeId.MARINE
                                else AbilityId.EFFECT_STIM_MARAUDER
                            )
                            u(stim_ab)

            # [26] MICRO_SIEGE_TRANSFORM (Siege Tank Siege / Unsiege, Supply Depot Lower / Raise, Viking Morph)
            elif act_name == "MICRO_SIEGE_TRANSFORM":
                if self.race == Race.Terran:
                    raised = self.structures(UnitTypeId.SUPPLYDEPOT).ready
                    lowered = self.structures(UnitTypeId.SUPPLYDEPOTLOWERED).ready
                    if raised:
                        raised.closest_to(target_pt)(AbilityId.MORPH_SUPPLYDEPOT_LOWER)
                    elif lowered:
                        lowered.closest_to(target_pt)(AbilityId.MORPH_SUPPLYDEPOT_RAISE)
                for tank in self.units(UnitTypeId.SIEGETANK):
                    tank(AbilityId.SIEGEMODE_SIEGEMODE)
                for sieged in self.units(UnitTypeId.SIEGETANKSIEGED):
                    sieged(AbilityId.UNSIEGE_UNSIEGE)

            # [27] MICRO_HEAL_REPAIR_SHIELD
            elif act_name == "MICRO_HEAL_REPAIR_SHIELD":
                damaged_mech = [
                    u for u in (self.units + self.structures)
                    if u.is_mechanical and u.health_percentage < 1.0
                ]
                if damaged_mech and self.workers:
                    target_mech = min(damaged_mech, key=lambda m: m.health_percentage)
                    rep_worker = self.workers.closest_to(target_mech)
                    rep_worker.repair(target_mech)
                for med in self.units(UnitTypeId.MEDIVAC):
                    if army:
                        med.attack(army.center)

            # [28] DEFEND_MAIN_NATURAL
            elif act_name == "DEFEND_MAIN_NATURAL":
                threats = self.enemy_units.closer_than(25.0, target_pt) if self.enemy_units else []
                for u in army:
                    if threats:
                        u.attack(threats.closest_to(u), queue=queue)
                    else:
                        u.attack(target_pt, queue=queue)

            # [29] HARASS_ENEMY_WORKERS
            elif act_name == "HARASS_ENEMY_WORKERS":
                enemy_workers = (
                    self.enemy_units.of_type([UnitTypeId.SCV, UnitTypeId.PROBE, UnitTypeId.DRONE]).closer_than(15.0, target_pt)
                    if self.enemy_units else []
                )
                dest = enemy_workers.center if enemy_workers else target_pt
                harass_squad = self.units([UnitTypeId.REAPER, UnitTypeId.HELLION, UnitTypeId.BANSHEE]) or army
                for u in harass_squad:
                    u.attack(dest, queue=queue)

            # [30] RALLY_PRODUCTION
            elif act_name == "RALLY_PRODUCTION":
                for prod in self.structures([UnitTypeId.BARRACKS, UnitTypeId.FACTORY, UnitTypeId.STARPORT]).ready:
                    prod(AbilityId.RALLY_BUILDING, target_pt)

            # [31] CANCEL_OR_HOLD_POSITION
            elif act_name == "CANCEL_OR_HOLD_POSITION":
                for s in self.structures.not_ready:
                    if s.health_percentage < 0.20:
                        self.cancelled_structure_tags.add(s.tag)
                        s(AbilityId.CANCEL_BUILDINPROGRESS)
                        return
                for u in army.closer_than(10.0, target_pt):
                    u.hold_position(queue=queue)

        def apply_rl_exploration(
            self,
            chunks: List[Dict[str, object]],
            obs_tensor: Optional[np.ndarray] = None,
        ) -> List[Dict[str, object]]:
            """
            Exploración táctica (ε-greedy) durante --collect-rl:
            Con probabilidad explore_eps muestrea acciones y celdas (gx, gy) en la malla 16x16
            guiándose únicamente por los canales del propio tensor de observación (Canal 3 = enemigos
            visibles, Canal 0 = celdas aún no ocupadas por unidades aliadas) para evitar ruido de entrenamiento.
            """
            if not collect_rl or explore_eps <= 0.0:
                return chunks

            map_w = max(1.0, float(self.game_info.map_size.width))
            map_h = max(1.0, float(self.game_info.map_size.height))
            base_pos = self.townhalls.first.position if self.townhalls else self.start_location
            bgx = int(np.clip((float(base_pos.x) / map_w) * GRID_W, 1, GRID_W - 2))
            bgy = int(np.clip((float(base_pos.y) / map_h) * GRID_H, 1, GRID_H - 2))

            def near_base_cell() -> Tuple[int, int]:
                return (
                    int(np.clip(bgx + int(self.rng.integers(-2, 3)), 0, GRID_W - 1)),
                    int(np.clip(bgy + int(self.rng.integers(-2, 3)), 0, GRID_H - 1)),
                )

            def rand_map_cell() -> Tuple[int, int]:
                return int(self.rng.integers(0, GRID_W)), int(self.rng.integers(0, GRID_H))

            def sample_combat_cell() -> Tuple[int, int]:
                if obs_tensor is not None:
                    enemy_yx = np.argwhere(obs_tensor[0, 3] > 0.0)
                    if len(enemy_yx) > 0 and self.rng.random() < 0.70:
                        idx = int(self.rng.integers(0, len(enemy_yx)))
                        return int(enemy_yx[idx][1]), int(enemy_yx[idx][0])
                    unvisited_yx = np.argwhere(~self.army_visited_cells)
                    if len(unvisited_yx) > 0:
                        idx = int(self.rng.integers(0, len(unvisited_yx)))
                        return int(unvisited_yx[idx][1]), int(unvisited_yx[idx][0])
                return rand_map_cell()

            candidate_actions: List[Tuple[str, int, int]] = []
            # Candidatos económicos y de suministro
            if self.supply_left <= 3 and self.supply_cap < 200 and self.minerals >= 100:
                candidate_actions.append(("BUILD_SUPPLY", *near_base_cell()))
            if self.supply_workers < min(66, self.townhalls.amount * 22):
                candidate_actions.append(("TRAIN_WORKER", bgx, bgy))
            if self.minerals >= 400 and self.townhalls.amount < 3:
                candidate_actions.append(("BUILD_EXPAND_BASE", *rand_map_cell()))
            if self.townhalls.ready and self.gas_buildings.amount < self.townhalls.amount * 2 and self.minerals >= 75:
                candidate_actions.append(("BUILD_GAS_EXTRACTOR", bgx, bgy))
            if self.gas_buildings.ready:
                candidate_actions.append(("HARVEST_GATHER_GAS", bgx, bgy))

            # Candidatos de infraestructura y producción
            if self.minerals >= 150 and self.structures(UnitTypeId.BARRACKS).amount < 3:
                candidate_actions.append(("BUILD_BARRACKS_GATEWAY_POOL", *near_base_cell()))
            if self.structures(UnitTypeId.BARRACKS).ready:
                candidate_actions.append(("TRAIN_BASIC_INFANTRY", bgx, bgy))
                candidate_actions.append(("MACRO_CHRONO_MULE_INJECT", bgx, bgy))
                if self.vespene >= 50:
                    candidate_actions.append(("BUILD_TECH_LAB_CORE_LAIR", bgx, bgy))
                    candidate_actions.append(("TRAIN_RANGED_CORE", bgx, bgy))
                    candidate_actions.append(("RESEARCH_COMBAT_UPGRADE", bgx, bgy))
                if self.minerals >= 150 and self.vespene >= 100 and self.structures(UnitTypeId.FACTORY).amount < 2:
                    candidate_actions.append(("BUILD_FACTORY_ROBO_ROACH", *near_base_cell()))
            if self.structures(UnitTypeId.FACTORY).ready:
                candidate_actions.append(("TRAIN_SIEGE_HEAVY", bgx, bgy))
                if self.minerals >= 150 and self.vespene >= 100 and self.structures(UnitTypeId.STARPORT).amount < 1:
                    candidate_actions.append(("BUILD_STARPORT_STARGATE_SPIRE", *near_base_cell()))
            if self.structures(UnitTypeId.STARPORT).ready:
                candidate_actions.append(("TRAIN_AIR_UNIT", bgx, bgy))

            # Candidatos de combate y micro (muestreando celdas con enemigos visibles en Canal 3 o zonas no exploradas)
            if self.supply_army >= 12:
                cx, cy = sample_combat_cell()
                candidate_actions.append(("ATTACK_MOVE_COORDINATE", cx, cy))
                candidate_actions.append(("FOCUS_FIRE_TARGET", cx, cy))
                # Habilidades sin destino espacial: solo se exploran cuando existen las unidades que las usan
                if self.enemy_units and self.units([UnitTypeId.MARINE, UnitTypeId.MARAUDER]):
                    candidate_actions.append(("MICRO_USE_STIM_ABILITY", cx, cy))
                if self.units([UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED]):
                    candidate_actions.append(("MICRO_SIEGE_TRANSFORM", cx, cy))
            if self.enemy_units and self.townhalls and self.enemy_units.closer_than(25.0, self.townhalls.first):
                candidate_actions.append(("DEFEND_MAIN_NATURAL", bgx, bgy))

            if not candidate_actions:
                return chunks

            out_chunks = [dict(c) for c in chunks]
            for k in range(len(out_chunks)):
                is_redundant_harvest = (k >= 1 and out_chunks[k]["action_name"] == "HARVEST_GATHER_MINERALS")
                prob = min(0.65, explore_eps * 2.0) if is_redundant_harvest else explore_eps
                if self.rng.random() < prob:
                    act_name, gx, gy = candidate_actions[int(self.rng.integers(0, len(candidate_actions)))]
                    out_chunks[k] = {
                        "step": k,
                        "action_id": ACTION_TO_ID[act_name],
                        "action_name": act_name,
                        "target_grid": [gx, gy],
                        "target_sc2_coord": [int((gx + 0.5) * 4), int((gy + 0.5) * 4)],
                        "queue": False,
                    }
            return out_chunks

        def compute_chunk_reward(self) -> float:
            """
            Calcula la recompensa densa r_t del último Action-Chunk ejecutado:
            - Premia destruir unidades/estructuras enemigas, crecer economía/ejército y expandir bases.
            - Penaliza perder estructuras propias (simétrico al premio por destruir estructuras enemigas)
              y perder centros de mando / bases (simétrico al premio por expandir).
            - Premia alinear las coordenadas de ataque (gx, gy) con celdas donde el Canal 3 muestra enemigos visibles.
            - Premia (una sola vez por celda) llevar el ejército a celdas nunca recorridas cuando no hay enemigos visibles.
            - Penaliza ignorar enemigos visibles en el Canal 3, quedarse estancado atacando una celda vacía donde ya
              está el propio ejército, los bloqueos de suministro y la acumulación ociosa de minerales.
            """
            score = self.state.score
            killed_val = float(score.killed_value_units) + 1.5 * float(score.killed_value_structures)
            army_val = float(score.total_value_units)
            workers = float(self.supply_workers)
            bases = int(self.townhalls.amount)

            d_killed = killed_val - self.prev_killed_val
            d_army = army_val - self.prev_army_val
            d_workers = workers - self.prev_workers
            d_bases = bases - self.prev_bases

            self.prev_killed_val = killed_val
            self.prev_army_val = army_val
            self.prev_workers = workers
            self.prev_bases = bases

            r = (
                (d_killed / 200.0)
                + (d_army / 400.0)
                + (0.15 * d_workers)
                + (0.80 * max(0, d_bases))
            )

            # Penalización por estructuras aliadas destruidas (misma escala que el premio por estructuras enemigas)
            if self.pending_lost_structure_value > 0.0:
                r -= 1.5 * self.pending_lost_structure_value / 200.0
                r -= 0.80 * self.pending_lost_townhalls
                self.pending_lost_structure_value = 0.0
                self.pending_lost_townhalls = 0

            if self.supply_left <= 0 and self.supply_cap < 200 and self.minerals >= 100:
                r -= 0.15
            if self.minerals > 600 and self.supply_used < 185:
                r -= 0.10

            # Evaluación espacial de la atención (Canal 3 = enemigos visibles, Canal 0 = unidades propias)
            if self.rl_states and self.rl_actions:
                last_state = self.rl_states[-1]
                last_chunk = self.rl_actions[-1]
                enemy_map = last_state[3]
                ally_map = last_state[0]
                has_visible_enemies = bool(np.any(enemy_map > 0.0))

                spatial_delta = 0.0
                for k in range(len(last_chunk)):
                    act_id = int(last_chunk[k, 0])
                    gx = int(np.clip(last_chunk[k, 1], 0, GRID_W - 1))
                    gy = int(np.clip(last_chunk[k, 2], 0, GRID_H - 1))

                    if act_id in (22, 23, 28, 29):  # Acciones ofensivas / combate
                        y0, y1 = max(0, gy - 1), min(GRID_H, gy + 2)
                        x0, x1 = max(0, gx - 1), min(GRID_W, gx + 2)
                        local_enemy = float(np.max(enemy_map[y0:y1, x0:x1]))
                        if has_visible_enemies:
                            if local_enemy > 0.0:
                                spatial_delta = max(spatial_delta, 0.30)
                            elif act_id in (22, 23) and spatial_delta == 0.0:
                                spatial_delta = min(spatial_delta, -0.25)
                        else:
                            # Sin enemigos visibles: penalizar quedarse atacando la celda vacía donde ya está nuestro ejército
                            if ally_map[gy, gx] >= 0.20:
                                spatial_delta = min(spatial_delta, -0.20)
                            elif self.supply_army >= 10 and not self.army_visited_cells[gy, gx]:
                                spatial_delta = max(spatial_delta, 0.12)
                    elif (
                        act_id == 20
                        and not has_visible_enemies
                        and self.supply_army >= 10
                        and not self.army_visited_cells[gy, gx]
                    ):
                        spatial_delta = max(spatial_delta, 0.12)

                r += spatial_delta

            # Actualizar el mapa de celdas recorridas por el ejército (solo unidades de combate)
            map_w = max(1.0, float(self.game_info.map_size.width))
            map_h = max(1.0, float(self.game_info.map_size.height))
            for u in self.units.exclude_type([UnitTypeId.SCV, UnitTypeId.PROBE, UnitTypeId.DRONE, UnitTypeId.MULE, UnitTypeId.OVERLORD]):
                vx = int(np.clip((float(u.position.x) / map_w) * GRID_W, 0, GRID_W - 1))
                vy = int(np.clip((float(u.position.y) / map_h) * GRID_H, 0, GRID_H - 1))
                self.army_visited_cells[vy, vx] = True

            return float(np.clip(r, -2.5, 2.5))

        async def on_step(self, iteration: int):
            if max_game_steps > 0 and iteration >= max_game_steps:
                self.reached_step_limit = True
                print(f"🏁 Límite de pasos alcanzado ({max_game_steps}). Evaluando ventaja y cerrando partida...")
                await self.client.leave()
                return

            # Consultamos al HAT Hailo-8 en un hilo asíncrono cuando la cola del Action-Chunk (K=4) se vacía
            if not self.chunk_queue and self.lan_client is not None:
                if collect_rl and self.rl_states:
                    self.rl_rewards.append(self.compute_chunk_reward())

                obs_tensor = self.build_obs_tensor()
                t0 = time.perf_counter()
                resp = await asyncio.to_thread(self.lan_client.request_action_chunk, obs_tensor)
                self.last_rtt_ms = (time.perf_counter() - t0) * 1000.0
                self.last_npu_ms = float(resp.get("latency_ms", 0.0))
                raw_chunks = list(resp.get("chunks", []))
                self.chunk_queue = self.apply_rl_exploration(raw_chunks, obs_tensor=obs_tensor)
                self.total_inferences += 1

                if collect_rl and len(self.chunk_queue) == ACTION_CHUNK_SIZE:
                    self.rl_states.append(obs_tensor[0].copy())
                    chunk_arr = np.zeros((ACTION_CHUNK_SIZE, 4), dtype=np.int64)
                    for k, ch_item in enumerate(self.chunk_queue):
                        chunk_arr[k, 0] = int(ch_item["action_id"])
                        chunk_arr[k, 1] = int(ch_item["target_grid"][0])
                        chunk_arr[k, 2] = int(ch_item["target_grid"][1])
                        chunk_arr[k, 3] = 1 if ch_item.get("queue", False) else 0
                    self.rl_actions.append(chunk_arr)

            if self.chunk_queue:
                next_act = self.chunk_queue.pop(0)
                self.last_action_name = str(next_act["action_name"])
                try:
                    await self.execute_decoded_action(next_act)
                except Exception as exc:
                    print(f"[Aviso Bot] Acción omitida ({self.last_action_name}): {exc}")

            if not human_vs_ai and not collect_rl:
                army_cnt = self.units.exclude_type([UnitTypeId.SCV, UnitTypeId.MULE]).amount
                hud_text = (
                    f"Hailo-8 NPU (100% IA Pura): {self.last_npu_ms:.2f} ms | LAN RTT: {self.last_rtt_ms:.1f} ms\n"
                    f"Inferences: {self.total_inferences} | Action: {self.last_action_name}\n"
                    f"Minerals: {self.minerals} | Gas: {self.vespene} | Workers: {self.supply_workers} | Army: {army_cnt}"
                )
                self.client.debug_text_screen(hud_text, pos=(0.01, 0.02), size=14)

            log_interval = 60 if collect_rl else 20
            if iteration % log_interval == 0:
                army_cnt = self.units.exclude_type([UnitTypeId.SCV, UnitTypeId.MULE]).amount
                print(
                    f"[SC2 t={self.time_formatted}] Hailo-8 NPU: {self.last_npu_ms:.2f} ms | "
                    f"RTT: {self.last_rtt_ms:.1f} ms | Min: {self.minerals} | Workers: {self.supply_workers} | "
                    f"Army: {army_cnt} | Acción: {self.last_action_name}"
                )

        async def on_end(self, game_result):
            result_tag = "Result.Timeout" if (self.reached_step_limit and game_result != Result.Victory) else str(game_result)
            print(f"🏆 Partida finalizada con resultado: {result_tag}")
            if collect_rl and self.rl_states:
                while len(self.rl_rewards) < len(self.rl_states):
                    self.rl_rewards.append(self.compute_chunk_reward())

                # Recompensa terminal: Victoria real (+10), Tiempo agotado (+2.5 / -2.5), o Derrota real (-8)
                won = 0
                if game_result == Result.Victory:
                    terminal_bonus = 10.0
                    won = 1
                elif self.reached_step_limit:
                    score = self.state.score
                    net_adv = (
                        float(score.killed_value_units)
                        + 1.5 * float(score.killed_value_structures)
                        + 0.5 * float(score.total_value_units)
                    )
                    terminal_bonus = 2.5 if (net_adv >= 1800.0 or self.supply_army >= 22) else -2.5
                    won = 1 if terminal_bonus > 0 else 0
                elif game_result == Result.Defeat:
                    terminal_bonus = -8.0
                    won = 0
                else:
                    terminal_bonus = 0.0
                    won = 0

                self.rl_rewards[-1] += terminal_bonus

                # Calcular retornos descontados G_t (gamma = 0.97)
                gamma = 0.97
                T = len(self.rl_rewards)
                returns = np.zeros(T, dtype=np.float32)
                running_g = 0.0
                for t in reversed(range(T)):
                    running_g = float(self.rl_rewards[t]) + gamma * running_g
                    returns[t] = running_g

                ep_states = np.stack(self.rl_states, axis=0).astype(np.float32)
                ep_actions = np.stack(self.rl_actions, axis=0).astype(np.int64)
                self_race_id = RACES.get(player_race, 0)
                ep_races = np.full(T, self_race_id, dtype=np.int64)

                total_buf = append_rl_rollout_to_drive(
                    ep_states=ep_states,
                    ep_actions=ep_actions,
                    ep_returns=returns,
                    ep_races=ep_races,
                    ep_won=won,
                    rollout_path=rollout_path,
                )
                match_summary["won"] = won
                match_summary["result_str"] = result_tag
                match_summary["steps_recorded"] = T
                match_summary["total_buffer_samples"] = total_buf
                match_summary["episode_return"] = round(float(returns[0]), 2)
                print(
                    f"💾 [RL Guardado en Drive] +{T} muestras | Retorno G_0={returns[0]:+.2f} | "
                    f"Total acumulado en '{rollout_path}': {total_buf} muestras"
                )

            if self.lan_client is not None:
                self.lan_client.close()

    if human_vs_ai:
        import sc2.main as sc2_main
        orig_sc2_proc = sc2_main.SC2Process
        instance_counter = {"count": 0}

        def patched_sc2_process(*args, **kwargs):
            instance_counter["count"] += 1
            if instance_counter["count"] == 1:
                kwargs.setdefault("resolution", (1280, 720))
                kwargs.setdefault("placement", (50, 50))
            else:
                kwargs.setdefault("resolution", (320, 240))
                kwargs.setdefault("placement", (1400, 700))
            return orig_sc2_proc(*args, **kwargs)

        sc2_main.SC2Process = patched_sc2_process

    players_list = (
        [
            Human(race_map.get(enemy_race, Race.Zerg), name="Stalin"),
            Bot(race_map.get(player_race, Race.Terran), Hailo8StarCraftBot(), name="Hailo8_Transformer"),
        ]
        if human_vs_ai
        else [
            Bot(race_map.get(player_race, Race.Terran), Hailo8StarCraftBot(), name="Hailo8_Transformer"),
            Computer(race_map.get(enemy_race, Race.Terran), diff_map.get(difficulty, Difficulty.Medium)),
        ]
    )

    run_game(
        maps.get(map_name),
        players_list,
        realtime=True if human_vs_ai else realtime,
    )
    return match_summary


def _disable_windows_quickedit():
    """Evita que un clic accidental en la ventana CMD congele el proceso y mantiene encendida la pantalla."""
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED (evita que el monitor entre en suspensión durante RL)
        kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
        h_stdin = kernel32.GetStdHandle(-10)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(h_stdin, ctypes.byref(mode)):
            # Desactivar ENABLE_QUICK_EDIT_MODE (0x0040) y activar ENABLE_EXTENDED_FLAGS (0x0080)
            new_mode = (mode.value & ~0x0040) | 0x0080
            kernel32.SetConsoleMode(h_stdin, new_mode)
    except Exception:
        pass


def _cleanup_orphan_sc2():
    """Limpia cualquier proceso SC2_x64.exe huérfano o bloqueado antes/después de cada episodio."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/IM", "SC2_x64.exe", "/T"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def run_rl_collection_loop(
    rpi_ip: str = "192.168.50.7",
    port: int = 5555,
    player_race: str = "Terran",
    start_difficulty: str = "Easy",
    episodes: int = 30,
    max_game_steps: int = 2400,
    explore_eps: float = 0.20,
    rollout_path: str = DEFAULT_DRIVE_ROLLOUT_PATH,
):
    """
    Ejecuta un bucle automático de N partidas aceleradas (realtime=False) a prueba de bloqueos:
    - Cada episodio corre en un subproceso aislado con timeout estricto (240s) y limpieza de SC2_x64.exe.
    - Rota entre mapas oficiales y las 3 razas enemigas (Zerg, Terran, Protoss).
    - Si el porcentaje de victorias alcanza >= 60%, sube automáticamente la dificultad
      (Easy -> Medium -> Hard -> VeryHard).
    """
    _disable_windows_quickedit()
    _cleanup_orphan_sc2()
    time.sleep(1.0)

    diff_ladder = ["Easy", "Medium", "Hard", "VeryHard"]
    curr_diff_idx = diff_ladder.index(start_difficulty) if start_difficulty in diff_ladder else 0
    map_pool = ["Simple64", "AbyssalReefLE", "BelShirVestigeLE", "CactusValleyLE"]
    enemy_races = ["Zerg", "Terran", "Protoss"]

    recent_wins: List[int] = []
    total_wins = 0
    completed_eps = 0
    t_start = time.perf_counter()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("============================================================================")
    print(f" 🚀 INICIANDO RECOLECCIÓN RL ACELERADA ({episodes} Partidas | Supervisor Anti-Hang)")
    print(f" 📁 Guardado automático en Drive: {rollout_path}")
    print("============================================================================")

    for ep in range(1, episodes + 1):
        curr_diff = diff_ladder[curr_diff_idx]
        curr_map = map_pool[(ep - 1) % len(map_pool)]
        curr_enemy = enemy_races[(ep - 1) % len(enemy_races)]
        ep_eps = max(0.08, explore_eps * (1.0 - 0.4 * ((ep - 1) / max(1, episodes))))

        summary: Dict[str, object] = {}
        ep_t0 = time.perf_counter()

        for attempt in (1, 2):
            print(
                f"\n▶️ [Episodio {ep:02d}/{episodes:02d} | Intento {attempt}/2] "
                f"Iniciando {curr_map} ({player_race} vs {curr_enemy} [{curr_diff}], eps={ep_eps:.2f})...",
                flush=True,
            )
            fd, summary_json_path = tempfile.mkstemp(prefix="sc2_ep_", suffix=".json")
            os.close(fd)
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "src.pc_sc2_client",
                "--single-rl-ep",
                "--rpi-ip",
                rpi_ip,
                "--port",
                str(port),
                "--map",
                curr_map,
                "--race",
                player_race,
                "--enemy-race",
                curr_enemy,
                "--difficulty",
                curr_diff,
                "--max-steps",
                str(max_game_steps),
                "--explore-eps",
                f"{ep_eps:.4f}",
                "--rollout-path",
                rollout_path,
                "--summary-json",
                summary_json_path,
            ]
            proc = subprocess.Popen(cmd, cwd=project_root)
            try:
                proc.wait(timeout=240)
                if os.path.exists(summary_json_path) and os.path.getsize(summary_json_path) > 0:
                    with open(summary_json_path, "r", encoding="utf-8") as sf:
                        summary = json.load(sf)
            except subprocess.TimeoutExpired:
                print(
                    f"⚠️ [Supervisor Anti-Hang] SC2_x64.exe excedió 240s en Episodio {ep:02d} (intento {attempt}/2). "
                    f"Terminando árbol de procesos...",
                    flush=True,
                )
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                else:
                    proc.kill()
            finally:
                if os.path.exists(summary_json_path):
                    try:
                        os.remove(summary_json_path)
                    except OSError:
                        pass
                _cleanup_orphan_sc2()
                time.sleep(2.0)

            if summary.get("steps_recorded", 0) > 0:
                break

        ep_sec = time.perf_counter() - ep_t0
        if summary.get("steps_recorded", 0) == 0:
            print(f"⚠️ [Supervisor] Episodio {ep:02d} omitido tras 2 intentos fallidos de SC2_x64.exe. Continuando...")
            continue

        completed_eps += 1
        won = int(summary.get("won", 0))
        total_wins += won
        recent_wins.append(won)
        if len(recent_wins) > 8:
            recent_wins.pop(0)

        rolling_wr = (sum(recent_wins) / len(recent_wins)) * 100.0
        overall_wr = (total_wins / max(1, completed_eps)) * 100.0
        elapsed_min = (time.perf_counter() - t_start) / 60.0

        print(
            f"📊 [Episodio {ep:02d}/{episodes:02d} | {ep_sec:.1f}s | Total {elapsed_min:.1f}m] "
            f"Mapa: {curr_map} vs {curr_enemy} ({curr_diff}) | "
            f"WinRate Global: {overall_wr:.1f}% ( Reciente: {rolling_wr:.1f}% ) | "
            f"Muestras Drive: {summary.get('total_buffer_samples', 0)}"
        )

        # Promoción automática de currículum si domina el nivel actual (>= 60% en últimas 5+ partidas)
        if len(recent_wins) >= 5 and rolling_wr >= 60.0 and curr_diff_idx < len(diff_ladder) - 1:
            curr_diff_idx += 1
            recent_wins.clear()
            print(f"🔥 ¡CURRÍCULUM SUPERADO! Subiendo dificultad automáticamente a: {diff_ladder[curr_diff_idx]}")

    print("============================================================================")
    print(f" 🎉 Recolección RL completada ({completed_eps} partidas, WinRate: {(total_wins/max(1,completed_eps))*100:.1f}%).")
    print(f" 👉 Ahora abre SC2_Hailo8_Transformer.ipynb en Google Colab y ejecuta la Celda RL.")
    print("============================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cliente y Bot SC2 100% IA Pura conectado a Raspberry Pi + Hailo-8")
    parser.add_argument("--rpi-ip", default="192.168.50.7", help="IP de la Raspberry Pi en la red local")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--play", action="store_true", help="Lanzar partida real de StarCraft II controlada por la RPi")
    parser.add_argument("--collect-rl", action="store_true", help="Ejecutar bucle acelerado de partidas RL y guardar en Drive")
    parser.add_argument("--single-rl-ep", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--summary-json", default="", help=argparse.SUPPRESS)
    parser.add_argument("--episodes", type=int, default=30, help="Número de partidas aceleradas a jugar en modo --collect-rl")
    parser.add_argument("--explore-eps", type=float, default=0.20, help="Probabilidad de exploración táctica en modo --collect-rl")
    parser.add_argument("--rollout-path", default=DEFAULT_DRIVE_ROLLOUT_PATH, help="Ruta del archivo .npz en Google Drive")
    parser.add_argument("--map", default="AbyssalReefLE", help="Nombre del mapa en StarCraft II/Maps")
    parser.add_argument("--race", default="Terran", choices=["Terran", "Protoss", "Zerg"])
    parser.add_argument("--enemy-race", default="Terran", choices=["Terran", "Protoss", "Zerg", "Random"])
    parser.add_argument("--difficulty", default="Medium", choices=["VeryEasy", "Easy", "Medium", "Hard", "VeryHard"])
    parser.add_argument("--realtime", action="store_true", help="Ejecutar en velocidad humana en vez de acelerada")
    parser.add_argument("--human", action="store_true", help="Jugar tú mismo (Humano) contra tu Raspberry Pi + Hailo-8")
    parser.add_argument("--max-steps", type=int, default=0, help="Límite opcional de pasos (0 = partida completa)")
    parser.add_argument("--steps", type=int, default=10, help="Pasos para modo benchmark sin abrir SC2")
    args = parser.parse_args()

    if args.single_rl_ep:
        res = launch_live_sc2_match(
            rpi_ip=args.rpi_ip,
            port=args.port,
            map_name=args.map,
            player_race=args.race,
            enemy_race=args.enemy_race,
            difficulty=args.difficulty,
            realtime=False,
            max_game_steps=args.max_steps if args.max_steps > 0 else 2400,
            human_vs_ai=False,
            collect_rl=True,
            explore_eps=args.explore_eps,
            rollout_path=args.rollout_path,
        )
        if args.summary_json:
            with open(args.summary_json, "w", encoding="utf-8") as jf:
                json.dump(res, jf)
    elif args.collect_rl:
        run_rl_collection_loop(
            rpi_ip=args.rpi_ip,
            port=args.port,
            player_race=args.race,
            start_difficulty=args.difficulty,
            episodes=args.episodes,
            max_game_steps=args.max_steps if args.max_steps > 0 else 2400,
            explore_eps=args.explore_eps,
            rollout_path=args.rollout_path,
        )
    elif args.play:
        launch_live_sc2_match(
            rpi_ip=args.rpi_ip,
            port=args.port,
            map_name=args.map,
            player_race=args.race,
            enemy_race=args.enemy_race,
            difficulty=args.difficulty,
            realtime=args.realtime,
            max_game_steps=args.max_steps,
            human_vs_ai=args.human,
        )
    else:
        run_lan_benchmark(args.rpi_ip, args.port, args.steps)
