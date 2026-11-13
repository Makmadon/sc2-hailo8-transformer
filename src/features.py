"""
Extracción unificada de características V2 y Action Masking estricto para Terran.
Compartida entre el bot en vivo (python-sc2) y el extractor de replays (.SC2Replay)
para garantizar cero domain-shift entre preentrenamiento y ejecución real.
"""

from typing import Optional, Tuple
import numpy as np

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


def get_playable_bounds(game_info) -> Tuple[float, float, float, float]:
    """Retorna (x0, y0, width, height) del área jugable del mapa."""
    area = getattr(game_info, "playable_area", None)
    if area is not None:
        return (
            float(area.x),
            float(area.y),
            max(1.0, float(area.width)),
            max(1.0, float(area.height)),
        )
    ms = game_info.map_size
    return 0.0, 0.0, max(1.0, float(ms.width)), max(1.0, float(ms.height))


def world_to_grid(pos, bounds: Tuple[float, float, float, float]) -> Tuple[int, int]:
    """Proyecta coordenadas del mundo SC2 al grid 32x32 dentro del área jugable."""
    x0, y0, w, h = bounds
    gx = int(np.clip(((float(pos.x) - x0) / w) * GRID_W, 0, GRID_W - 1))
    gy = int(np.clip(((float(pos.y) - y0) / h) * GRID_H, 0, GRID_H - 1))
    return gx, gy


def grid_to_world(gx: int, gy: int, bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Convierte una celda (gx, gy) del grid 32x32 al centro de esa celda en coordenadas SC2."""
    x0, y0, w, h = bounds
    wx = x0 + ((float(gx) + 0.5) / GRID_W) * w
    wy = y0 + ((float(gy) + 0.5) / GRID_H) * h
    wx = float(np.clip(wx, x0 + 2.0, x0 + w - 2.0))
    wy = float(np.clip(wy, y0 + 2.0, y0 + h - 2.0))
    return wx, wy


class EpisodeTrackerV2:
    """
    Memoria ligera por episodio para evitar depender de una LSTM dentro del chip Hailo-8:
    rastrea acciones recientes, tiempo sin combate y potencial PBRS (Φ(s)).
    """

    def __init__(self):
        self.last_macro_id: int = 0
        self.last_micro_id: int = 0
        self.steps_since_attack: int = 0
        self.ema_econ: float = 0.0
        self.ema_prod: float = 0.0
        self.ema_combat: float = 0.0
        self.ema_scout: float = 0.0
        self.lost_value_total: float = 0.0
        self.prev_phi: Optional[float] = None

    def update_action(self, macro_id: int, micro_id: int):
        self.last_macro_id = int(macro_id)
        self.last_micro_id = int(micro_id)

        decay = 0.90
        is_econ = 1.0 if macro_id in (1, 2, 3, 4, 5, 6, 14, 17) else 0.0
        is_prod = 1.0 if macro_id in range(7, 28) and macro_id not in (14, 17) else 0.0
        is_combat = 1.0 if micro_id in (4, 5, 6, 7, 8, 9) else 0.0
        is_scout = 1.0 if micro_id in (1, 2, 3) else 0.0

        self.ema_econ = decay * self.ema_econ + (1.0 - decay) * is_econ
        self.ema_prod = decay * self.ema_prod + (1.0 - decay) * is_prod
        self.ema_combat = decay * self.ema_combat + (1.0 - decay) * is_combat
        self.ema_scout = decay * self.ema_scout + (1.0 - decay) * is_scout

        if is_combat > 0.0:
            self.steps_since_attack = 0
        else:
            self.steps_since_attack = min(100, self.steps_since_attack + 1)


def _upgrade_progress(bot, upg_id) -> float:
    if upg_id in bot.state.upgrades:
        return 1.0
    try:
        if bot.already_pending_upgrade(upg_id) > 0.0:
            return 0.5
    except Exception:
        pass
    return 0.0


def _count_upgrade_level(bot, levels: Tuple) -> float:
    done = 0
    for upg in levels:
        if upg in bot.state.upgrades:
            done += 1
    return float(done) / max(1, len(levels))


def extract_state_and_masks(
    bot,
    tracker: Optional[EpisodeTrackerV2] = None,
    terrain_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extrae desde un BotAI de python-sc2:
      - map_tensor:  [1, 16, 32, 32] float32
      - global_vec:  [1, 64, 1, 1]  float32
      - macro_mask:  [28]           bool (True = acción válida)
      - micro_mask:  [12]           bool (True = acción válida)
      - terrain_cache reutilizable (pathable_grid, height_grid) en 32x32
    """
    from sc2.ids.unit_typeid import UnitTypeId
    from sc2.ids.upgrade_id import UpgradeId

    bounds = get_playable_bounds(bot.game_info)
    x0, y0, w, h = bounds

    map_tensor = np.zeros((1, MAP_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
    global_vec = np.zeros((1, GLOBAL_DIM, 1, 1), dtype=np.float32)

    # ------------------------------------------------------------------
    # 1. Terreno estático (Canal 14) y Niebla de guerra / Visibilidad (Canal 15)
    # ------------------------------------------------------------------
    if terrain_cache is None:
        path_grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        p_raw = bot.game_info.pathing_grid
        h_raw = bot.game_info.terrain_height
        for gy in range(GRID_H):
            wy = int(np.clip(y0 + ((gy + 0.5) / GRID_H) * h, 0, p_raw.height - 1))
            for gx in range(GRID_W):
                wx = int(np.clip(x0 + ((gx + 0.5) / GRID_W) * w, 0, p_raw.width - 1))
                is_p = 1.0 if p_raw[wx, wy] != 0 else 0.0
                if is_p > 0.0:
                    th_norm = float(np.clip(h_raw[wx, wy] / 255.0, 0.2, 1.0))
                    path_grid[gy, gx] = 0.25 + 0.75 * th_norm
        terrain_cache = (path_grid, np.zeros((GRID_H, GRID_W), dtype=np.float32))

    map_tensor[0, 14] = terrain_cache[0]

    vis_raw = getattr(bot.state, "visibility", None)
    if vis_raw is not None:
        for gy in range(GRID_H):
            wy = int(np.clip(y0 + ((gy + 0.5) / GRID_H) * h, 0, vis_raw.height - 1))
            for gx in range(GRID_W):
                wx = int(np.clip(x0 + ((gx + 0.5) / GRID_W) * w, 0, vis_raw.width - 1))
                v_val = vis_raw[wx, wy]
                map_tensor[0, 15, gy, gx] = 1.0 if v_val == 2 else (0.5 if v_val == 1 else 0.0)

    # ------------------------------------------------------------------
    # 2. Recursos: Minerales y Géiseres de Gas (Canal 13)
    # ------------------------------------------------------------------
    for mf in bot.mineral_field:
        gx, gy = world_to_grid(mf.position, bounds)
        map_tensor[0, 13, gy, gx] = min(1.0, map_tensor[0, 13, gy, gx] + 0.15)
    for vg in bot.vespene_geyser:
        gx, gy = world_to_grid(vg.position, bounds)
        map_tensor[0, 13, gy, gx] = min(1.0, map_tensor[0, 13, gy, gx] + 0.35)

    # ------------------------------------------------------------------
    # 3. Unidades Propias (Canales 0..4) con promedio real de HP
    # ------------------------------------------------------------------
    ally_hp_sum = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    ally_cnt = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    worker_types = {UnitTypeId.SCV, UnitTypeId.MULE}
    bio_types = {UnitTypeId.MARINE, UnitTypeId.MARAUDER, UnitTypeId.REAPER, UnitTypeId.GHOST}
    mech_types = {
        UnitTypeId.HELLION, UnitTypeId.HELLIONTANK, UnitTypeId.CYCLONE,
        UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED, UnitTypeId.THOR, UnitTypeId.THORAP,
    }
    air_types = {
        UnitTypeId.MEDIVAC, UnitTypeId.VIKINGFIGHTER, UnitTypeId.VIKINGASSAULT,
        UnitTypeId.LIBERATOR, UnitTypeId.LIBERATORAG, UnitTypeId.BANSHEE,
        UnitTypeId.RAVEN, UnitTypeId.BATTLECRUISER,
    }

    unit_counts = {
        "marine": 0, "marauder": 0, "reaper": 0, "ghost": 0,
        "hellion": 0, "tank": 0, "thor": 0, "medivac": 0, "air_combat": 0, "bc_raven": 0,
    }

    for u in bot.units:
        gx, gy = world_to_grid(u.position, bounds)
        t = u.type_id
        ally_hp_sum[gy, gx] += float(u.health_percentage)
        ally_cnt[gy, gx] += 1.0

        if t in worker_types:
            map_tensor[0, 0, gy, gx] = min(1.0, map_tensor[0, 0, gy, gx] + 1.0 / 12.0)
        elif t in bio_types:
            map_tensor[0, 1, gy, gx] = min(1.0, map_tensor[0, 1, gy, gx] + 1.0 / 12.0)
            if t == UnitTypeId.MARINE:
                unit_counts["marine"] += 1
            elif t == UnitTypeId.MARAUDER:
                unit_counts["marauder"] += 1
            elif t == UnitTypeId.REAPER:
                unit_counts["reaper"] += 1
            elif t == UnitTypeId.GHOST:
                unit_counts["ghost"] += 1
        elif t in mech_types:
            map_tensor[0, 2, gy, gx] = min(1.0, map_tensor[0, 2, gy, gx] + 1.0 / 6.0)
            if t in (UnitTypeId.HELLION, UnitTypeId.HELLIONTANK):
                unit_counts["hellion"] += 1
            elif t in (UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED):
                unit_counts["tank"] += 1
            else:
                unit_counts["thor"] += 1
        elif t in air_types:
            map_tensor[0, 3, gy, gx] = min(1.0, map_tensor[0, 3, gy, gx] + 1.0 / 6.0)
            if t == UnitTypeId.MEDIVAC:
                unit_counts["medivac"] += 1
            elif t in (UnitTypeId.BATTLECRUISER, UnitTypeId.RAVEN):
                unit_counts["bc_raven"] += 1
            else:
                unit_counts["air_combat"] += 1

    valid_ally = ally_cnt > 0
    map_tensor[0, 4][valid_ally] = ally_hp_sum[valid_ally] / ally_cnt[valid_ally]

    # ------------------------------------------------------------------
    # 4. Estructuras Propias (Canales 5..7) y Conteos de Tech Tree
    # ------------------------------------------------------------------
    th_types = {
        UnitTypeId.COMMANDCENTER, UnitTypeId.COMMANDCENTERFLYING,
        UnitTypeId.ORBITALCOMMAND, UnitTypeId.ORBITALCOMMANDFLYING,
        UnitTypeId.PLANETARYFORTRESS,
    }
    prod_types = {
        UnitTypeId.BARRACKS, UnitTypeId.BARRACKSFLYING, UnitTypeId.BARRACKSTECHLAB, UnitTypeId.BARRACKSREACTOR,
        UnitTypeId.FACTORY, UnitTypeId.FACTORYFLYING, UnitTypeId.FACTORYTECHLAB, UnitTypeId.FACTORYREACTOR,
        UnitTypeId.STARPORT, UnitTypeId.STARPORTFLYING, UnitTypeId.STARPORTTECHLAB, UnitTypeId.STARPORTREACTOR,
        UnitTypeId.ENGINEERINGBAY, UnitTypeId.ARMORY, UnitTypeId.GHOSTACADEMY, UnitTypeId.FUSIONCORE,
    }
    sup_def_types = {
        UnitTypeId.SUPPLYDEPOT, UnitTypeId.SUPPLYDEPOTLOWERED, UnitTypeId.REFINERY,
        UnitTypeId.BUNKER, UnitTypeId.MISSILETURRET, UnitTypeId.SENSORTOWER,
    }

    s_counts = {
        "cc_all": 0, "orbital": 0, "depot": 0, "refinery": 0,
        "rax": 0, "rax_tl": 0, "rax_re": 0,
        "fac": 0, "fac_tl": 0, "sp": 0,
        "ebay": 0, "armory": 0, "def": 0, "high_tech": 0,
    }
    orbital_energy_ready = False

    for s in bot.structures:
        gx, gy = world_to_grid(s.position, bounds)
        t = s.type_id
        w_build = 1.0 if s.is_ready else 0.5
        if t in th_types:
            map_tensor[0, 5, gy, gx] = min(1.0, map_tensor[0, 5, gy, gx] + 0.5 * w_build)
            s_counts["cc_all"] += 1
            if t in (UnitTypeId.ORBITALCOMMAND, UnitTypeId.ORBITALCOMMANDFLYING):
                s_counts["orbital"] += 1
                if s.is_ready and getattr(s, "energy", 0) >= 50:
                    orbital_energy_ready = True
        elif t in prod_types:
            map_tensor[0, 6, gy, gx] = min(1.0, map_tensor[0, 6, gy, gx] + 0.25 * w_build)
            if t in (UnitTypeId.BARRACKS, UnitTypeId.BARRACKSFLYING):
                s_counts["rax"] += 1
            elif t == UnitTypeId.BARRACKSTECHLAB:
                s_counts["rax_tl"] += 1
            elif t == UnitTypeId.BARRACKSREACTOR:
                s_counts["rax_re"] += 1
            elif t in (UnitTypeId.FACTORY, UnitTypeId.FACTORYFLYING):
                s_counts["fac"] += 1
            elif t == UnitTypeId.FACTORYTECHLAB:
                s_counts["fac_tl"] += 1
            elif t in (UnitTypeId.STARPORT, UnitTypeId.STARPORTFLYING):
                s_counts["sp"] += 1
            elif t == UnitTypeId.ENGINEERINGBAY:
                s_counts["ebay"] += 1
            elif t == UnitTypeId.ARMORY:
                s_counts["armory"] += 1
            else:
                s_counts["high_tech"] += 1
        elif t in sup_def_types:
            map_tensor[0, 7, gy, gx] = min(1.0, map_tensor[0, 7, gy, gx] + 0.25 * w_build)
            if t in (UnitTypeId.SUPPLYDEPOT, UnitTypeId.SUPPLYDEPOTLOWERED):
                s_counts["depot"] += 1
            elif t == UnitTypeId.REFINERY:
                s_counts["refinery"] += 1
            else:
                s_counts["def"] += 1

    # ------------------------------------------------------------------
    # 5. Enemigos Visibles (Canales 8..12) con promedio real de HP
    # ------------------------------------------------------------------
    enemy_hp_sum = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    enemy_cnt = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    enemy_worker_ids = {UnitTypeId.SCV, UnitTypeId.PROBE, UnitTypeId.DRONE, UnitTypeId.MULE}
    visible_enemy_army = 0

    for eu in bot.enemy_units:
        gx, gy = world_to_grid(eu.position, bounds)
        hp_frac = float(eu.health + getattr(eu, "shield", 0.0)) / max(1.0, float(eu.health_max + getattr(eu, "shield_max", 0.0)))
        enemy_hp_sum[gy, gx] += float(np.clip(hp_frac, 0.0, 1.0))
        enemy_cnt[gy, gx] += 1.0

        if eu.type_id in enemy_worker_ids:
            map_tensor[0, 8, gy, gx] = min(1.0, map_tensor[0, 8, gy, gx] + 1.0 / 12.0)
        elif getattr(eu, "is_flying", False):
            map_tensor[0, 10, gy, gx] = min(1.0, map_tensor[0, 10, gy, gx] + 1.0 / 8.0)
            visible_enemy_army += 2
        else:
            map_tensor[0, 9, gy, gx] = min(1.0, map_tensor[0, 9, gy, gx] + 1.0 / 10.0)
            visible_enemy_army += 1

    for es in bot.enemy_structures:
        gx, gy = world_to_grid(es.position, bounds)
        hp_frac = float(es.health + getattr(es, "shield", 0.0)) / max(1.0, float(es.health_max + getattr(es, "shield_max", 0.0)))
        enemy_hp_sum[gy, gx] += float(np.clip(hp_frac, 0.0, 1.0))
        enemy_cnt[gy, gx] += 1.0
        map_tensor[0, 11, gy, gx] = min(1.0, map_tensor[0, 11, gy, gx] + 0.25)

    valid_enemy = enemy_cnt > 0
    map_tensor[0, 12][valid_enemy] = enemy_hp_sum[valid_enemy] / enemy_cnt[valid_enemy]

    # ------------------------------------------------------------------
    # 6. Vector Global [1, 64, 1, 1]
    # ------------------------------------------------------------------
    gv = global_vec[0, :, 0, 0]
    minerals = float(bot.minerals)
    vespene = float(bot.vespene)
    sup_used = float(bot.supply_used)
    sup_cap = float(bot.supply_cap)
    sup_left = float(bot.supply_left)

    gv[0] = np.clip(minerals / 2000.0, 0.0, 1.0)
    gv[1] = np.clip(vespene / 1500.0, 0.0, 1.0)
    gv[2] = np.clip(sup_used / 200.0, 0.0, 1.0)
    gv[3] = np.clip(sup_cap / 200.0, 0.0, 1.0)
    gv[4] = np.clip(sup_left / 20.0, 0.0, 1.0)
    gv[5] = np.clip(float(bot.supply_workers) / 80.0, 0.0, 1.0)
    gv[6] = np.clip(float(bot.supply_army) / 150.0, 0.0, 1.0)
    gv[7] = np.clip(float(bot.idle_worker_count) / 10.0, 0.0, 1.0)

    gv[8] = np.clip(float(bot.time) / 1800.0, 0.0, 1.0)
    gv[9] = np.clip(float(bot.state.score.collection_rate_minerals) / 2500.0, 0.0, 1.0)
    gv[10] = np.clip(float(bot.state.score.collection_rate_vespene) / 1200.0, 0.0, 1.0)
    erace_id = ENEMY_RACES.get(getattr(bot.enemy_race, "name", "Terran"), 0)
    if erace_id in (0, 1, 2):
        gv[11 + erace_id] = 1.0

    gv[14] = np.clip(s_counts["cc_all"] / 5.0, 0.0, 1.0)
    gv[15] = np.clip(s_counts["orbital"] / 4.0, 0.0, 1.0)
    gv[16] = np.clip(s_counts["depot"] / 20.0, 0.0, 1.0)
    gv[17] = np.clip(s_counts["refinery"] / 8.0, 0.0, 1.0)
    gv[18] = np.clip(s_counts["rax"] / 10.0, 0.0, 1.0)
    gv[19] = np.clip(s_counts["rax_tl"] / 6.0, 0.0, 1.0)
    gv[20] = np.clip(s_counts["rax_re"] / 6.0, 0.0, 1.0)
    gv[21] = np.clip(s_counts["fac"] / 5.0, 0.0, 1.0)
    gv[22] = np.clip(s_counts["fac_tl"] / 4.0, 0.0, 1.0)
    gv[23] = np.clip(s_counts["sp"] / 4.0, 0.0, 1.0)
    gv[24] = np.clip(s_counts["ebay"] / 2.0, 0.0, 1.0)
    gv[25] = np.clip(s_counts["armory"] / 2.0, 0.0, 1.0)
    gv[26] = np.clip(s_counts["def"] / 6.0, 0.0, 1.0)
    gv[27] = np.clip(s_counts["high_tech"] / 2.0, 0.0, 1.0)

    gv[28] = np.clip(unit_counts["marine"] / 60.0, 0.0, 1.0)
    gv[29] = np.clip(unit_counts["marauder"] / 30.0, 0.0, 1.0)
    gv[30] = np.clip(unit_counts["reaper"] / 10.0, 0.0, 1.0)
    gv[31] = np.clip(unit_counts["ghost"] / 10.0, 0.0, 1.0)
    gv[32] = np.clip(unit_counts["hellion"] / 20.0, 0.0, 1.0)
    gv[33] = np.clip(unit_counts["tank"] / 16.0, 0.0, 1.0)
    gv[34] = np.clip(unit_counts["thor"] / 6.0, 0.0, 1.0)
    gv[35] = np.clip(unit_counts["medivac"] / 10.0, 0.0, 1.0)
    gv[36] = np.clip(unit_counts["air_combat"] / 16.0, 0.0, 1.0)
    gv[37] = np.clip(unit_counts["bc_raven"] / 6.0, 0.0, 1.0)

    idle_th = bot.townhalls.ready.idle.amount
    idle_rax = bot.structures(UnitTypeId.BARRACKS).ready.idle.amount
    idle_fac = bot.structures(UnitTypeId.FACTORY).ready.idle.amount
    idle_sp = bot.structures(UnitTypeId.STARPORT).ready.idle.amount

    gv[38] = np.clip(idle_th / 4.0, 0.0, 1.0)
    gv[39] = np.clip(idle_rax / 8.0, 0.0, 1.0)
    gv[40] = np.clip(idle_fac / 4.0, 0.0, 1.0)
    gv[41] = np.clip(idle_sp / 4.0, 0.0, 1.0)
    gv[42] = 1.0 if bot.already_pending(UnitTypeId.COMMANDCENTER) else 0.0
    gv[43] = 1.0 if bot.already_pending(UnitTypeId.SUPPLYDEPOT) else 0.0
    pend_prod = (
        bot.already_pending(UnitTypeId.BARRACKS)
        + bot.already_pending(UnitTypeId.FACTORY)
        + bot.already_pending(UnitTypeId.STARPORT)
    )
    gv[44] = np.clip(float(pend_prod) / 4.0, 0.0, 1.0)
    gv[45] = 1.0 if orbital_energy_ready else 0.0

    gv[46] = _upgrade_progress(bot, UpgradeId.STIMPACK)
    gv[47] = _upgrade_progress(bot, UpgradeId.SHIELDWALL)
    gv[48] = _upgrade_progress(bot, UpgradeId.PUNISHERGRENADES)
    gv[49] = _count_upgrade_level(bot, (
        UpgradeId.TERRANINFANTRYWEAPONSLEVEL1,
        UpgradeId.TERRANINFANTRYWEAPONSLEVEL2,
        UpgradeId.TERRANINFANTRYWEAPONSLEVEL3,
    ))
    gv[50] = _count_upgrade_level(bot, (
        UpgradeId.TERRANINFANTRYARMORSLEVEL1,
        UpgradeId.TERRANINFANTRYARMORSLEVEL2,
        UpgradeId.TERRANINFANTRYARMORSLEVEL3,
    ))
    gv[51] = _count_upgrade_level(bot, (
        UpgradeId.TERRANVEHICLEWEAPONSLEVEL1,
        UpgradeId.TERRANVEHICLEWEAPONSLEVEL2,
        UpgradeId.TERRANVEHICLEWEAPONSLEVEL3,
        UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL1,
        UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL2,
        UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL3,
    ))
    gv[52] = _count_upgrade_level(bot, (
        UpgradeId.TERRANSHIPWEAPONSLEVEL1,
        UpgradeId.BANSHEECLOAK,
    ))

    under_attack = False
    if bot.enemy_units:
        for eu in bot.enemy_units:
            if any(s.distance_to(eu) <= 22.0 for s in bot.structures) or any(w.distance_to(eu) <= 14.0 for w in bot.workers):
                under_attack = True
                break
    gv[53] = 1.0 if under_attack else 0.0

    score = bot.state.score
    killed_val = float(score.killed_value_units + score.killed_value_structures)
    lost_val = tracker.lost_value_total if tracker is not None else 0.0
    army_val = float(score.total_value_units)

    gv[54] = np.clip(visible_enemy_army / 60.0, 0.0, 1.0)
    gv[55] = np.clip(army_val / 5000.0, 0.0, 1.0)
    gv[56] = float(0.5 + 0.5 * np.tanh((killed_val - lost_val) / 1000.0))

    if tracker is not None:
        gv[57] = float(tracker.last_macro_id) / max(1, NUM_MACRO)
        gv[58] = float(tracker.last_micro_id) / max(1, NUM_MICRO)
        gv[59] = np.clip(float(tracker.steps_since_attack) / 100.0, 0.0, 1.0)
        gv[60] = float(np.clip(tracker.ema_econ, 0.0, 1.0))
        gv[61] = float(np.clip(tracker.ema_prod, 0.0, 1.0))
        gv[62] = float(np.clip(tracker.ema_combat, 0.0, 1.0))
        gv[63] = float(np.clip(tracker.ema_scout, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 7. Action Masks Estricto (Macro [28], Micro [12])
    # ------------------------------------------------------------------
    macro_mask = np.zeros(NUM_MACRO, dtype=bool)
    micro_mask = np.zeros(NUM_MICRO, dtype=bool)
    macro_mask[MACRO_TO_ID["MACRO_NO_OP"]] = True
    micro_mask[MICRO_TO_ID["MICRO_NO_OP"]] = True

    has_workers = bot.workers.amount > 0
    ready_th = bot.townhalls.ready
    ready_rax = bot.structures(UnitTypeId.BARRACKS).ready
    ready_fac = bot.structures(UnitTypeId.FACTORY).ready
    ready_sp = bot.structures(UnitTypeId.STARPORT).ready
    ready_depot = bot.structures({UnitTypeId.SUPPLYDEPOT, UnitTypeId.SUPPLYDEPOTLOWERED}).ready
    ready_ebay = bot.structures(UnitTypeId.ENGINEERINGBAY).ready
    ready_armory = bot.structures(UnitTypeId.ARMORY).ready

    # Economía y suministro
    if has_workers and bot.mineral_field:
        has_surplus = any(th.surplus_harvesters > 0 for th in ready_th)
        if bot.idle_worker_count > 0 or has_surplus:
            macro_mask[MACRO_TO_ID["ASSIGN_MINERALS"]] = True

    if has_workers and any(g.assigned_harvesters < g.ideal_harvesters for g in bot.gas_buildings.ready):
        macro_mask[MACRO_TO_ID["ASSIGN_GAS"]] = True

    if ready_th and minerals >= 50 and sup_left >= 1 and any(len(th.orders) < 2 for th in ready_th):
        macro_mask[MACRO_TO_ID["TRAIN_SCV"]] = True

    if has_workers and minerals >= 100 and sup_cap < 200:
        macro_mask[MACRO_TO_ID["BUILD_SUPPLY_DEPOT"]] = True

    if has_workers and ready_th and minerals >= 75:
        for th in ready_th:
            if any(not bot.gas_buildings.closer_than(1.0, vg) for vg in bot.vespene_geyser.closer_than(10.0, th)):
                macro_mask[MACRO_TO_ID["BUILD_REFINERY"]] = True
                break

    if has_workers and minerals >= 400:
        all_s = bot.structures + bot.enemy_structures
        if any(not all_s.closer_than(6.0, loc) for loc in bot.expansion_locations_list):
            macro_mask[MACRO_TO_ID["BUILD_COMMAND_CENTER"]] = True

    # Estructuras Terran
    if has_workers and ready_depot and minerals >= 150:
        macro_mask[MACRO_TO_ID["BUILD_BARRACKS"]] = True
        macro_mask[MACRO_TO_ID["BUILD_BUNKER"]] = True
    if has_workers and ready_rax and minerals >= 150 and vespene >= 100:
        macro_mask[MACRO_TO_ID["BUILD_FACTORY"]] = True
    if has_workers and ready_fac and minerals >= 150 and vespene >= 100:
        macro_mask[MACRO_TO_ID["BUILD_STARPORT"]] = True
    if has_workers and ready_th and minerals >= 125:
        macro_mask[MACRO_TO_ID["BUILD_ENGINEERING_BAY"]] = True
    if has_workers and ready_fac and minerals >= 150 and vespene >= 100:
        macro_mask[MACRO_TO_ID["BUILD_ARMORY"]] = True
    if has_workers and ready_ebay and minerals >= 100:
        macro_mask[MACRO_TO_ID["BUILD_MISSILE_TURRET"]] = True

    # Orbital / Addons / MULE
    if ready_rax and minerals >= 150 and bot.townhalls(UnitTypeId.COMMANDCENTER).ready.idle:
        macro_mask[MACRO_TO_ID["UPGRADE_ORBITAL_COMMAND"]] = True

    prod_no_addon = [
        p for p in (ready_rax + ready_fac + ready_sp)
        if p.is_idle and not p.has_add_on
    ]
    if prod_no_addon and minerals >= 50 and vespene >= 25:
        macro_mask[MACRO_TO_ID["BUILD_ADDON_TECHLAB"]] = True
    if prod_no_addon and minerals >= 50 and vespene >= 50:
        macro_mask[MACRO_TO_ID["BUILD_ADDON_REACTOR"]] = True

    if orbital_energy_ready and bot.mineral_field:
        macro_mask[MACRO_TO_ID["CALLDOWN_MULE"]] = True

    # Entrenamiento de unidades Terran (verificando recursos, tech, addon y supply)
    def has_free_slot(building, allow_reactor: bool = True) -> bool:
        cap = 2 if (allow_reactor and getattr(building, "has_reactor", False)) else 1
        return len(building.orders) < cap

    if ready_rax and sup_left >= 1 and minerals >= 50:
        if any(has_free_slot(r) for r in ready_rax):
            macro_mask[MACRO_TO_ID["TRAIN_MARINE"]] = True
        if vespene >= 50 and any(has_free_slot(r, allow_reactor=False) for r in ready_rax):
            macro_mask[MACRO_TO_ID["TRAIN_REAPER"]] = True

    if ready_rax and sup_left >= 2 and minerals >= 100 and vespene >= 25:
        if any(getattr(r, "has_techlab", False) and has_free_slot(r, allow_reactor=False) for r in ready_rax):
            macro_mask[MACRO_TO_ID["TRAIN_MARAUDER"]] = True

    if ready_fac and sup_left >= 2 and minerals >= 100:
        if any(has_free_slot(f) for f in ready_fac):
            macro_mask[MACRO_TO_ID["TRAIN_HELLION"]] = True

    if ready_fac and sup_left >= 3 and minerals >= 150 and vespene >= 125:
        if any(getattr(f, "has_techlab", False) and has_free_slot(f, allow_reactor=False) for f in ready_fac):
            macro_mask[MACRO_TO_ID["TRAIN_SIEGE_TANK"]] = True

    if ready_fac and sup_left >= 3 and minerals >= 150 and vespene >= 100:
        has_tl_fac = any(getattr(f, "has_techlab", False) and has_free_slot(f, allow_reactor=False) for f in ready_fac)
        if has_tl_fac and ((ready_armory and minerals >= 300 and vespene >= 200 and sup_left >= 6) or True):
            macro_mask[MACRO_TO_ID["TRAIN_THOR_OR_CYCLONE"]] = True

    if ready_sp and sup_left >= 2 and minerals >= 100 and vespene >= 100:
        if any(has_free_slot(sp) for sp in ready_sp):
            macro_mask[MACRO_TO_ID["TRAIN_MEDIVAC"]] = True

    if ready_sp and sup_left >= 2 and minerals >= 150 and vespene >= 75:
        if any(has_free_slot(sp) for sp in ready_sp):
            macro_mask[MACRO_TO_ID["TRAIN_VIKING_OR_LIBERATOR"]] = True

    # Investigaciones
    for tl in bot.structures(UnitTypeId.BARRACKSTECHLAB).ready.idle:
        for upg, m_cost, g_cost in (
            (UpgradeId.STIMPACK, 100, 100),
            (UpgradeId.SHIELDWALL, 100, 100),
            (UpgradeId.PUNISHERGRENADES, 50, 50),
        ):
            if upg not in bot.state.upgrades and minerals >= m_cost and vespene >= g_cost:
                macro_mask[MACRO_TO_ID["RESEARCH_BIO_TECH"]] = True
                break

    has_ebay_upg = False
    if ready_ebay.idle and minerals >= 100 and vespene >= 100:
        for upg in (
            UpgradeId.TERRANINFANTRYWEAPONSLEVEL1,
            UpgradeId.TERRANINFANTRYARMORSLEVEL1,
            UpgradeId.TERRANINFANTRYWEAPONSLEVEL2,
            UpgradeId.TERRANINFANTRYARMORSLEVEL2,
            UpgradeId.TERRANINFANTRYWEAPONSLEVEL3,
            UpgradeId.TERRANINFANTRYARMORSLEVEL3,
        ):
            if upg not in bot.state.upgrades:
                # Nivel 2 y 3 requieren Armory
                needs_armory = upg in (
                    UpgradeId.TERRANINFANTRYWEAPONSLEVEL2,
                    UpgradeId.TERRANINFANTRYARMORSLEVEL2,
                    UpgradeId.TERRANINFANTRYWEAPONSLEVEL3,
                    UpgradeId.TERRANINFANTRYARMORSLEVEL3,
                )
                if not needs_armory or ready_armory:
                    has_ebay_upg = True
                    break

    has_armory_upg = False
    if ready_armory.idle and minerals >= 100 and vespene >= 100:
        for upg in (
            UpgradeId.TERRANVEHICLEWEAPONSLEVEL1,
            UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL1,
            UpgradeId.TERRANSHIPWEAPONSLEVEL1,
        ):
            if upg not in bot.state.upgrades:
                has_armory_upg = True
                break

    if has_ebay_upg or has_armory_upg:
        macro_mask[MACRO_TO_ID["RESEARCH_UPGRADES"]] = True

    # Micro Masks
    combat_units = bot.units.exclude_type(worker_types)
    has_army = combat_units.amount > 0

    if has_workers or unit_counts["reaper"] > 0:
        micro_mask[MICRO_TO_ID["SCOUT_WITH_SCV_OR_REAPER"]] = True

    if orbital_energy_ready:
        micro_mask[MICRO_TO_ID["SCANNER_SWEEP"]] = True

    if has_army:
        micro_mask[MICRO_TO_ID["MOVE_ARMY"]] = True
        micro_mask[MICRO_TO_ID["ATTACK_MOVE_ARMY"]] = True
        micro_mask[MICRO_TO_ID["KITE_RETREAT_ARMY"]] = True
        if bot.enemy_units or bot.enemy_structures:
            micro_mask[MICRO_TO_ID["FOCUS_FIRE_AREA"]] = True

    if UpgradeId.STIMPACK in bot.state.upgrades and bot.enemy_units:
        bio_ready = [
            u for u in bot.units({UnitTypeId.MARINE, UnitTypeId.MARAUDER})
            if u.health_percentage > 0.55 and bot.enemy_units.closer_than(14.0, u)
        ]
        if bio_ready:
            micro_mask[MICRO_TO_ID["USE_STIMPACK"]] = True

    if bot.units(UnitTypeId.SIEGETANK):
        micro_mask[MICRO_TO_ID["SIEGE_TANKS_AT_AREA"]] = True
    if bot.units(UnitTypeId.SIEGETANKSIEGED):
        micro_mask[MICRO_TO_ID["UNSIEGE_TANKS_TO_MOVE"]] = True

    if has_workers:
        if any(u.is_mechanical and u.health_percentage < 1.0 for u in (bot.units + bot.structures)):
            micro_mask[MICRO_TO_ID["REPAIR_WITH_SCV"]] = True

    if ready_depot:
        micro_mask[MICRO_TO_ID["LOWER_OR_RAISE_DEPOTS"]] = True

    return map_tensor, global_vec, macro_mask, micro_mask, terrain_cache
