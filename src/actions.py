"""
Ejecutor Dual Terran V2 (Macro [28] + Micro [12]) para StarCraft II (python-sc2).
Cada acción se ejecuta limpiamente sobre la coordenada espacial (gx, gy) predicha por
el heatmap 32x32 correspondiente, sin reglas de fallback encadenadas que oculten errores.
"""

from typing import Optional
import numpy as np

from .config import ID_TO_MACRO, ID_TO_MICRO
from .features import get_playable_bounds, grid_to_world


class TerranActionExecutorV2:
    """Traduce (macro_id, macro_xy) y (micro_id, micro_xy) en comandos reales de python-sc2."""

    def __init__(self):
        self.scout_worker_tag: Optional[int] = None

    def _to_point2(self, bot, gx: int, gy: int):
        from sc2.position import Point2

        bounds = get_playable_bounds(bot.game_info)
        wx, wy = grid_to_world(gx, gy, bounds)
        return Point2((wx, wy))

    async def execute_step(
        self,
        bot,
        macro_id: int,
        macro_gx: int,
        macro_gy: int,
        micro_id: int,
        micro_gx: int,
        micro_gy: int,
    ):
        await self.execute_macro(bot, macro_id, macro_gx, macro_gy)
        await self.execute_micro(bot, micro_id, micro_gx, micro_gy)

    async def execute_macro(self, bot, macro_id: int, gx: int, gy: int):
        from sc2.ids.ability_id import AbilityId
        from sc2.ids.unit_typeid import UnitTypeId
        from sc2.ids.upgrade_id import UpgradeId

        act = ID_TO_MACRO.get(int(macro_id), "MACRO_NO_OP")
        if act == "MACRO_NO_OP":
            return

        target_pt = self._to_point2(bot, gx, gy)

        # [01] ASSIGN_MINERALS
        if act == "ASSIGN_MINERALS":
            await bot.distribute_workers()

        # [02] ASSIGN_GAS
        elif act == "ASSIGN_GAS":
            for gas_b in bot.gas_buildings.ready:
                if gas_b.assigned_harvesters < gas_b.ideal_harvesters and bot.workers:
                    w = bot.workers.closest_to(gas_b)
                    w.gather(gas_b)
                    break

        # [03] TRAIN_SCV
        elif act == "TRAIN_SCV":
            if bot.can_afford(UnitTypeId.SCV):
                for th in bot.townhalls.ready:
                    if len(th.orders) < 2:
                        th.train(UnitTypeId.SCV)
                        break

        # [04] BUILD_SUPPLY_DEPOT
        elif act == "BUILD_SUPPLY_DEPOT":
            if bot.supply_cap < 200 and bot.can_afford(UnitTypeId.SUPPLYDEPOT):
                await bot.build(UnitTypeId.SUPPLYDEPOT, near=target_pt)

        # [05] BUILD_REFINERY
        elif act == "BUILD_REFINERY":
            if bot.can_afford(UnitTypeId.REFINERY) and bot.townhalls.ready:
                candidates = []
                for th in bot.townhalls.ready:
                    for vg in bot.vespene_geyser.closer_than(10.0, th):
                        if not bot.gas_buildings.closer_than(1.0, vg):
                            candidates.append(vg)
                if candidates:
                    chosen = min(candidates, key=lambda v: v.position.distance_to(target_pt))
                    w = bot.select_build_worker(chosen.position)
                    if w:
                        w.build_gas(chosen)

        # [06] BUILD_COMMAND_CENTER
        elif act == "BUILD_COMMAND_CENTER":
            if bot.can_afford(UnitTypeId.COMMANDCENTER):
                all_s = bot.structures + bot.enemy_structures
                free_exps = [
                    loc for loc in bot.expansion_locations_list
                    if not all_s.closer_than(6.0, loc)
                ]
                if free_exps:
                    chosen = min(free_exps, key=lambda loc: loc.distance_to(target_pt))
                    await bot.build(
                        UnitTypeId.COMMANDCENTER,
                        near=chosen,
                        max_distance=6,
                        random_alternative=False,
                        placement_step=1,
                    )

        # [07..13] Construcción de estructuras Terran en coordenada (gx, gy)
        elif act == "BUILD_BARRACKS":
            if bot.can_afford(UnitTypeId.BARRACKS):
                await bot.build(UnitTypeId.BARRACKS, near=target_pt)

        elif act == "BUILD_FACTORY":
            if bot.can_afford(UnitTypeId.FACTORY):
                await bot.build(UnitTypeId.FACTORY, near=target_pt)

        elif act == "BUILD_STARPORT":
            if bot.can_afford(UnitTypeId.STARPORT):
                await bot.build(UnitTypeId.STARPORT, near=target_pt)

        elif act == "BUILD_ENGINEERING_BAY":
            if bot.can_afford(UnitTypeId.ENGINEERINGBAY):
                await bot.build(UnitTypeId.ENGINEERINGBAY, near=target_pt)

        elif act == "BUILD_ARMORY":
            if bot.can_afford(UnitTypeId.ARMORY):
                await bot.build(UnitTypeId.ARMORY, near=target_pt)

        elif act == "BUILD_BUNKER":
            if bot.can_afford(UnitTypeId.BUNKER):
                await bot.build(UnitTypeId.BUNKER, near=target_pt)

        elif act == "BUILD_MISSILE_TURRET":
            if bot.can_afford(UnitTypeId.MISSILETURRET):
                await bot.build(UnitTypeId.MISSILETURRET, near=target_pt)

        # [14] UPGRADE_ORBITAL_COMMAND
        elif act == "UPGRADE_ORBITAL_COMMAND":
            ccs = bot.townhalls(UnitTypeId.COMMANDCENTER).ready.idle
            if ccs and bot.can_afford(UnitTypeId.ORBITALCOMMAND):
                chosen_cc = ccs.closest_to(target_pt)
                chosen_cc(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND)

        # [15] BUILD_ADDON_TECHLAB
        elif act == "BUILD_ADDON_TECHLAB":
            candidates = [
                b for b in bot.structures({UnitTypeId.BARRACKS, UnitTypeId.FACTORY, UnitTypeId.STARPORT}).ready.idle
                if not b.has_add_on
            ]
            if candidates:
                b = min(candidates, key=lambda s: s.distance_to(target_pt))
                addon_type = (
                    UnitTypeId.BARRACKSTECHLAB if b.type_id == UnitTypeId.BARRACKS
                    else (UnitTypeId.FACTORYTECHLAB if b.type_id == UnitTypeId.FACTORY else UnitTypeId.STARPORTTECHLAB)
                )
                if bot.can_afford(addon_type):
                    b.build(addon_type)

        # [16] BUILD_ADDON_REACTOR
        elif act == "BUILD_ADDON_REACTOR":
            candidates = [
                b for b in bot.structures({UnitTypeId.BARRACKS, UnitTypeId.FACTORY, UnitTypeId.STARPORT}).ready.idle
                if not b.has_add_on
            ]
            if candidates:
                b = min(candidates, key=lambda s: s.distance_to(target_pt))
                addon_type = (
                    UnitTypeId.BARRACKSREACTOR if b.type_id == UnitTypeId.BARRACKS
                    else (UnitTypeId.FACTORYREACTOR if b.type_id == UnitTypeId.FACTORY else UnitTypeId.STARPORTREACTOR)
                )
                if bot.can_afford(addon_type):
                    b.build(addon_type)

        # [17] CALLDOWN_MULE
        elif act == "CALLDOWN_MULE":
            if bot.mineral_field:
                for oc in bot.townhalls(UnitTypeId.ORBITALCOMMAND).ready:
                    if oc.energy >= 50:
                        mf = bot.mineral_field.closest_to(target_pt)
                        oc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, mf)
                        break

        # [18..25] Producción de unidades Terran
        elif act == "TRAIN_MARINE":
            for rax in bot.structures(UnitTypeId.BARRACKS).ready:
                cap = 2 if getattr(rax, "has_reactor", False) else 1
                if len(rax.orders) < cap and bot.can_afford(UnitTypeId.MARINE):
                    rax.train(UnitTypeId.MARINE)
                    break

        elif act == "TRAIN_MARAUDER":
            for rax in bot.structures(UnitTypeId.BARRACKS).ready:
                if getattr(rax, "has_techlab", False) and len(rax.orders) < 1 and bot.can_afford(UnitTypeId.MARAUDER):
                    rax.train(UnitTypeId.MARAUDER)
                    break

        elif act == "TRAIN_REAPER":
            for rax in bot.structures(UnitTypeId.BARRACKS).ready:
                if len(rax.orders) < 1 and bot.can_afford(UnitTypeId.REAPER):
                    rax.train(UnitTypeId.REAPER)
                    break

        elif act == "TRAIN_HELLION":
            for fac in bot.structures(UnitTypeId.FACTORY).ready:
                cap = 2 if getattr(fac, "has_reactor", False) else 1
                if len(fac.orders) < cap and bot.can_afford(UnitTypeId.HELLION):
                    fac.train(UnitTypeId.HELLION)
                    break

        elif act == "TRAIN_SIEGE_TANK":
            for fac in bot.structures(UnitTypeId.FACTORY).ready:
                if getattr(fac, "has_techlab", False) and len(fac.orders) < 1 and bot.can_afford(UnitTypeId.SIEGETANK):
                    fac.train(UnitTypeId.SIEGETANK)
                    break

        elif act == "TRAIN_THOR_OR_CYCLONE":
            has_armory = bot.structures(UnitTypeId.ARMORY).ready.exists
            for fac in bot.structures(UnitTypeId.FACTORY).ready:
                if getattr(fac, "has_techlab", False) and len(fac.orders) < 1:
                    if has_armory and bot.can_afford(UnitTypeId.THOR):
                        fac.train(UnitTypeId.THOR)
                        break
                    elif bot.can_afford(UnitTypeId.CYCLONE):
                        fac.train(UnitTypeId.CYCLONE)
                        break

        elif act == "TRAIN_MEDIVAC":
            for sp in bot.structures(UnitTypeId.STARPORT).ready:
                cap = 2 if getattr(sp, "has_reactor", False) else 1
                if len(sp.orders) < cap and bot.can_afford(UnitTypeId.MEDIVAC):
                    sp.train(UnitTypeId.MEDIVAC)
                    break

        elif act == "TRAIN_VIKING_OR_LIBERATOR":
            enemy_has_air = any(getattr(eu, "is_flying", False) for eu in bot.enemy_units)
            unit_choice = UnitTypeId.VIKINGFIGHTER if enemy_has_air else UnitTypeId.LIBERATOR
            for sp in bot.structures(UnitTypeId.STARPORT).ready:
                cap = 2 if getattr(sp, "has_reactor", False) else 1
                if len(sp.orders) < cap and bot.can_afford(unit_choice):
                    sp.train(unit_choice)
                    break

        # [26] RESEARCH_BIO_TECH
        elif act == "RESEARCH_BIO_TECH":
            for tl in bot.structures(UnitTypeId.BARRACKSTECHLAB).ready.idle:
                for upg in (UpgradeId.STIMPACK, UpgradeId.SHIELDWALL, UpgradeId.PUNISHERGRENADES):
                    if upg not in bot.state.upgrades and bot.can_afford(upg):
                        tl.research(upg)
                        return

        # [27] RESEARCH_UPGRADES
        elif act == "RESEARCH_UPGRADES":
            for eb in bot.structures(UnitTypeId.ENGINEERINGBAY).ready.idle:
                for upg in (
                    UpgradeId.TERRANINFANTRYWEAPONSLEVEL1,
                    UpgradeId.TERRANINFANTRYARMORSLEVEL1,
                    UpgradeId.TERRANINFANTRYWEAPONSLEVEL2,
                    UpgradeId.TERRANINFANTRYARMORSLEVEL2,
                    UpgradeId.TERRANINFANTRYWEAPONSLEVEL3,
                    UpgradeId.TERRANINFANTRYARMORSLEVEL3,
                ):
                    if upg not in bot.state.upgrades and bot.can_afford(upg):
                        eb.research(upg)
                        return
            for arm in bot.structures(UnitTypeId.ARMORY).ready.idle:
                for upg in (
                    UpgradeId.TERRANVEHICLEWEAPONSLEVEL1,
                    UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL1,
                    UpgradeId.TERRANSHIPWEAPONSLEVEL1,
                ):
                    if upg not in bot.state.upgrades and bot.can_afford(upg):
                        arm.research(upg)
                        return

    async def execute_micro(self, bot, micro_id: int, gx: int, gy: int):
        from sc2.ids.ability_id import AbilityId
        from sc2.ids.buff_id import BuffId
        from sc2.ids.unit_typeid import UnitTypeId

        act = ID_TO_MICRO.get(int(micro_id), "MICRO_NO_OP")
        if act == "MICRO_NO_OP":
            return

        target_pt = self._to_point2(bot, gx, gy)
        army = bot.units.exclude_type({UnitTypeId.SCV, UnitTypeId.MULE})

        # [01] SCOUT_WITH_SCV_OR_REAPER
        if act == "SCOUT_WITH_SCV_OR_REAPER":
            reapers = bot.units(UnitTypeId.REAPER)
            if reapers:
                reapers.closest_to(target_pt).move(target_pt)
            elif bot.workers:
                scout = bot.workers.find_by_tag(self.scout_worker_tag) if self.scout_worker_tag else None
                if scout is None and bot.supply_workers >= 14:
                    scout = bot.workers.closest_to(target_pt)
                    self.scout_worker_tag = scout.tag
                if scout is not None:
                    scout.move(target_pt)

        # [02] SCANNER_SWEEP
        elif act == "SCANNER_SWEEP":
            for oc in bot.townhalls(UnitTypeId.ORBITALCOMMAND).ready:
                if oc.energy >= 50:
                    oc(AbilityId.SCANNERSWEEP_SCAN, target_pt)
                    break

        # [03] MOVE_ARMY
        elif act == "MOVE_ARMY":
            for u in army:
                if u.type_id != UnitTypeId.SIEGETANKSIEGED:
                    u.move(target_pt)

        # [04] ATTACK_MOVE_ARMY
        elif act == "ATTACK_MOVE_ARMY":
            non_med = army.exclude_type({UnitTypeId.MEDIVAC, UnitTypeId.SIEGETANKSIEGED})
            med_dest = non_med.center if non_med else target_pt
            for u in army:
                if u.type_id == UnitTypeId.MEDIVAC:
                    u.attack(med_dest)
                elif u.type_id != UnitTypeId.SIEGETANKSIEGED:
                    u.attack(target_pt)

        # [05] FOCUS_FIRE_AREA
        elif act == "FOCUS_FIRE_AREA":
            enemies = (bot.enemy_units + bot.enemy_structures).closer_than(18.0, target_pt)
            if enemies and army:
                weakest = min(enemies, key=lambda e: e.health + getattr(e, "shield", 0.0))
                for u in army:
                    if u.type_id != UnitTypeId.MEDIVAC:
                        u.attack(weakest)
            else:
                for u in army:
                    if u.type_id != UnitTypeId.SIEGETANKSIEGED:
                        u.attack(target_pt)

        # [06] KITE_RETREAT_ARMY
        elif act == "KITE_RETREAT_ARMY":
            for u in army:
                if u.type_id == UnitTypeId.SIEGETANKSIEGED:
                    continue
                if u.health_percentage < 0.70 or (bot.enemy_units and bot.enemy_units.closer_than(6.5, u)):
                    u.move(u.position.towards(target_pt, 5.5))

        # [07] USE_STIMPACK
        elif act == "USE_STIMPACK":
            for u in bot.units({UnitTypeId.MARINE, UnitTypeId.MARAUDER}):
                if u.health_percentage > 0.55 and not u.has_buff(BuffId.STIMPACK):
                    if bot.enemy_units and bot.enemy_units.closer_than(14.0, u):
                        ab = AbilityId.EFFECT_STIM_MARINE if u.type_id == UnitTypeId.MARINE else AbilityId.EFFECT_STIM_MARAUDER
                        u(ab)

        # [08] SIEGE_TANKS_AT_AREA
        elif act == "SIEGE_TANKS_AT_AREA":
            tanks = bot.units(UnitTypeId.SIEGETANK)
            local = tanks.closer_than(16.0, target_pt) if tanks else []
            for tank in (local or tanks):
                tank(AbilityId.SIEGEMODE_SIEGEMODE)

        # [09] UNSIEGE_TANKS_TO_MOVE
        elif act == "UNSIEGE_TANKS_TO_MOVE":
            sieged = bot.units(UnitTypeId.SIEGETANKSIEGED)
            local = sieged.closer_than(16.0, target_pt) if sieged else []
            for tank in (local or sieged):
                tank(AbilityId.UNSIEGE_UNSIEGE)

        # [10] REPAIR_WITH_SCV
        elif act == "REPAIR_WITH_SCV":
            damaged = [
                u for u in (bot.units + bot.structures)
                if u.is_mechanical and u.health_percentage < 1.0
            ]
            if damaged and bot.workers:
                target_mech = min(damaged, key=lambda m: m.distance_to(target_pt) + 25.0 * m.health_percentage)
                scv = bot.workers.closest_to(target_mech)
                scv.repair(target_mech)

        # [11] LOWER_OR_RAISE_DEPOTS
        elif act == "LOWER_OR_RAISE_DEPOTS":
            for d in bot.structures(UnitTypeId.SUPPLYDEPOT).ready:
                if not bot.enemy_units or not bot.enemy_units.closer_than(12.0, d):
                    d(AbilityId.MORPH_SUPPLYDEPOT_LOWER)
            for d in bot.structures(UnitTypeId.SUPPLYDEPOTLOWERED).ready:
                if bot.enemy_units and bot.enemy_units.closer_than(12.0, d):
                    d(AbilityId.MORPH_SUPPLYDEPOT_RAISE)
