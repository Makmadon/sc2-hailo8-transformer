"""
Configuración central del modelo SC2-Hailo8-Transformer y mapeo de acciones/estados.
Diseñado específicamente para tensores 4D estáticos compatibles con Hailo-8 (26 TOPS INT8).
"""

from dataclasses import dataclass
from typing import Dict, List

# Dimensiones espaciales y de entrada para Hailo-8 (Formato NCHW: [Batch=1, C=12, H=16, W=16])
# 16x16 = 256 tokens espaciales/estado (ideal para atención estática en Hailo-8 sin desbordar SRAM)
GRID_H: int = 16
GRID_W: int = 16
IN_CHANNELS: int = 12  # 6 canales espaciales (minimapa) + 6 canales de estado económico/temporal

# Horizonte de Action-Chunking (predice K=4 acciones futuras en un solo pase de inferencia)
ACTION_CHUNK_SIZE: int = 4

# Razas de StarCraft II
RACES: Dict[str, int] = {
    "Terran": 0,
    "Protoss": 1,
    "Zerg": 2,
    "Random": 3,
}

# Vocabulario unificado de 32 acciones macro y micro multi-raza (Terran / Protoss / Zerg)
ACTION_VOCAB: List[str] = [
    "NO_OP",                        # 0: Sin acción / esperar
    "HARVEST_GATHER_MINERALS",      # 1: Enviar trabajadores a minerales
    "HARVEST_GATHER_GAS",           # 2: Enviar trabajadores a refinería/asimilador/extractor
    "TRAIN_WORKER",                 # 3: Entrenar SCV / Probe / Drone
    "BUILD_SUPPLY",                 # 4: Supply Depot / Pylon / Overlord
    "BUILD_GAS_EXTRACTOR",          # 5: Refinery / Assimilator / Extractor
    "BUILD_EXPAND_BASE",            # 6: Command Center / Nexus / Hatchery
    "BUILD_BARRACKS_GATEWAY_POOL",  # 7: Estructura básica de infantería (Barracks/Gateway/SpawningPool)
    "BUILD_FACTORY_ROBO_ROACH",     # 8: Estructura mecanizada/media (Factory/Robotics/RoachWarren)
    "BUILD_STARPORT_STARGATE_SPIRE",# 9: Estructura aérea (Starport/Stargate/Spire)
    "BUILD_TECH_LAB_CORE_LAIR",     # 10: Estructura tecnológica (TechLab/CyberneticsCore/Lair/Hive)
    "BUILD_DEFENSE_STRUCTURE",      # 11: Bunker / MissileTurret / PhotonCannon / SpineCrawler
    "TRAIN_BASIC_INFANTRY",         # 12: Marine / Zealot / Zergling
    "TRAIN_RANGED_CORE",            # 13: Marauder / Stalker / Roach / Hydralisk
    "TRAIN_SPECIALIST_UNIT",        # 14: Reaper / Ghost / Sentry / HighTemplar / Baneling / Infestor
    "TRAIN_SIEGE_HEAVY",            # 15: SiegeTank / Thor / Immortal / Colossus / Ultralisk / Lurker
    "TRAIN_AIR_UNIT",               # 16: Medivac / Viking / Battlecruiser / VoidRay / Carrier / Mutalisk
    "RESEARCH_COMBAT_UPGRADE",      # 17: Stimpack / CombatShield / Blink / Charge / ZerglingSpeed
    "RESEARCH_WEAPONS_ARMOR",       # 18: Mejoras +1/+2/+3 de ataque y armadura en EngineeringBay/Forge/Evo
    "MACRO_CHRONO_MULE_INJECT",     # 19: Habilidad macro de raza (Calldown MULE / ChronoBoost / SpawnLarva)
    "MOVE_ARMY_POSITION",           # 20: Mover unidades a coordenada (x, y)
    "MOVE_WORKER_SCOUT",            # 21: Mover trabajador a explorar coordenada (x, y)
    "ATTACK_MOVE_COORDINATE",       # 22: Attack-Move (A-Move) hacia coordenada (x, y)
    "FOCUS_FIRE_TARGET",            # 23: Concentrar fuego en unidad/estructura enemiga en (x, y)
    "MICRO_KITE_RETREAT",           # 24: Retroceder unidades heridas / kite hacia (x, y)
    "MICRO_USE_STIM_ABILITY",       # 25: Activar habilidad ofensiva (Stimpack / Blink / ForceField / RavagerBile)
    "MICRO_SIEGE_TRANSFORM",        # 26: Siege/Unsiege Tank, Liberator, Burrow/Unburrow, Viking mode
    "MICRO_HEAL_REPAIR_SHIELD",     # 27: Reparar con SCV / Medivac Heal / Shield Battery
    "DEFEND_MAIN_NATURAL",          # 28: Reagrupar ejército en base principal o expansión natural
    "HARASS_ENEMY_WORKERS",         # 29: Ataque rápido a línea de minerales enemiga en (x, y)
    "RALLY_PRODUCTION",             # 30: Configurar punto de reunión de estructuras en (x, y)
    "CANCEL_OR_HOLD_POSITION",      # 31: Hold position / Stop / Cancelar construcción en riesgo
]

NUM_ACTIONS: int = len(ACTION_VOCAB)
ACTION_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(ACTION_VOCAB)}
ID_TO_ACTION: Dict[int, str] = {idx: name for idx, name in enumerate(ACTION_VOCAB)}

# Dimensiones por cada paso del Action-Chunk:
# [NUM_ACTIONS (32 logits) + GRID_W (16 logits X) + GRID_H (16 logits Y) + QUEUE_FLAG (2 logits)] = 66 valores por paso
STEP_OUTPUT_DIM: int = NUM_ACTIONS + GRID_W + GRID_H + 2


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 15
    focal_gamma: float = 2.0
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 4
    mlp_ratio: int = 2
    dropout: float = 0.1
    onnx_opset: int = 16
    calib_samples: int = 64
    lan_host: str = "0.0.0.0"
    lan_port: int = 5555
