"""
Configuración central de SC2-Hailo8-Transformer V2 (Terran-First, Dual-Branch 32x32 + Global Vector).
Diseñado para inferencia INT8 single-context en Hailo-8 (RPi 5) y entrenamiento CPU/GPU en WSL/Colab.
"""

from dataclasses import dataclass
from typing import Dict, List, Set

# Dimensiones estáticas NCHW para Hailo-8
GRID_H: int = 32
GRID_W: int = 32
BOTTLENECK_H: int = 16
BOTTLENECK_W: int = 16
MAP_CHANNELS: int = 16
GLOBAL_DIM: int = 64
GLOBAL_TOKENS: int = 4

# Razas enemigas codificadas en global_vec
ENEMY_RACES: Dict[str, int] = {
    "Terran": 0,
    "Protoss": 1,
    "Zerg": 2,
    "Random": 3,
}

# ============================================================================
# Vocabulario Dual Terran (Macro = 28 acciones, Micro = 12 acciones)
# Ejecutados simultáneamente en cada tick (K=1, lazo cerrado) con Action Masking
# ============================================================================

MACRO_VOCAB: List[str] = [
    "MACRO_NO_OP",                  # 0
    "ASSIGN_MINERALS",              # 1
    "ASSIGN_GAS",                   # 2
    "TRAIN_SCV",                    # 3
    "BUILD_SUPPLY_DEPOT",           # 4  [espacial]
    "BUILD_REFINERY",               # 5  [espacial]
    "BUILD_COMMAND_CENTER",         # 6  [espacial]
    "BUILD_BARRACKS",               # 7  [espacial]
    "BUILD_FACTORY",                # 8  [espacial]
    "BUILD_STARPORT",               # 9  [espacial]
    "BUILD_ENGINEERING_BAY",        # 10 [espacial]
    "BUILD_ARMORY",                 # 11 [espacial]
    "BUILD_BUNKER",                 # 12 [espacial]
    "BUILD_MISSILE_TURRET",         # 13 [espacial]
    "UPGRADE_ORBITAL_COMMAND",      # 14 [espacial: CC objetivo]
    "BUILD_ADDON_TECHLAB",          # 15 [espacial: edificio producción]
    "BUILD_ADDON_REACTOR",          # 16 [espacial: edificio producción]
    "CALLDOWN_MULE",                # 17 [espacial: línea mineral]
    "TRAIN_MARINE",                 # 18
    "TRAIN_MARAUDER",               # 19
    "TRAIN_REAPER",                 # 20
    "TRAIN_HELLION",                # 21
    "TRAIN_SIEGE_TANK",             # 22
    "TRAIN_THOR_OR_CYCLONE",        # 23
    "TRAIN_MEDIVAC",                # 24
    "TRAIN_VIKING_OR_LIBERATOR",    # 25
    "RESEARCH_BIO_TECH",            # 26 (Stimpack / Combat Shield / Concussive)
    "RESEARCH_UPGRADES",            # 27 (+1/+2/+3 Armas y Armadura Bio/Mech)
]

MICRO_VOCAB: List[str] = [
    "MICRO_NO_OP",                  # 0
    "SCOUT_WITH_SCV_OR_REAPER",     # 1  [espacial]
    "SCANNER_SWEEP",                # 2  [espacial]
    "MOVE_ARMY",                    # 3  [espacial]
    "ATTACK_MOVE_ARMY",             # 4  [espacial]
    "FOCUS_FIRE_AREA",              # 5  [espacial]
    "KITE_RETREAT_ARMY",            # 6  [espacial]
    "USE_STIMPACK",                 # 7
    "SIEGE_TANKS_AT_AREA",          # 8  [espacial]
    "UNSIEGE_TANKS_TO_MOVE",        # 9  [espacial]
    "REPAIR_WITH_SCV",              # 10 [espacial]
    "LOWER_OR_RAISE_DEPOTS",        # 11
]

NUM_MACRO: int = len(MACRO_VOCAB)
NUM_MICRO: int = len(MICRO_VOCAB)

MACRO_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(MACRO_VOCAB)}
ID_TO_MACRO: Dict[int, str] = {i: name for i, name in enumerate(MACRO_VOCAB)}
MICRO_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(MICRO_VOCAB)}
ID_TO_MICRO: Dict[int, str] = {i: name for i, name in enumerate(MICRO_VOCAB)}

# Acciones que utilizan el heatmap espacial 32x32 correspondiente
SPATIAL_MACRO_IDS: Set[int] = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
SPATIAL_MICRO_IDS: Set[int] = {1, 2, 3, 4, 5, 6, 8, 9, 10}

# Protocolo binario RPi5 <-> PC
PROTOCOL_MAGIC: int = 0x53433256  # 'SC2V'
PROTOCOL_VERSION: int = 2
MAP_BYTES: int = 1 * MAP_CHANNELS * GRID_H * GRID_W * 4       # 65,536 bytes
VEC_BYTES: int = 1 * GLOBAL_DIM * 1 * 1 * 4                   # 256 bytes
REQ_BYTES: int = MAP_BYTES + VEC_BYTES                        # 65,792 bytes
SPATIAL_CELLS: int = GRID_H * GRID_W                          # 1,024


@dataclass
class V2Config:
    d_model: int = 64
    stem_channels: int = 48
    n_heads: int = 4
    num_layers: int = 3
    mlp_ratio: int = 2
    dropout: float = 0.05

    # Entrenamiento BC / PPO
    batch_size: int = 64
    bc_epochs: int = 20
    bc_lr: float = 5e-4
    rl_epochs: int = 6
    rl_lr: float = 1e-4
    weight_decay: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.20
    kl_beta: float = 0.05
    entropy_coef: float = 0.015
    onnx_opset: int = 16
    calib_samples: int = 128

    # Red local RPi5
    lan_host: str = "0.0.0.0"
    lan_port: int = 5555

    # Rutas relativas al repositorio (compatibles con Windows y WSL)
    actor_weights: str = "models/sc2_v2_actor.pth"
    bc_anchor_weights: str = "models/sc2_v2_bc_anchor.pth"
    critic_weights: str = "models/sc2_v2_critic.pth"
    onnx_path: str = "models/sc2_v2_actor.onnx"
    hef_path: str = "models/sc2_v2_agent.hef"
    har_path: str = "models/sc2_v2_quantized.har"
    calib_path: str = "models/sc2_v2_calib.npz"
    rollout_path: str = "models/sc2_v2_rollouts.npz"
    bc_dataset_path: str = "data/sc2_v2_bc_dataset.npz"
