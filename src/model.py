"""
Arquitectura SC2HailoV2Actor & SC2HailoV2Critic (100% NCHW 4D Estática para Hailo-8).
Mejoras clave frente a V1:
  - Entrada Dual sin contaminación de canales: map_tensor [B, 16, 32, 32] + global_vec [B, 64, 1, 1].
  - Sin chunk_pool por franjas verticales: K=1 en lazo cerrado con 2 cabezales simultáneos (Macro + Micro).
  - Cross-Attention bidireccional Espacial <-> Global usando solo operadores nativos Hailo-8
    (Conv2d 1x1, BatchNorm2d, ReLU, Reshape, Permute, MatMul, Softmax, Add).
  - Decoder espacial ConvTranspose2d 16x16 -> 32x32 con skip-connection para emitir heatmaps
    espaciales conjuntos [B, 2, 32, 32] (1,024 celdas para Macro y 1,024 para Micro).
"""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    BOTTLENECK_H,
    BOTTLENECK_W,
    GLOBAL_DIM,
    GLOBAL_TOKENS,
    GRID_H,
    GRID_W,
    MAP_CHANNELS,
    NUM_MACRO,
    NUM_MICRO,
)


class HailoConvNeXtBlock2D(nn.Module):
    """Bloque residual ligero (Depthwise 3x3 + Pointwise 1x1) 100% fusionable en Hailo-8."""

    def __init__(self, channels: int, mlp_ratio: int = 2, dropout: float = 0.05):
        super().__init__()
        hidden = channels * mlp_ratio
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class HailoDualAttentionBlock(nn.Module):
    """
    Bloque Transformer Dual 4D para Hailo-8:
      1. Spatial Self-Attention (16x16 = 256 tokens espaciales).
      2. Spatial <- Global Cross-Attention (cada celda consulta a los 4 tokens globales).
      3. Global <- Spatial Cross-Attention (los 4 tokens globales leen del tablero 16x16).
      4. FFN espacial (ConvNeXt) + FFN global (Conv 1x1).
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, mlp_ratio: int = 2, dropout: float = 0.05):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # 1. Spatial Self-Attention
        self.norm_s1 = nn.BatchNorm2d(d_model)
        self.q_s = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
        self.k_s = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
        self.v_s = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
        self.out_s = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)

        # 2. Spatial <- Global Cross-Attention
        self.norm_s2 = nn.BatchNorm2d(d_model)
        self.norm_g1 = nn.BatchNorm2d(d_model)
        self.q_sg = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
        self.k_sg = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
        self.v_sg = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
        self.out_sg = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)

        # 3. Global <- Spatial Cross-Attention
        self.norm_g2 = nn.BatchNorm2d(d_model)
        self.norm_s3 = nn.BatchNorm2d(d_model)
        self.q_gs = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
        self.k_gs = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
        self.v_gs = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
        self.out_gs = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)

        self.attn_drop = nn.Dropout(dropout)

        # 4. FFNs
        hidden = d_model * mlp_ratio
        self.norm_s4 = nn.BatchNorm2d(d_model)
        self.ffn_s = nn.Sequential(
            nn.Conv2d(d_model, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, d_model, kernel_size=1, bias=True),
        )

        self.norm_g3 = nn.BatchNorm2d(d_model)
        self.ffn_g = nn.Sequential(
            nn.Conv2d(d_model, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, d_model, kernel_size=1, bias=True),
        )

    def _mha(
        self,
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        v_proj: torch.Tensor,
        n_q: int,
        n_kv: int,
        out_h: int,
        out_w: int,
    ) -> torch.Tensor:
        B = q_proj.shape[0]
        q = q_proj.reshape(B, self.n_heads, self.head_dim, n_q).permute(0, 1, 3, 2)   # [B, h, n_q, hd]
        k = k_proj.reshape(B, self.n_heads, self.head_dim, n_kv)                       # [B, h, hd, n_kv]
        v = v_proj.reshape(B, self.n_heads, self.head_dim, n_kv).permute(0, 1, 3, 2)  # [B, h, n_kv, hd]

        attn = F.softmax(torch.matmul(q, k) * self.scale, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)                                                    # [B, h, n_q, hd]
        return out.permute(0, 1, 3, 2).reshape(B, self.d_model, out_h, out_w)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, 16, 16], g: [B, C, 1, 4]
        H, W = x.shape[2], x.shape[3]
        N_s = H * W
        N_g = g.shape[3]

        # 1. Spatial Self-Attention
        hs = self.norm_s1(x)
        x = x + self.out_s(self._mha(self.q_s(hs), self.k_s(hs), self.v_s(hs), N_s, N_s, H, W))

        # 2. Spatial <- Global Cross-Attention
        hs2 = self.norm_s2(x)
        hg1 = self.norm_g1(g)
        x = x + self.out_sg(self._mha(self.q_sg(hs2), self.k_sg(hg1), self.v_sg(hg1), N_s, N_g, H, W))

        # 3. Global <- Spatial Cross-Attention
        hg2 = self.norm_g2(g)
        hs3 = self.norm_s3(x)
        g = g + self.out_gs(self._mha(self.q_gs(hg2), self.k_gs(hs3), self.v_gs(hs3), N_g, N_s, 1, N_g))

        # 4. FFNs
        x = x + self.ffn_s(self.norm_s4(x))
        g = g + self.ffn_g(self.norm_g3(g))
        return x, g


class SC2HailoV2Actor(nn.Module):
    """
    Actor Dual-Branch 100% 4D estático para Hailo-8:
      Inputs:
        - map_tensor: [B, 16, 32, 32]
        - global_vec: [B, 64, 1, 1]
      Outputs:
        - macro_logits:   [B, 28, 1, 1]
        - micro_logits:   [B, 12, 1, 1]
        - spatial_logits: [B, 2, 32, 32] (canal 0 = macro x,y; canal 1 = micro x,y)
    """

    def __init__(
        self,
        map_channels: int = MAP_CHANNELS,
        global_dim: int = GLOBAL_DIM,
        stem_channels: int = 48,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 3,
        mlp_ratio: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model

        # Encoder espacial 32x32 -> 16x16
        self.stem_32 = nn.Sequential(
            nn.Conv2d(map_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            HailoConvNeXtBlock2D(stem_channels, mlp_ratio=2, dropout=dropout),
        )
        self.down_16 = nn.Sequential(
            nn.Conv2d(stem_channels, d_model, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, d_model, BOTTLENECK_H, BOTTLENECK_W) * 0.02)

        # Encoder global [B, 64, 1, 1] -> [B, d_model, 1, 4]
        global_hidden = max(96, d_model)
        self.global_enc = nn.Sequential(
            nn.Conv2d(global_dim, global_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(global_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(global_hidden, d_model * GLOBAL_TOKENS, kernel_size=1, bias=True),
        )
        self.global_pos = nn.Parameter(torch.randn(1, d_model, 1, GLOBAL_TOKENS) * 0.02)

        # Bloques Duales Spatial <-> Global Attention
        self.blocks = nn.ModuleList([
            HailoDualAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm_s = nn.BatchNorm2d(d_model)
        self.final_norm_g = nn.BatchNorm2d(d_model)

        # Agregador global de los 4 tokens -> [B, d_model, 1, 1]
        self.g_pool = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, GLOBAL_TOKENS), bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        # Cabezales discretos Macro y Micro [B, 2*d_model, 1, 1] -> logits
        head_in = d_model * 2
        macro_hidden = max(96, d_model)
        micro_hidden = max(64, d_model)
        self.macro_head = nn.Sequential(
            nn.Conv2d(head_in, macro_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(macro_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(macro_hidden, NUM_MACRO, kernel_size=1, bias=True),
        )
        self.micro_head = nn.Sequential(
            nn.Conv2d(head_in, micro_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(micro_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(micro_hidden, NUM_MICRO, kernel_size=1, bias=True),
        )

        # Decoder espacial U-Net 16x16 -> 32x32 con skip connection
        self.up_32 = nn.Sequential(
            nn.ConvTranspose2d(d_model, stem_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.spatial_head = nn.Sequential(
            nn.Conv2d(stem_channels * 2, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            HailoConvNeXtBlock2D(64, mlp_ratio=2, dropout=dropout),
            nn.Conv2d(64, 2, kernel_size=1, bias=True),
        )

    def forward(
        self,
        map_tensor: torch.Tensor,
        global_vec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = map_tensor.shape[0]

        skip_32 = self.stem_32(map_tensor)                     # [B, 48, 32, 32]
        x = self.down_16(skip_32) + self.pos_embed             # [B, 96, 16, 16]

        g_raw = self.global_enc(global_vec)                    # [B, 96*4, 1, 1]
        g = (
            g_raw.reshape(B, GLOBAL_TOKENS, self.d_model)
            .permute(0, 2, 1)
            .reshape(B, self.d_model, 1, GLOBAL_TOKENS)
            + self.global_pos
        )                                                      # [B, 96, 1, 4]

        for blk in self.blocks:
            x, g = blk(x, g)

        x = self.final_norm_s(x)
        g = self.final_norm_g(g)

        x_avg = F.avg_pool2d(x, kernel_size=(BOTTLENECK_H, BOTTLENECK_W))  # [B, 96, 1, 1]
        g_flat = self.g_pool(g)                                            # [B, 96, 1, 1]
        joint = torch.cat([x_avg, g_flat], dim=1)                          # [B, 192, 1, 1]

        macro_logits = self.macro_head(joint)                              # [B, 28, 1, 1]
        micro_logits = self.micro_head(joint)                              # [B, 12, 1, 1]

        up = self.up_32(x)                                                 # [B, 48, 32, 32]
        spatial_logits = self.spatial_head(torch.cat([up, skip_32], dim=1))  # [B, 2, 32, 32]
        return macro_logits, micro_logits, spatial_logits


class SC2HailoV2Critic(nn.Module):
    """
    Red Crítica de Valor V_phi(s) para PPO/GAE (solo entrenamiento en CPU/GPU, nunca se exporta al Hailo-8).
    """

    def __init__(self, map_channels: int = MAP_CHANNELS, global_dim: int = GLOBAL_DIM, hidden_dim: int = 96):
        super().__init__()
        self.map_enc = nn.Sequential(
            nn.Conv2d(map_channels, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),  # 16x16
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            HailoConvNeXtBlock2D(hidden_dim, mlp_ratio=2, dropout=0.05),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),  # 8x8
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.vec_enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(global_dim, 96),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 96, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, map_tensor: torch.Tensor, global_vec: torch.Tensor) -> torch.Tensor:
        m_feat = self.map_enc(map_tensor).flatten(1)
        v_feat = self.vec_enc(global_vec)
        return self.head(torch.cat([m_feat, v_feat], dim=1)).squeeze(-1)


# ============================================================================
# Utilidades probabilísticas con Action Masking para BC y PPO
# ============================================================================

def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Enmascara acciones inválidas fijando sus logits a -1e9 antes del softmax."""
    return logits.masked_fill(~mask.bool(), -1e9)


def masked_categorical_stats(
    logits: torch.Tensor,
    actions: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calcula (log_prob, entropy) bajo una distribución categórica con máscara opcional."""
    if mask is not None:
        logits = apply_action_mask(logits, mask)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    act_log_prob = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    # Evitar 0 * (-inf) = NaN cuando logits enmascarados tienen prob ~ 0
    entropy = -(probs * log_probs.clamp(min=-30.0)).sum(dim=-1)
    return act_log_prob, entropy


def masked_kl_divergence(
    logits_theta: torch.Tensor,
    logits_ref: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """KL(pi_ref || pi_theta) restringida a las acciones válidas."""
    if mask is not None:
        logits_theta = apply_action_mask(logits_theta, mask)
        logits_ref = apply_action_mask(logits_ref, mask)
    log_p_theta = F.log_softmax(logits_theta, dim=-1)
    p_ref = F.softmax(logits_ref, dim=-1)
    log_p_ref = F.log_softmax(logits_ref, dim=-1)
    kl_elem = p_ref * (log_p_ref.clamp(min=-30.0) - log_p_theta.clamp(min=-30.0))
    return kl_elem.sum(dim=-1).mean()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_v2_to_onnx(
    model: SC2HailoV2Actor,
    onnx_path: str = "models/sc2_v2_actor.onnx",
    opset_version: int = 16,
) -> str:
    """
    Exporta SC2HailoV2Actor a ONNX 100% estático (Batch=1, archivo único sin .data externo)
    para Hailo Dataflow Compiler v3.34.0.
    """
    import onnx

    model_cpu = model.eval().to("cpu")
    dummy_map = torch.randn(1, MAP_CHANNELS, GRID_H, GRID_W, dtype=torch.float32)
    dummy_vec = torch.randn(1, GLOBAL_DIM, 1, 1, dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    export_kwargs = dict(
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["map_tensor", "global_vec"],
        output_names=["macro_logits", "micro_logits", "spatial_logits"],
    )
    try:
        torch.onnx.export(model_cpu, (dummy_map, dummy_vec), onnx_path, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model_cpu, (dummy_map, dummy_vec), onnx_path, **export_kwargs)

    onnx_model = onnx.load(onnx_path, load_external_data=True)
    onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)

    onnx.checker.check_model(onnx.load(onnx_path))
    size_kb = os.path.getsize(onnx_path) / 1024.0
    print(f"[ONNX V2] Exportado y verificado (estático, 1 archivo, {size_kb:.1f} KB): {onnx_path}")
    return onnx_path
