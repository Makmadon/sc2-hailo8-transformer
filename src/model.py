"""
Arquitectura SC2HailoTransformer (Non-AlphaStar, 100% NCHW 4D Estática para Hailo-8).
Evita los operadores que causan fallos en el Hailo Dataflow Compiler:
  - Sin Pointer Networks de longitud variable ni LSTMs recurrentes.
  - Sin tensores 3D dinámicos ni LayerNorm 1D que requieran 'normalization(decomposition=false)'.
  - Utiliza bloques Conv-Transformer 4D (Conv2d 1x1 + BatchNorm2d + Multi-Head Self-Attention estática)
    y cabezal de Action-Chunking que predice K=4 pasos futuros en un único tensor 4D [B, 66, 1, 4].
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    ACTION_CHUNK_SIZE,
    GRID_H,
    GRID_W,
    IN_CHANNELS,
    NUM_ACTIONS,
    STEP_OUTPUT_DIM,
)


class HailoAttentionBlock2D(nn.Module):
    """
    Bloque Transformer 4D optimizado para la NPU Hailo-8 (26 TOPS INT8).
    Opera íntegramente sobre tensores [B, C, H, W] con proyecciones Conv2d(1x1),
    BatchNorm2d (que se fusiona gratuitamente en hardware durante la cuantización)
    y atención multi-cabezal con formas estáticas conocidas en tiempo de compilación.
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model debe ser divisible entre n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # Proyecciones Q, K, V mediante Conv2d 1x1 (aceleradas nativamente en los contextos de Hailo-8)
        self.norm1 = nn.BatchNorm2d(d_model)
        self.q_proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
        self.v_proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
        self.attn_drop = nn.Dropout(dropout)

        # Feed-Forward Espacial + Canal (Depthwise 3x3 + Pointwise 1x1)
        hidden_dim = d_model * mlp_ratio
        self.norm2 = nn.BatchNorm2d(d_model)
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, d_model, kernel_size=1, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] con H=16, W=16 -> N = 256 tokens espaciales/estado
        B, C, H, W = x.shape
        N = H * W

        h = self.norm1(x)
        # Reorganización 4D explícita para compatibilidad con el parser ONNX de Hailo DFC
        q = self.q_proj(h).reshape(B, self.n_heads, self.head_dim, N).permute(0, 1, 3, 2)  # [B, heads, N, head_dim]
        k = self.k_proj(h).reshape(B, self.n_heads, self.head_dim, N)                      # [B, heads, head_dim, N]
        v = self.v_proj(h).reshape(B, self.n_heads, self.head_dim, N).permute(0, 1, 3, 2)  # [B, heads, N, head_dim]

        attn_scores = torch.matmul(q, k) * self.scale                                      # [B, heads, N, N]
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        attn_out = torch.matmul(attn_probs, v)                                             # [B, heads, N, head_dim]
        attn_out = attn_out.permute(0, 1, 3, 2).reshape(B, C, H, W)                        # [B, C, H, W]
        x = x + self.out_proj(attn_out)

        # Bloque Conv-FFN con conexión residual
        x = x + self.ffn(self.norm2(x))
        return x


class SC2HailoTransformer(nn.Module):
    """
    Modelo completo de toma de decisiones tácticas y estratégicas para StarCraft II.
    Entrada:
      - obs_tensor: [B, 12, 16, 16] (Minimapa 6 canales + Estado Económico/Raza/Tiempo 6 canales)
    Salida:
      - action_chunks_out: [B, 66, 1, 4]
        Donde para cada uno de los K=4 pasos del Action-Chunk se predicen 66 logits:
          * [0:32]  -> action_id (32 acciones macro/micro multi-raza)
          * [32:48] -> target_x (16 celdas espaciales X)
          * [48:64] -> target_y (16 celdas espaciales Y)
          * [64:66] -> queue_flag (2 estados: inmediato vs encolado Shift)
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        # Positional Embedding 2D aprendible (formato 4D estático [1, d_model, 16, 16])
        self.pos_embed = nn.Parameter(torch.randn(1, d_model, GRID_H, GRID_W) * 0.02)

        self.blocks = nn.ModuleList([
            HailoAttentionBlock2D(
                d_model=d_model,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.BatchNorm2d(d_model)

        # Reducción espacial [B, d_model, 16, 16] -> [B, d_model, 1, 4] (4 slots de Action-Chunking)
        # Implementada como Conv2d con kernel=(16, 4) y stride=(16, 4) para máxima velocidad en Hailo-8
        self.chunk_pool = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(GRID_H, ACTION_CHUNK_SIZE), stride=(GRID_H, ACTION_CHUNK_SIZE), bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        # Cabezal unificado 1x1 hacia los 66 logits por cada uno de los K=4 pasos del horizonte
        self.action_head = nn.Conv2d(d_model, STEP_OUTPUT_DIM, kernel_size=1, bias=True)

    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        x = self.stem(obs_tensor) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x)
        chunks_feat = self.chunk_pool(x)          # [B, d_model, 1, 4]
        out = self.action_head(chunks_feat)       # [B, 66, 1, 4]
        return out

    @staticmethod
    def split_predictions(out_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Separa el tensor unificado [B, 66, 1, 4] en los 4 cabezales semánticos con forma [B, 4, NumClasses]:
          - action_logits: [B, 4, 32]
          - tx_logits:     [B, 4, 16]
          - ty_logits:     [B, 4, 16]
          - queue_logits:  [B, 4, 2]
        """
        # out_tensor: [B, 66, 1, 4] -> [B, 4, 66]
        seq = out_tensor.squeeze(2).permute(0, 2, 1)
        c1 = NUM_ACTIONS
        c2 = c1 + GRID_W
        c3 = c2 + GRID_H
        action_logits = seq[:, :, :c1]
        tx_logits = seq[:, :, c1:c2]
        ty_logits = seq[:, :, c2:c3]
        queue_logits = seq[:, :, c3:]
        return action_logits, tx_logits, ty_logits, queue_logits


class ClassBalancedFocalLoss(nn.Module):
    """
    Función de pérdida multi-tarea con Focal Loss ponderada por clase para combatir
    el desbalanceo extremo de acciones en replays reales de StarCraft II.
    """

    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma

    def forward(self, raw_out: torch.Tensor, target_chunks: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        raw_out: [B, 66, 1, 4]
        target_chunks: [B, 4, 4] -> (action_id, target_x, target_y, queue_flag)
        """
        act_logits, tx_logits, ty_logits, q_logits = SC2HailoTransformer.split_predictions(raw_out)

        t_act = target_chunks[:, :, 0].reshape(-1)
        t_tx = target_chunks[:, :, 1].reshape(-1)
        t_ty = target_chunks[:, :, 2].reshape(-1)
        t_q = target_chunks[:, :, 3].reshape(-1)

        # 1. Class-Balanced Focal Loss para el vocabulario de 32 acciones
        flat_act_logits = act_logits.reshape(-1, NUM_ACTIONS)
        ce_act = F.cross_entropy(flat_act_logits, t_act, reduction="none")
        pt = torch.exp(-ce_act)
        alpha_t = self.class_weights[t_act]
        focal_act_loss = (alpha_t * ((1.0 - pt) ** self.gamma) * ce_act).mean()

        # 2. Pérdidas espaciales (X, Y) y bandera de cola (Queue)
        loss_tx = F.cross_entropy(tx_logits.reshape(-1, GRID_W), t_tx)
        loss_ty = F.cross_entropy(ty_logits.reshape(-1, GRID_H), t_ty)
        loss_q = F.cross_entropy(q_logits.reshape(-1, 2), t_q)

        total_loss = focal_act_loss + 0.5 * (loss_tx + loss_ty) + 0.25 * loss_q

        with torch.no_grad():
            acc_act = (flat_act_logits.argmax(dim=-1) == t_act).float().mean().item()
            acc_tx = (tx_logits.reshape(-1, GRID_W).argmax(dim=-1) == t_tx).float().mean().item()
            acc_ty = (ty_logits.reshape(-1, GRID_H).argmax(dim=-1) == t_ty).float().mean().item()

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_focal_action": float(focal_act_loss.item()),
            "loss_coords": float((loss_tx + loss_ty).item()),
            "acc_action": acc_act,
            "acc_coords": 0.5 * (acc_tx + acc_ty),
        }
        return total_loss, metrics


def export_to_hailo_onnx(
    model: SC2HailoTransformer,
    onnx_path: str = "sc2_hailo8_transformer.onnx",
    opset_version: int = 16,
) -> str:
    """
    Exporta el modelo a un archivo ONNX 100% estático (Batch=1, sin dynamic_axes ni archivos .data externos)
    listo para ser compilado por Hailo Dataflow Compiler v3.34.0 hacia formato .hef.
    """
    import onnx

    model_cpu = model.eval().to("cpu")
    dummy_input = torch.randn(1, IN_CHANNELS, GRID_H, GRID_W, dtype=torch.float32)

    # Usamos dynamo=False en PyTorch 2.x para producir un grafo estático limpio y autocontenido
    export_kwargs = dict(
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["obs_tensor"],
        output_names=["action_chunks_out"],
    )
    try:
        torch.onnx.export(model_cpu, dummy_input, onnx_path, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model_cpu, dummy_input, onnx_path, **export_kwargs)

    # Si PyTorch generó un archivo .data externo, lo empaquetamos dentro del único .onnx
    onnx_model = onnx.load(onnx_path, load_external_data=True)
    onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)
    data_file = onnx_path + ".data"
    import os
    if os.path.exists(data_file):
        os.remove(data_file)

    onnx.checker.check_model(onnx.load(onnx_path))
    print(f"[ONNX] Exportado y verificado exitosamente (100% estático, un solo archivo): {onnx_path}")
    return onnx_path
