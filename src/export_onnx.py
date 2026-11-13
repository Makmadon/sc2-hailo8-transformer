"""
Exporta SC2HailoV2Actor a ONNX 100% estático (Batch=1, Dual-Input NCHW, Opset 16)
listo para compilación en Hailo Dataflow Compiler (WSL / Colab).
"""

import argparse
import os
import torch

from .config import V2Config
from .model import SC2HailoV2Actor, count_parameters, export_v2_to_onnx


def main():
    cfg = V2Config()
    parser = argparse.ArgumentParser(description="Exportar SC2HailoV2Actor a ONNX estático")
    parser.add_argument("--weights", type=str, default=cfg.actor_weights, help="Ruta al checkpoint .pth")
    parser.add_argument("--out", type=str, default=cfg.onnx_path, help="Ruta de salida .onnx")
    parser.add_argument("--opset", type=int, default=cfg.onnx_opset)
    args = parser.parse_args()

    model = SC2HailoV2Actor(
        stem_channels=cfg.stem_channels,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        num_layers=cfg.num_layers,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
    )
    print(f"[Export ONNX V2] SC2HailoV2Actor ({count_parameters(model):,} parámetros)")

    if os.path.exists(args.weights):
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=True)
        print(f"[Export ONNX V2] Pesos cargados: {args.weights}")
    else:
        print(f"[Aviso] No existe '{args.weights}'. Exportando pesos iniciales para verificación de grafo.")

    export_v2_to_onnx(model, onnx_path=args.out, opset_version=args.opset)


if __name__ == "__main__":
    main()
