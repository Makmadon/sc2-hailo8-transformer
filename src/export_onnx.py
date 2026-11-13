"""
Script para exportar el modelo SC2HailoTransformer a formato ONNX 100% estático
para compilación en Hailo Dataflow Compiler (WSL / Colab).
"""

import argparse
import os
import torch

from src.config import IN_CHANNELS, GRID_H, GRID_W
from src.model import SC2HailoTransformer, export_to_hailo_onnx


def main():
    parser = argparse.ArgumentParser(description="Exportar SC2HailoTransformer a ONNX estático")
    parser.add_argument("--weights", type=str, default="models/sc2_hailo8_transformer_best.pth", help="Ruta al checkpoint .pth")
    parser.add_argument("--out", type=str, default="models/sc2_hailo8_transformer.onnx", help="Ruta de salida .onnx")
    args = parser.parse_args()

    print(f"📦 Inicializando SC2HailoTransformer (in_channels={IN_CHANNELS}, grid={GRID_H}x{GRID_W})...")
    model = SC2HailoTransformer(in_channels=IN_CHANNELS)

    if os.path.exists(args.weights):
        print(f"📥 Cargando pesos desde: {args.weights}")
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        # Adaptar si el checkpoint contiene metadata de entrenamiento
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print("✅ Pesos cargados correctamente.")
    else:
        print(f"⚠️ No se encontró checkpoint en '{args.weights}'. Exportando arquitectura con pesos iniciales.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    export_to_hailo_onnx(model, onnx_path=args.out)
    print(f"🎯 Modelo ONNX listo en: {args.out}")


if __name__ == "__main__":
    main()
