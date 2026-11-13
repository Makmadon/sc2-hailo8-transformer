"""
Compilador ONNX -> HAR -> HEF V2 para Hailo-8 usando Hailo Dataflow Compiler (v3.34.0) en WSL/Linux.
Soporta automáticamente la arquitectura Dual-Input V2:
  - map_tensor: [1, 16, 32, 32] (NCHW -> NHWC [N, 32, 32, 16])
  - global_vec: [1, 64, 1, 1]   (NCHW -> NHWC [N, 1, 1, 64])
"""

import argparse
import os
from typing import Dict

import numpy as np

try:
    from .config import GLOBAL_DIM, GRID_H, GRID_W, MAP_CHANNELS, V2Config
except ImportError:
    from config import GLOBAL_DIM, GRID_H, GRID_W, MAP_CHANNELS, V2Config

os.environ.setdefault("USER", "colab" if os.path.exists("/content") else os.environ.get("USER", "wsl_user"))
os.environ.setdefault("LOGNAME", os.environ.get("USER", "wsl_user"))
if os.path.exists("/content"):
    os.environ.setdefault("HOME", "/root")


def get_default_paths() -> Dict[str, str]:
    cfg = V2Config()
    if os.path.exists("/content/drive/MyDrive/hailo_files"):
        base = "/content/drive/MyDrive/hailo_files"
        return {
            "onnx": os.path.join(base, os.path.basename(cfg.onnx_path)),
            "calib": os.path.join(base, os.path.basename(cfg.calib_path)),
            "hef": os.path.join(base, os.path.basename(cfg.hef_path)),
            "har": os.path.join(base, os.path.basename(cfg.har_path)),
        }
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    return {
        "onnx": os.path.join(repo_root, cfg.onnx_path),
        "calib": os.path.join(repo_root, cfg.calib_path),
        "hef": os.path.join(repo_root, cfg.hef_path),
        "har": os.path.join(repo_root, cfg.har_path),
    }


def _build_calib_dict(runner, calib_map_nchw: np.ndarray, calib_vec_nchw: np.ndarray) -> Dict[str, np.ndarray]:
    """Asocia las muestras de calibración NHWC a los nombres reales de input_layer en el HN de Hailo."""
    map_nhwc = np.transpose(calib_map_nchw, (0, 2, 3, 1)).astype(np.float32)  # [N, 32, 32, 16]
    vec_nhwc = np.transpose(calib_vec_nchw, (0, 2, 3, 1)).astype(np.float32)  # [N, 1, 1, 64]

    calib_dict: Dict[str, np.ndarray] = {}
    try:
        hn_dict = runner.get_hn_dict()
        for layer_name, info in hn_dict.get("layers", {}).items():
            if info.get("type") != "input_layer":
                continue
            out_shapes = info.get("output_shapes", [[]])
            shape = tuple(out_shapes[0][1:]) if out_shapes and len(out_shapes[0]) >= 4 else ()
            if shape == (GRID_H, GRID_W, MAP_CHANNELS) or "map" in layer_name.lower():
                calib_dict[layer_name] = map_nhwc
            elif shape == (1, 1, GLOBAL_DIM) or "vec" in layer_name.lower() or "global" in layer_name.lower():
                calib_dict[layer_name] = vec_nhwc
    except Exception:
        pass

    if len(calib_dict) < 2:
        calib_dict = {"map_tensor": map_nhwc, "global_vec": vec_nhwc}
    return calib_dict


def compile_onnx_to_hef(
    onnx_path: str = None,
    calib_path: str = None,
    hef_output_path: str = None,
    har_output_path: str = None,
    force_quant: bool = False,
    opt_level: int = 0,
):
    defaults = get_default_paths()
    onnx_path = onnx_path or defaults["onnx"]
    calib_path = calib_path or defaults["calib"]
    hef_output_path = hef_output_path or defaults["hef"]
    har_output_path = har_output_path or defaults["har"]

    from hailo_sdk_client import ClientRunner

    print("=== Iniciando Compilación V2 para Hailo-8 (26 TOPS INT8) ===")
    runner = ClientRunner(hw_arch="hailo8")

    if not force_quant and os.path.exists(har_output_path):
        print(f"[Hailo DFC V2] Cargando HAR existente: {har_output_path}")
        runner.load_har(har_output_path)
    else:
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"No se encontró el ONNX V2 en: {onnx_path}")

        runner.translate_onnx_model(
            onnx_path,
            "sc2_hailo8_v2",
            start_node_names=["map_tensor", "global_vec"],
            end_node_names=["macro_logits", "micro_logits", "spatial_logits"],
            net_input_shapes={
                "map_tensor": [1, MAP_CHANNELS, GRID_H, GRID_W],
                "global_vec": [1, GLOBAL_DIM, 1, 1],
            },
        )
        print("[Hailo DFC V2] Traducción ONNX -> HN completada.")

        alls_script = f"model_optimization_flavor(optimization_level={int(opt_level)}, compression_level=0)\n"
        runner.load_model_script(alls_script)

        if calib_path and os.path.exists(calib_path):
            cdata = np.load(calib_path)
            calib_map = cdata["map_tensor"].astype(np.float32)
            calib_vec = cdata["global_vec"].astype(np.float32)
        else:
            print("[Hailo DFC V2] Aviso: Sin calibración real, usando muestras aleatorias [0, 1] (solo prueba).")
            rng = np.random.default_rng(42)
            calib_map = rng.uniform(0.0, 1.0, size=(64, MAP_CHANNELS, GRID_H, GRID_W)).astype(np.float32)
            calib_vec = rng.uniform(0.0, 1.0, size=(64, GLOBAL_DIM, 1, 1)).astype(np.float32)

        calib_dict = _build_calib_dict(runner, calib_map, calib_vec)
        print(f"[Hailo DFC V2] Cuantizando INT8 con {len(calib_map)} muestras | layers={list(calib_dict.keys())}")
        runner.optimize(calib_dict)

        os.makedirs(os.path.dirname(os.path.abspath(har_output_path)), exist_ok=True)
        runner.save_har(har_output_path)
        print(f"[Hailo DFC V2] HAR guardado: {har_output_path}")

    print("[Hailo DFC V2] Compilando binario .hef para Hailo-8...")
    hef_bytes = runner.compile()

    os.makedirs(os.path.dirname(os.path.abspath(hef_output_path)), exist_ok=True)
    with open(hef_output_path, "wb") as f:
        f.write(hef_bytes)

    size_kb = os.path.getsize(hef_output_path) / 1024.0
    print(f"[Hailo DFC V2] ¡Compilación exitosa! HEF listo ({size_kb:.1f} KB): {hef_output_path}")
    return hef_output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compilador Hailo DFC V2 (ONNX -> HAR -> HEF)")
    parser.add_argument("--onnx", type=str, default=None)
    parser.add_argument("--calib", type=str, default=None)
    parser.add_argument("--hef", type=str, default=None)
    parser.add_argument("--har", type=str, default=None)
    parser.add_argument("--force-quant", action="store_true")
    parser.add_argument("--opt-level", type=int, default=0, help="Nivel optimización Hailo (0=rápido, 1-2=mayor precisión INT8)")
    args = parser.parse_args()

    compile_onnx_to_hef(
        onnx_path=args.onnx,
        calib_path=args.calib,
        hef_output_path=args.hef,
        har_output_path=args.har,
        force_quant=args.force_quant,
        opt_level=args.opt_level,
    )
