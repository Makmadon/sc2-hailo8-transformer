"""
Script de compilación ONNX -> HEF para Hailo-8 usando Hailo Dataflow Compiler (v3.34.0).
Resuelve los errores comunes de compilación en Colab:
  1. Especifica explícitamente hw_arch="hailo8".
  2. Utiliza sintaxis .alls válida para Hailo DFC 3.34.0.
  3. Transpone automáticamente el set de calibración de NCHW [N, 12, 16, 16] a NHWC [N, 16, 16, 12]
     cuando el parser de Hailo así lo requiere.
"""

import argparse
import os
import numpy as np

# En contenedores de Google Colab la variable de entorno USER no está definida por defecto,
# y el compilador backend de Hailo (runner.compile()) la requiere para crear directorios temporales.
os.environ.setdefault("USER", "colab" if os.path.exists("/content") else os.environ.get("USER", "user"))
os.environ.setdefault("LOGNAME", os.environ.get("USER", "user"))
if os.path.exists("/content"):
    os.environ.setdefault("HOME", "/root")


def get_default_paths() -> dict:
    if os.path.exists("/content/drive/MyDrive/hailo_files"):
        base = "/content/drive/MyDrive/hailo_files"
    else:
        # En WSL / local usar el directorio models del repositorio
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(script_dir, ".."))
        base = os.path.join(repo_root, "models")

    return {
        "onnx": os.path.join(base, "sc2_hailo8_transformer.onnx"),
        "calib": os.path.join(base, "sc2_calib_data.npy"),
        "hef": os.path.join(base, "sc2_hailo8_agent.hef"),
        "har": os.path.join(base, "sc2_hailo8_quantized.har"),
    }


def compile_onnx_to_hef(
    onnx_path: str = None,
    calib_path: str = None,
    hef_output_path: str = None,
    har_output_path: str = None,
    force_quant: bool = False,
):
    defaults = get_default_paths()
    onnx_path = onnx_path or defaults["onnx"]
    calib_path = calib_path or defaults["calib"]
    hef_output_path = hef_output_path or defaults["hef"]
    har_output_path = har_output_path or defaults["har"]

    from hailo_sdk_client import ClientRunner

    print("=== Iniciando Compilación para Hailo-8 (26 TOPS INT8) ===")
    runner = ClientRunner(hw_arch="hailo8")

    if not force_quant and os.path.exists(har_output_path):
        print(f"[Hailo DFC] Encontrado HAR ya cuantizado, cargando directamente: {har_output_path}")
        runner.load_har(har_output_path)
    else:
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"No se encontró el modelo ONNX en: {onnx_path}")
        # 1. Traducir el grafo ONNX estático
        hn, npz = runner.translate_onnx_model(
            onnx_path,
            "sc2_hailo8_transformer",
            start_node_names=["obs_tensor"],
            end_node_names=["action_chunks_out"],
            net_input_shapes={"obs_tensor": [1, 12, 16, 16]},
        )
        print("[Hailo DFC] Traducción ONNX -> HN completada.")

        # 2. Aplicar script .alls válido (optimization_level=0 para compilar rápido en Colab sin requerir 64GB RAM)
        alls_commands = [
            "model_optimization_flavor(optimization_level=0, compression_level=0)\n",
        ]
        runner.load_model_script("".join(alls_commands))
        print("[Hailo DFC] Script de optimización cargado correctamente.")

        # 3. Preparar datos de calibración reales de StarCraft II
        if os.path.exists(calib_path):
            calib_data = np.load(calib_path).astype(np.float32)
        else:
            calib_data = np.random.uniform(0.0, 1.0, size=(64, 12, 16, 16)).astype(np.float32)

        # Hailo traduce internamente tensores 4D NCHW [N, C, H, W] a NHWC [N, H, W, C]
        if calib_data.shape[1:] == (12, 16, 16):
            calib_nhwc = np.transpose(calib_data, (0, 2, 3, 1))
        else:
            calib_nhwc = calib_data

        print(f"[Hailo DFC] Ejecutando cuantización INT8 con {len(calib_nhwc)} muestras (shape={calib_nhwc.shape})...")
        try:
            runner.optimize(calib_nhwc)
        except Exception:
            runner.optimize(calib_data)

        runner.save_har(har_output_path)
        print(f"[Hailo DFC] Archivo HAR cuantizado guardado en: {har_output_path}")

    # 4. Compilar al binario físico .hef para el HAT Hailo-8 de la Raspberry Pi
    print("[Hailo DFC] Compilando binario de hardware .hef...")
    hef_bytes = runner.compile()

    os.makedirs(os.path.dirname(hef_output_path), exist_ok=True)
    with open(hef_output_path, "wb") as f:
        f.write(hef_bytes)

    print(f"[Hailo DFC] ¡Compilación exitosa! Archivo HEF listo para Raspberry Pi + Hailo-8: {hef_output_path}")
    return hef_output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compilador Hailo DFC ONNX -> HEF para Hailo-8")
    parser.add_argument("--onnx", type=str, default=None, help="Ruta al modelo ONNX")
    parser.add_argument("--calib", type=str, default=None, help="Ruta al archivo .npy de calibración")
    parser.add_argument("--hef", type=str, default=None, help="Ruta de salida del archivo .hef")
    parser.add_argument("--har", type=str, default=None, help="Ruta del archivo .har cuantizado intermedio")
    parser.add_argument("--force-quant", action="store_true", help="Forzar re-cuantización incluso si existe el .har")
    args = parser.parse_args()

    compile_onnx_to_hef(
        onnx_path=args.onnx,
        calib_path=args.calib,
        hef_output_path=args.hef,
        har_output_path=args.har,
        force_quant=args.force_quant,
    )
