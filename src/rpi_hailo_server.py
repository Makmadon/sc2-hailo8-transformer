"""
Servidor de Inferencia en Tiempo Real para Raspberry Pi 5 + HAT Hailo-8.
Recibe el tensor de estado de StarCraft II [1, 12, 16, 16] por socket TCP desde la PC,
ejecuta la inferencia INT8 en el acelerador Hailo-8 (o fallback ONNX Runtime si se prueba sin el HAT),
y devuelve el Action-Chunk de K=4 acciones decodificadas en < 2 ms.
"""

import argparse
import json
import socket
import struct
import time
from typing import Dict, List

import numpy as np

try:
    from .config import (
        ACTION_CHUNK_SIZE,
        GRID_H,
        GRID_W,
        ID_TO_ACTION,
        IN_CHANNELS,
        NUM_ACTIONS,
        STEP_OUTPUT_DIM,
    )
except ImportError:
    from config import (
        ACTION_CHUNK_SIZE,
        GRID_H,
        GRID_W,
        ID_TO_ACTION,
        IN_CHANNELS,
        NUM_ACTIONS,
        STEP_OUTPUT_DIM,
    )


def decode_action_chunks(raw_output: np.ndarray) -> List[Dict[str, object]]:
    """
    Decodifica la salida del modelo [1, 66, 1, 4] o [1, 1, 4, 66] en una lista de K=4 órdenes listas para PySC2.
    """
    arr = np.squeeze(raw_output)
    # Asegurar forma [4, 66] (K=4 pasos, 66 logits por paso)
    if arr.shape == (STEP_OUTPUT_DIM, ACTION_CHUNK_SIZE):
        arr = arr.T
    elif arr.shape != (ACTION_CHUNK_SIZE, STEP_OUTPUT_DIM):
        arr = arr.reshape(ACTION_CHUNK_SIZE, STEP_OUTPUT_DIM)

    c1 = NUM_ACTIONS
    c2 = c1 + GRID_W
    c3 = c2 + GRID_H

    chunks: List[Dict[str, object]] = []
    for k in range(ACTION_CHUNK_SIZE):
        step_vec = arr[k]
        act_id = int(np.argmax(step_vec[:c1]))
        tx = int(np.argmax(step_vec[c1:c2]))
        ty = int(np.argmax(step_vec[c2:c3]))
        queue_flag = int(np.argmax(step_vec[c3:]))
        chunks.append({
            "step": k,
            "action_id": act_id,
            "action_name": ID_TO_ACTION.get(act_id, "NO_OP"),
            "target_grid": [tx, ty],
            "target_sc2_coord": [int((tx + 0.5) * (64.0 / GRID_W)), int((ty + 0.5) * (64.0 / GRID_H))],
            "queue": bool(queue_flag),
        })
    return chunks


class Hailo8InferenceEngine:
    """
    Motor de inferencia con soporte nativo para `hailo_platform` (.hef) en Raspberry Pi 5
    y modo de respaldo automático con `onnxruntime` (.onnx).
    """

    def __init__(self, hef_path: str = "sc2_hailo8_agent.hef", onnx_path: str = "sc2_hailo8_transformer.onnx"):
        self.backend = "none"
        self.hef_path = hef_path
        self.onnx_path = onnx_path
        self._persistent_pipeline = False

        try:
            from hailo_platform import (
                ConfigureParams,
                FormatType,
                HEF,
                HailoStreamInterface,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )

            self.hef = HEF(hef_path)
            self.target = VDevice()
            configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
            self.network_groups = self.target.configure(self.hef, configure_params)
            self.network_group = self.network_groups[0]
            self.network_group_params = self.network_group.create_params()
            self.input_vstreams_params = InputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
            self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
            self.input_name = self.hef.get_input_vstream_infos()[0].name
            self.backend = "hailo8"

            self._start_persistent_hailo()
            print(f"[Hailo-8 HAT] Pipeline persistente activado exitosamente en NPU PCIe: {hef_path}")
        except Exception as e:
            print(f"[Aviso] hailo_platform no disponible o sin HAT físico ({e}). Intentando ONNX Runtime...")
            try:
                import onnxruntime as ort
                self.ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                self.backend = "onnxruntime"
                print(f"[ONNX Runtime] Modelo cargado exitosamente: {onnx_path}")
            except Exception as ort_err:
                print(f"[Aviso] ONNX Runtime no instalado en este dispositivo ({ort_err}).")

    def _start_persistent_hailo(self):
        try:
            from hailo_platform import InferVStreams
            self.activation_ctx = self.network_group.activate(self.network_group_params)
            self.activation = self.activation_ctx.__enter__()
            self.pipeline_ctx = InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params)
            self.pipeline = self.pipeline_ctx.__enter__()
            self._persistent_pipeline = True
        except Exception as exc:
            print(f"[Hailo-8 Aviso] Modo persistente no soportado ({exc}), usando activación por frame.")
            self._persistent_pipeline = False

    def infer(self, obs_nchw: np.ndarray) -> List[Dict[str, object]]:
        if self.backend == "hailo8":
            obs_nhwc = np.transpose(obs_nchw, (0, 2, 3, 1)).astype(np.float32)
            try:
                if self._persistent_pipeline:
                    results = self.pipeline.infer({self.input_name: obs_nhwc})
                    raw_out = list(results.values())[0]
                else:
                    from hailo_platform import InferVStreams
                    with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as pipe:
                        with self.network_group.activate(self.network_group_params):
                            results = pipe.infer({self.input_name: obs_nhwc})
                            raw_out = list(results.values())[0]
            except Exception as e:
                print(f"[Hailo-8 Error] Fallo en inferencia ({e}). Reintentando con reinicio de pipeline...")
                self._start_persistent_hailo()
                if self._persistent_pipeline:
                    results = self.pipeline.infer({self.input_name: obs_nhwc})
                    raw_out = list(results.values())[0]
                else:
                    raise e
        else:
            raw_out = self.ort_session.run(None, {"obs_tensor": obs_nchw.astype(np.float32)})[0]

        return decode_action_chunks(raw_out)


def handle_client(
    conn: socket.socket,
    addr: tuple,
    engine: Hailo8InferenceEngine,
    expected_bytes: int,
):
    """Maneja la conexión persistente TCP de un cliente de StarCraft II de forma secuencial y directa."""
    with conn:
        print(f"[RPi Servidor] Cliente PC conectado desde {addr}")
        while True:
            buf = b""
            try:
                while len(buf) < expected_bytes:
                    packet = conn.recv(expected_bytes - len(buf))
                    if not packet:
                        break
                    buf += packet
                if len(buf) < expected_bytes:
                    print(f"[RPi Servidor] Conexión cerrada normalmente por {addr}.")
                    break

                t0 = time.perf_counter()
                obs_tensor = np.frombuffer(buf, dtype=np.float32).reshape(1, IN_CHANNELS, GRID_H, GRID_W)
                actions = engine.infer(obs_tensor)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                payload = json.dumps({
                    "latency_ms": round(latency_ms, 3),
                    "backend": engine.backend,
                    "chunks": actions,
                }).encode("utf-8")
                conn.sendall(struct.pack("!I", len(payload)) + payload)
            except Exception as exc:
                print(f"[RPi Servidor] Error atendiendo cliente {addr}: {exc}")
                try:
                    fallback_chunks = [
                        {"step": k, "action_id": 0, "action_name": "NO_OP", "target_grid": [8, 8], "target_sc2_coord": [32, 32], "queue": False}
                        for k in range(ACTION_CHUNK_SIZE)
                    ]
                    err_payload = json.dumps({
                        "latency_ms": 0.0,
                        "backend": "server_fallback",
                        "chunks": fallback_chunks,
                    }).encode("utf-8")
                    conn.sendall(struct.pack("!I", len(err_payload)) + err_payload)
                except Exception:
                    pass
                break


def run_server(
    host: str = "0.0.0.0",
    port: int = 5555,
    hef_path: str = "sc2_hailo8_agent.hef",
    onnx_path: str = "sc2_hailo8_transformer.onnx",
):
    engine = Hailo8InferenceEngine(hef_path=hef_path, onnx_path=onnx_path)
    expected_bytes = 1 * IN_CHANNELS * GRID_H * GRID_W * 4  # 12 * 16 * 16 * 4 = 12,288 bytes por frame

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        print(f"[RPi Servidor Dedicado] Escuchando en {host}:{port} (Backend: {engine.backend})...")

        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            handle_client(conn, addr, engine, expected_bytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Servidor de Inferencia SC2 en Raspberry Pi + Hailo-8")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--hef", default="models/sc2_hailo8_agent.hef")
    parser.add_argument("--onnx", default="models/sc2_hailo8_transformer.onnx")
    args = parser.parse_args()
    run_server(args.host, args.port, args.hef, args.onnx)
