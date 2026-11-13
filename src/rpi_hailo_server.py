"""
Servidor de Inferencia Multihilo V2 para Raspberry Pi 5 + HAT Hailo-8 (o fallback ONNX Runtime).
Características V2:
  - Concurrencia TCP real (threading.Thread por cliente) con `infer_lock` sobre PCIe NPU.
  - Protocolo binario de baja latencia (sin JSON):
      Req:  magic(uint32) + map_tensor(16x32x32 float32) + global_vec(64 float32) = 65,796 bytes
      Resp: magic(uint32) + backend_code(uint32) + latency_ms(float32) +
            macro_logits(28 float32) + micro_logits(12 float32) + spatial_logits(2x32x32 float32) = 8,364 bytes
  - Devuelve los LOGITS brutos al cliente PC para aplicar Action Masking exacto y muestreo PPO estocástico.
"""

import argparse
import socket
import struct
import threading
import time
from typing import Tuple

import numpy as np

try:
    from .config import (
        GLOBAL_DIM,
        GRID_H,
        GRID_W,
        MAP_BYTES,
        MAP_CHANNELS,
        NUM_MACRO,
        NUM_MICRO,
        PROTOCOL_MAGIC,
        REQ_BYTES,
        VEC_BYTES,
        V2Config,
    )
except ImportError:
    from config import (
        GLOBAL_DIM,
        GRID_H,
        GRID_W,
        MAP_BYTES,
        MAP_CHANNELS,
        NUM_MACRO,
        NUM_MICRO,
        PROTOCOL_MAGIC,
        REQ_BYTES,
        VEC_BYTES,
        V2Config,
    )

import time


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return b""
        buf.extend(chunk)
    return bytes(buf)


class Hailo8V2InferenceEngine:
    """Motor multihilo con soporte nativo para Hailo-8 (.hef) y fallback ONNX Runtime (.onnx)."""

    def __init__(self, hef_path: str, onnx_path: str):
        self.backend = "none"
        self.backend_code = 0
        self.infer_lock = threading.Lock()
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
            self.network_group = self.target.configure(self.hef, configure_params)[0]
            self.network_group_params = self.network_group.create_params()
            self.input_vstreams_params = InputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
            self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)

            # Identificar nombres exactos de los vstreams de entrada y salida por forma
            self.map_input_name = None
            self.vec_input_name = None
            for info in self.hef.get_input_vstream_infos():
                if tuple(info.shape) == (GRID_H, GRID_W, MAP_CHANNELS) or "map" in info.name.lower():
                    self.map_input_name = info.name
                else:
                    self.vec_input_name = info.name

            self.out_macro_name = None
            self.out_micro_name = None
            self.out_spatial_name = None
            for info in self.hef.get_output_vstream_infos():
                shp = tuple(info.shape)
                if shp == (1, 1, NUM_MACRO) or shp == (NUM_MACRO,) or "macro" in info.name.lower():
                    self.out_macro_name = info.name
                elif shp == (1, 1, NUM_MICRO) or shp == (NUM_MICRO,) or "micro" in info.name.lower():
                    self.out_micro_name = info.name
                else:
                    self.out_spatial_name = info.name

            self.backend = "hailo8"
            self.backend_code = 1
            self._start_persistent_hailo()
            print(
                f"[Hailo-8 V2] NPU activa ({hef_path}) | "
                f"Inputs=({self.map_input_name}, {self.vec_input_name}) | "
                f"Outputs=({self.out_macro_name}, {self.out_micro_name}, {self.out_spatial_name})"
            )
        except Exception as e:
            print(f"[Aviso V2] hailo_platform no disponible ({e}). Intentando ONNX Runtime...")
            import onnxruntime as ort

            self.ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.backend = "onnxruntime"
            self.backend_code = 2
            print(f"[ONNX Runtime V2] Modelo cargado en CPU: {onnx_path}")

    def _start_persistent_hailo(self):
        try:
            from hailo_platform import InferVStreams

            self.activation_ctx = self.network_group.activate(self.network_group_params)
            self.activation = self.activation_ctx.__enter__()
            self.pipeline_ctx = InferVStreams(
                self.network_group, self.input_vstreams_params, self.output_vstreams_params
            )
            self.pipeline = self.pipeline_ctx.__enter__()
            self._persistent_pipeline = True
        except Exception as exc:
            print(f"[Hailo-8 Aviso] Modo persistente no disponible ({exc}), activando por frame.")
            self._persistent_pipeline = False

    def infer_raw_logits(
        self,
        map_nchw: np.ndarray,
        vec_nchw: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retorna (macro_logits [28], micro_logits [12], spatial_logits [2, 32, 32]) float32."""
        if self.backend == "hailo8":
            map_nhwc = np.ascontiguousarray(np.transpose(map_nchw, (0, 2, 3, 1)), dtype=np.float32)
            vec_nhwc = np.ascontiguousarray(np.transpose(vec_nchw, (0, 2, 3, 1)), dtype=np.float32)
            inputs = {self.map_input_name: map_nhwc, self.vec_input_name: vec_nhwc}

            with self.infer_lock:
                if self._persistent_pipeline:
                    try:
                        res = self.pipeline.infer(inputs)
                    except Exception:
                        self._start_persistent_hailo()
                        res = self.pipeline.infer(inputs)
                else:
                    from hailo_platform import InferVStreams

                    with InferVStreams(
                        self.network_group, self.input_vstreams_params, self.output_vstreams_params
                    ) as pipe:
                        with self.network_group.activate(self.network_group_params):
                            res = pipe.infer(inputs)

            macro_out = np.asarray(res[self.out_macro_name], dtype=np.float32).reshape(NUM_MACRO)
            micro_out = np.asarray(res[self.out_micro_name], dtype=np.float32).reshape(NUM_MICRO)
            sp_raw = np.asarray(res[self.out_spatial_name], dtype=np.float32)
            if sp_raw.shape == (1, GRID_H, GRID_W, 2):
                spatial_out = np.transpose(sp_raw, (0, 3, 1, 2)).reshape(2, GRID_H, GRID_W)
            else:
                spatial_out = sp_raw.reshape(2, GRID_H, GRID_W)
            return np.ascontiguousarray(macro_out), np.ascontiguousarray(micro_out), np.ascontiguousarray(spatial_out)

        with self.infer_lock:
            outputs = self.ort_session.run(
                ["macro_logits", "micro_logits", "spatial_logits"],
                {"map_tensor": map_nchw.astype(np.float32), "global_vec": vec_nchw.astype(np.float32)},
            )
        return (
            outputs[0].astype(np.float32).reshape(NUM_MACRO),
            outputs[1].astype(np.float32).reshape(NUM_MICRO),
            outputs[2].astype(np.float32).reshape(2, GRID_H, GRID_W),
        )


def handle_client(conn: socket.socket, addr: tuple, engine: Hailo8V2InferenceEngine):
    req_frame_len = 4 + REQ_BYTES
    with conn:
        print(f"[RPi V2] Cliente conectado: {addr}")
        while True:
            raw = _recv_exact(conn, req_frame_len)
            if not raw:
                print(f"[RPi V2] Cliente desconectado: {addr}")
                break

            magic = struct.unpack("!I", raw[:4])[0]
            if magic != PROTOCOL_MAGIC:
                print(f"[RPi V2] Magic inválido ({hex(magic)}) desde {addr}. Cerrando.")
                break

            t0 = time.perf_counter()
            map_arr = np.frombuffer(raw, dtype=np.float32, count=MAP_CHANNELS * GRID_H * GRID_W, offset=4).reshape(
                1, MAP_CHANNELS, GRID_H, GRID_W
            )
            vec_arr = np.frombuffer(raw, dtype=np.float32, count=GLOBAL_DIM, offset=4 + MAP_BYTES).reshape(
                1, GLOBAL_DIM, 1, 1
            )

            try:
                macro_l, micro_l, spatial_l = engine.infer_raw_logits(map_arr, vec_arr)
                latency_ms = float((time.perf_counter() - t0) * 1000.0)
                header = struct.pack("!IIf", PROTOCOL_MAGIC, engine.backend_code, latency_ms)
                conn.sendall(header + macro_l.tobytes() + micro_l.tobytes() + spatial_l.tobytes())
            except Exception as exc:
                print(f"[RPi V2 Error] {addr}: {exc}")
                break


def run_server(host: str, port: int, hef_path: str, onnx_path: str):
    engine = Hailo8V2InferenceEngine(hef_path=hef_path, onnx_path=onnx_path)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(16)
        print(f"[RPi V2 Servidor Multihilo] Escuchando en {host}:{port} (Backend={engine.backend})...")
        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            th = threading.Thread(target=handle_client, args=(conn, addr, engine), daemon=True)
            th.start()


if __name__ == "__main__":
    cfg = V2Config()
    parser = argparse.ArgumentParser(description="Servidor de Inferencia Multihilo V2 (RPi5 + Hailo-8)")
    parser.add_argument("--host", default=cfg.lan_host)
    parser.add_argument("--port", type=int, default=cfg.lan_port)
    parser.add_argument("--hef", default=cfg.hef_path)
    parser.add_argument("--onnx", default=cfg.onnx_path)
    args = parser.parse_args()
    run_server(args.host, args.port, args.hef, args.onnx)
