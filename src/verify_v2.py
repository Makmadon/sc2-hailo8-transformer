"""
Suite de verificación local V2 (sin necesidad de StarCraft II ni NPU Hailo-8).
Se adapta automáticamente al entorno:
  - En Windows SC2 (burnysc2 + numpy, sin torch/onnx): valida módulos cliente, masking NumPy,
    rollout storage, parser offline, proyecciones 32x32 y protocolo TCP binario multihilo.
  - En WSL/GPU (torch + onnx + onnxruntime): además valida SC2HailoV2Actor/Critic, paridad ONNX/ORT,
    GAE/PPO completo, ancla KL inmutable y servidor con motor ONNX Runtime.
"""

import argparse
import importlib.util
import os
import socket
import struct
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import numpy as np

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
    SPATIAL_CELLS,
    SPATIAL_MACRO_IDS,
    SPATIAL_MICRO_IDS,
    V2Config,
)
from .features import EpisodeTrackerV2, grid_to_world, world_to_grid
from .pc_sc2_client import RESP_TOTAL_BYTES, SC2V2LANClient, _masked_softmax_sample, append_v2_rollouts
from .replay_dataset import parse_sc2_dynamics_terran_v2


def _has_pkg(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def test_core_sc2_client_and_parser(tmpdir: str):
    print("[Core 1/3] Verificando config, features 32x32, masking NumPy y rollout lock...")
    assert REQ_BYTES == MAP_BYTES + GLOBAL_DIM * 4
    assert RESP_TOTAL_BYTES == 12 + (NUM_MACRO + NUM_MICRO + 2 * SPATIAL_CELLS) * 4

    bounds = (10.0, 12.0, 128.0, 128.0)
    wx, wy = grid_to_world(16, 8, bounds)
    gx, gy = world_to_grid(SimpleNamespace(x=wx, y=wy), bounds)
    assert (gx, gy) == (16, 8), f"Idempotencia fallida: {(gx, gy)}"

    tracker = EpisodeTrackerV2()
    tracker.update_action(macro_id=3, micro_id=4)
    assert tracker.steps_since_attack == 0 and tracker.ema_econ > 0.0 and tracker.ema_combat > 0.0
    tracker.update_action(macro_id=0, micro_id=0)
    assert tracker.steps_since_attack == 1

    # Verificar _masked_softmax_sample (enmascara acciones inválidas y calcula log_prob temp=1.0)
    rng = np.random.default_rng(7)
    logits = np.array([1.0, 2.0, 1e6, -3.0], dtype=np.float32)
    mask = np.array([True, True, False, False], dtype=bool)
    for _ in range(25):
        act, lp = _masked_softmax_sample(logits, mask, rng, temperature=2.0, greedy=False)
        assert act in (0, 1), f"Acción inválida seleccionada a pesar de máscara: {act}"
        assert np.isfinite(lp) and lp < 0.0

    # Verificar append_v2_rollouts en archivo temporal
    rollout_file = os.path.join(tmpdir, "rollouts.npz")
    ep_data = {
        "map_obs": np.zeros((4, MAP_CHANNELS, GRID_H, GRID_W), dtype=np.float32),
        "vec_obs": np.zeros((4, GLOBAL_DIM, 1, 1), dtype=np.float32),
        "macro_mask": np.ones((4, NUM_MACRO), dtype=bool),
        "micro_mask": np.ones((4, NUM_MICRO), dtype=bool),
        "macro_id": np.zeros(4, dtype=np.int64),
        "micro_id": np.zeros(4, dtype=np.int64),
        "macro_xy": np.zeros(4, dtype=np.int64),
        "micro_xy": np.zeros(4, dtype=np.int64),
        "old_lp_macro": np.full(4, -1.0, dtype=np.float32),
        "old_lp_micro": np.full(4, -1.0, dtype=np.float32),
        "old_lp_s_macro": np.full(4, -6.9, dtype=np.float32),
        "old_lp_s_micro": np.full(4, -6.9, dtype=np.float32),
        "rewards": np.ones(4, dtype=np.float32),
        "dones": np.array([False, False, False, True], dtype=bool),
    }
    n_total = append_v2_rollouts(ep_data, ep_won=1, rollout_path=rollout_file)
    assert n_total == 4
    n_total = append_v2_rollouts(ep_data, ep_won=0, rollout_path=rollout_file)
    assert n_total == 8

    # Verificar parser Terran offline
    sample = (
        "#### Current Observation:\n"
        "Race: Terran | Enemy Race: Zerg | Time: 03:15\n"
        "Minerals: 250 (+420/min) | Gas: 100 (+180/min)\n"
        "Supply: 32/46 (Army: 12, Workers: 20)\n"
        "Own Units: SCV(20), Marine(8), Marauder(2), CommandCenter(2), Barracks(2), SupplyDepot(3)\n"
        "Enemy Units: Zergling(10) at (40, 42)\n"
        "#### Action Sequence:\n"
        "Build Barracks at (18, 22)\n"
    )
    parsed = parse_sc2_dynamics_terran_v2(sample)
    assert parsed is not None
    assert parsed["map_obs"].shape == (MAP_CHANNELS, GRID_H, GRID_W)
    assert parsed["vec_obs"].shape == (GLOBAL_DIM, 1, 1)
    assert bool(parsed["macro_mask"][parsed["macro_id"]]) is True
    assert bool(parsed["micro_mask"][parsed["micro_id"]]) is True
    print("  OK | Config, features 32x32, masking, lock rollouts y parser offline verificados.")


def test_wire_protocol_mock_tcp():
    print("[Core 2/3] Verificando protocolo binario TCP multihilo (modo determinista sin ORT)...")
    from .rpi_hailo_server import _recv_exact

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port = srv.getsockname()[1]

    def _mock_worker(conn: socket.socket):
        with conn:
            while True:
                raw = _recv_exact(conn, 4 + REQ_BYTES)
                if not raw:
                    break
                magic = struct.unpack("!I", raw[:4])[0]
                assert magic == PROTOCOL_MAGIC
                m = np.frombuffer(raw, dtype=np.float32, count=MAP_CHANNELS * GRID_H * GRID_W, offset=4)
                v = np.frombuffer(raw, dtype=np.float32, count=GLOBAL_DIM, offset=4 + MAP_BYTES)
                # Respuesta determinista basada en los inputs para verificar integridad byte-a-byte
                macro_l = np.full(NUM_MACRO, float(v[0]), dtype=np.float32)
                micro_l = np.full(NUM_MICRO, float(v[1]), dtype=np.float32)
                spatial_l = np.full((2, GRID_H, GRID_W), float(m[0]), dtype=np.float32)
                header = struct.pack("!IIf", PROTOCOL_MAGIC, 99, 0.25)
                conn.sendall(header + macro_l.tobytes() + micro_l.tobytes() + spatial_l.tobytes())

    def _accept_loop():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                break
            threading.Thread(target=_mock_worker, args=(conn,), daemon=True).start()

    threading.Thread(target=_accept_loop, daemon=True).start()
    try:
        map_in = np.full((1, MAP_CHANNELS, GRID_H, GRID_W), 0.375, dtype=np.float32)
        vec_in = np.zeros((1, GLOBAL_DIM, 1, 1), dtype=np.float32)
        vec_in[0, 0, 0, 0] = 1.25
        vec_in[0, 1, 0, 0] = -0.5

        c1 = SC2V2LANClient("127.0.0.1", port=port, timeout=5.0)
        c2 = SC2V2LANClient("127.0.0.1", port=port, timeout=5.0)
        try:
            ml, mil, spl, npu_ms, bcode = c1.request_logits(map_in, vec_in)
            ml2, mil2, spl2, _, bcode2 = c2.request_logits(map_in, vec_in)
        finally:
            c1.close()
            c2.close()

        assert bcode == 99 and bcode2 == 99 and abs(npu_ms - 0.25) < 1e-4
        assert np.allclose(ml, 1.25) and np.allclose(mil, -0.5) and np.allclose(spl, 0.375)
        assert np.allclose(ml2, 1.25) and np.allclose(mil2, -0.5) and np.allclose(spl2, 0.375)
        print("  OK | Protocolo TCP binario 65,796B -> 8,364B (dual cliente) verificado.")
    finally:
        srv.close()
        time.sleep(0.05)


def test_compile_script_import():
    print("[Core 3/3] Verificando importación de compilador HEF y rutas V2...")
    from .compile_hef import get_default_paths

    paths = get_default_paths()
    assert paths["onnx"].endswith("sc2_v2_actor.onnx")
    assert paths["hef"].endswith("sc2_v2_agent.hef")
    assert paths["calib"].endswith("sc2_v2_calib.npz")
    print("  OK | compile_hef.get_default_paths() verificado.")


def _make_synthetic_transitions(n: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    map_obs = rng.uniform(0.0, 1.0, size=(n, MAP_CHANNELS, GRID_H, GRID_W)).astype(np.float32)
    vec_obs = rng.uniform(0.0, 1.0, size=(n, GLOBAL_DIM, 1, 1)).astype(np.float32)
    macro_mask = np.zeros((n, NUM_MACRO), dtype=bool)
    micro_mask = np.zeros((n, NUM_MICRO), dtype=bool)
    macro_mask[:, 0] = True
    micro_mask[:, 0] = True

    sp_macro_list = sorted(SPATIAL_MACRO_IDS)
    sp_micro_list = sorted(SPATIAL_MICRO_IDS)

    macro_id = np.zeros(n, dtype=np.int64)
    micro_id = np.zeros(n, dtype=np.int64)
    for i in range(n):
        m_choice = int(rng.choice([0, 1, 2, sp_macro_list[0], sp_macro_list[1]]))
        mi_choice = int(rng.choice([0, 7, sp_micro_list[0], sp_micro_list[1]]))
        macro_id[i] = m_choice
        micro_id[i] = mi_choice
        macro_mask[i, m_choice] = True
        micro_mask[i, mi_choice] = True

    macro_xy = rng.integers(0, SPATIAL_CELLS, size=n, dtype=np.int64)
    micro_xy = rng.integers(0, SPATIAL_CELLS, size=n, dtype=np.int64)
    return map_obs, vec_obs, macro_mask, micro_mask, macro_id, micro_id, macro_xy, micro_xy


def test_torch_training_and_onnx(tmpdir: str, run_onnx: bool):
    import torch
    from .model import (
        SC2HailoV2Actor,
        SC2HailoV2Critic,
        apply_action_mask,
        count_parameters,
        export_v2_to_onnx,
        masked_categorical_stats,
        masked_kl_divergence,
    )
    from .train_bc import train_bc
    from .train_rl import compute_gae_returns, train_ppo_rl

    print("[Torch 1/3] Verificando SC2HailoV2Actor, SC2HailoV2Critic, Action Masking y GAE...")
    cfg = V2Config()
    actor = SC2HailoV2Actor(
        stem_channels=cfg.stem_channels,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        num_layers=cfg.num_layers,
        mlp_ratio=cfg.mlp_ratio,
        dropout=0.0,
    ).eval()
    critic = SC2HailoV2Critic(hidden_dim=cfg.d_model).eval()

    n_params = count_parameters(actor)
    assert n_params < 600_000, f"Actor demasiado pesado: {n_params}"

    rng = np.random.default_rng(123)
    map_np = rng.normal(size=(2, MAP_CHANNELS, GRID_H, GRID_W)).astype(np.float32)
    vec_np = rng.normal(size=(2, GLOBAL_DIM, 1, 1)).astype(np.float32)

    with torch.no_grad():
        m_pt, mi_pt, sp_pt = actor(torch.from_numpy(map_np), torch.from_numpy(vec_np))
        v_pt = critic(torch.from_numpy(map_np), torch.from_numpy(vec_np))

    assert m_pt.shape == (2, NUM_MACRO, 1, 1)
    assert mi_pt.shape == (2, NUM_MICRO, 1, 1)
    assert sp_pt.shape == (2, 2, GRID_H, GRID_W)
    assert v_pt.shape == (2,)

    logits = torch.tensor([[1.0, 2.0, 1e5, 0.5]])
    ref_logits = torch.tensor([[0.5, 1.5, -1e5, 0.5]])
    mask = torch.tensor([[True, True, False, True]])
    assert apply_action_mask(logits, mask)[0, 2].item() < -1e8
    lp, ent = masked_categorical_stats(logits, torch.tensor([1]), mask)
    kl = masked_kl_divergence(logits, ref_logits, mask)
    assert torch.isfinite(lp).all() and torch.isfinite(ent).all() and kl.item() < 1.0

    rewards = np.array([0.0, 1.0, 0.0, -5.0], dtype=np.float32)
    dones = np.array([False, True, False, True], dtype=bool)
    adv, ret = compute_gae_returns(rewards, dones, np.zeros(4, dtype=np.float32), gamma=0.99, gae_lambda=0.95)
    assert np.isfinite(adv).all() and ret[0] > 0.0 and ret[1] > 0.0 and ret[3] < 0.0
    print(f"  OK | Actor={n_params:,} params | Shapes + Masking + GAE verificados.")

    print("[Torch 2/3] Verificando pipeline completo BC -> Anchor KL -> PPO + Critic...")
    tmp_cfg = V2Config(
        calib_samples=16,
        actor_weights=os.path.join(tmpdir, "sc2_v2_actor.pth"),
        bc_anchor_weights=os.path.join(tmpdir, "sc2_v2_bc_anchor.pth"),
        critic_weights=os.path.join(tmpdir, "sc2_v2_critic.pth"),
        onnx_path=os.path.join(tmpdir, "sc2_v2_actor.onnx"),
        calib_path=os.path.join(tmpdir, "sc2_v2_calib.npz"),
        rollout_path=os.path.join(tmpdir, "sc2_v2_rollouts.npz"),
        bc_dataset_path=os.path.join(tmpdir, "sc2_v2_bc_dataset.npz"),
    )

    m_obs, v_obs, mm, mim, mid, miid, mxy, mixy = _make_synthetic_transitions(48, seed=11)
    np.savez_compressed(
        tmp_cfg.bc_dataset_path,
        map_obs=m_obs, vec_obs=v_obs, macro_mask=mm, micro_mask=mim,
        macro_id=mid, micro_id=miid, macro_xy=mxy, micro_xy=mixy,
    )
    train_bc(
        dataset_path=tmp_cfg.bc_dataset_path, epochs=2, batch_size=16, lr=5e-4,
        device_arg="cpu", cpu_threads=2, num_workers=0, export_onnx=run_onnx, cfg=tmp_cfg,
    )
    assert os.path.exists(tmp_cfg.actor_weights) and os.path.exists(tmp_cfg.bc_anchor_weights)
    assert os.path.exists(tmp_cfg.calib_path)
    with open(tmp_cfg.bc_anchor_weights, "rb") as f:
        anchor_before = f.read()

    rm_obs, rv_obs, rmm, rmim, rmid, rmiid, rmxy, rmixy = _make_synthetic_transitions(32, seed=29)
    dones_rl = np.zeros(32, dtype=bool)
    dones_rl[15] = True
    dones_rl[31] = True
    np.savez_compressed(
        tmp_cfg.rollout_path,
        map_obs=rm_obs, vec_obs=rv_obs, macro_mask=rmm, micro_mask=rmim,
        macro_id=rmid, micro_id=rmiid, macro_xy=rmxy, micro_xy=rmixy,
        old_lp_macro=np.full(32, -1.2, dtype=np.float32),
        old_lp_micro=np.full(32, -1.0, dtype=np.float32),
        old_lp_s_macro=np.full(32, -6.9, dtype=np.float32),
        old_lp_s_micro=np.full(32, -6.9, dtype=np.float32),
        rewards=np.linspace(-0.05, 0.25, 32, dtype=np.float32),
        dones=dones_rl,
        ep_wins=np.array([1, 0], dtype=np.int64),
    )
    train_ppo_rl(
        rollout_path=tmp_cfg.rollout_path, ppo_epochs=2, critic_epochs=2, batch_size=16,
        actor_lr=1e-4, critic_lr=5e-4, device_arg="cpu", cpu_threads=2,
        export_onnx=run_onnx, cfg=tmp_cfg,
    )
    assert os.path.exists(tmp_cfg.critic_weights)
    with open(tmp_cfg.bc_anchor_weights, "rb") as f:
        assert f.read() == anchor_before, "El entrenamiento RL modificó el ancla BC!"
    print("  OK | BC -> Calib -> PPO + Critic + Ancla KL inmutable verificados.")

    if run_onnx:
        import onnx
        import onnxruntime as ort
        from .rpi_hailo_server import Hailo8V2InferenceEngine, handle_client

        print("[Torch/ONNX 3/3] Verificando exportación ONNX, paridad ORT y servidor TCP con motor ORT...")
        onnx_path = tmp_cfg.onnx_path
        export_v2_to_onnx(actor, onnx_path=onnx_path, opset_version=cfg.onnx_opset)
        onnx.checker.check_model(onnx.load(onnx_path))

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        m_ort, mi_ort, sp_ort = sess.run(
            ["macro_logits", "micro_logits", "spatial_logits"],
            {"map_tensor": map_np[:1], "global_vec": vec_np[:1]},
        )
        with torch.no_grad():
            m_ref, mi_ref, sp_ref = actor(torch.from_numpy(map_np[:1]), torch.from_numpy(vec_np[:1]))
        max_diff = max(
            float(np.max(np.abs(m_ref.numpy() - m_ort))),
            float(np.max(np.abs(mi_ref.numpy() - mi_ort))),
            float(np.max(np.abs(sp_ref.numpy() - sp_ort))),
        )
        assert max_diff < 1e-4, f"Divergencia PyTorch vs ONNX Runtime: {max_diff}"

        engine = Hailo8V2InferenceEngine(hef_path=os.path.join(tmpdir, "missing.hef"), onnx_path=onnx_path)
        assert engine.backend == "onnxruntime"
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(4)
        port = srv.getsockname()[1]

        def _accept_loop():
            while True:
                try:
                    conn, addr = srv.accept()
                except OSError:
                    break
                threading.Thread(target=handle_client, args=(conn, addr, engine), daemon=True).start()

        threading.Thread(target=_accept_loop, daemon=True).start()
        try:
            ref_m, ref_mi, ref_sp = engine.infer_raw_logits(map_np[:1], vec_np[:1])
            c = SC2V2LANClient("127.0.0.1", port=port, timeout=5.0)
            try:
                ml, mil, spl, npu_ms, bcode = c.request_logits(map_np[:1], vec_np[:1])
            finally:
                c.close()
            assert bcode == 2 and npu_ms >= 0.0
            assert max(np.max(np.abs(ml - ref_m)), np.max(np.abs(mil - ref_mi)), np.max(np.abs(spl - ref_sp))) < 1e-5
            print(f"  OK | Paridad ONNX={max_diff:.2e} y TCP Hailo8V2InferenceEngine (ORT)={npu_ms:.2f}ms.")
        finally:
            srv.close()
            time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description="Verificación local SC2-Hailo8 V2")
    parser.add_argument("--require-all", action="store_true", help="Falla si faltan torch/onnx/onnxruntime")
    args = parser.parse_args()

    has_torch = _has_pkg("torch")
    has_onnx = _has_pkg("onnx") and _has_pkg("onnxruntime")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_core_sc2_client_and_parser(tmpdir)
        test_wire_protocol_mock_tcp()
        test_compile_script_import()

        if has_torch:
            test_torch_training_and_onnx(tmpdir, run_onnx=has_onnx)
            if not has_onnx:
                msg = "[SKIPPED] Faltan 'onnx'/'onnxruntime': checks ONNX/ORT omitidos."
                print(msg)
                if args.require_all:
                    sys.exit(2)
        else:
            msg = "[SKIPPED] No hay 'torch' en este intérprete: checks Actor/Critic, BC, PPO y ONNX omitidos."
            print(msg)
            if args.require_all:
                sys.exit(2)

    print("\n✅ Verificación completada en este entorno.")


if __name__ == "__main__":
    main()
