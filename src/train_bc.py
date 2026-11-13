"""
Entrenamiento Supervisado V2 (Behavior Cloning con Action Masking y Heatmap Espacial 32x32).
Optimizado para correr tanto en CPU multi-hilo (ej. Intel i5-11400 en WSL/Windows) como en GPU.
"""

import argparse
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import (
    SPATIAL_MACRO_IDS,
    SPATIAL_MICRO_IDS,
    V2Config,
)
from .model import (
    SC2HailoV2Actor,
    apply_action_mask,
    count_parameters,
    export_v2_to_onnx,
)
from .replay_dataset import load_bc_splits


def select_device(device_arg: str, cpu_threads: int) -> torch.device:
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def compute_bc_loss(
    model: SC2HailoV2Actor,
    batch,
    device: torch.device,
    spatial_macro_ids: torch.Tensor,
    spatial_micro_ids: torch.Tensor,
):
    map_obs, vec_obs, m_mask, mi_mask, t_macro, t_micro, t_mxy, t_mixy = [x.to(device) for x in batch]

    macro_logits, micro_logits, spatial_logits = model(map_obs, vec_obs)
    macro_logits = macro_logits.squeeze(-1).squeeze(-1)   # [B, 28]
    micro_logits = micro_logits.squeeze(-1).squeeze(-1)   # [B, 12]

    # Enmascarar acciones inválidas antes de la pérdida supervisada
    masked_macro = apply_action_mask(macro_logits, m_mask)
    masked_micro = apply_action_mask(micro_logits, mi_mask)

    loss_macro = F.cross_entropy(masked_macro, t_macro)
    loss_micro = F.cross_entropy(masked_micro, t_micro)

    # Heatmap espacial conjunto 32x32 = 1,024 celdas, calculado solo cuando la acción usa coordenada
    b_size = map_obs.shape[0]
    s_macro_flat = spatial_logits[:, 0].reshape(b_size, -1)
    s_micro_flat = spatial_logits[:, 1].reshape(b_size, -1)

    is_sm = torch.isin(t_macro, spatial_macro_ids).float()
    is_smi = torch.isin(t_micro, spatial_micro_ids).float()

    ce_sm = (F.cross_entropy(s_macro_flat, t_mxy, reduction="none") * is_sm).sum() / is_sm.sum().clamp(min=1.0)
    ce_smi = (F.cross_entropy(s_micro_flat, t_mixy, reduction="none") * is_smi).sum() / is_smi.sum().clamp(min=1.0)

    total_loss = loss_macro + loss_micro + 0.5 * (ce_sm + ce_smi)

    with torch.no_grad():
        acc_macro = (masked_macro.argmax(dim=-1) == t_macro).float().mean().item()
        acc_micro = (masked_micro.argmax(dim=-1) == t_micro).float().mean().item()
    return total_loss, acc_macro, acc_micro


def train_bc(
    dataset_path: str,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 5e-4,
    device_arg: str = "auto",
    cpu_threads: int = 0,
    num_workers: int = 2,
    export_onnx: bool = True,
    cfg: Optional[V2Config] = None,
):
    cfg = cfg or V2Config()
    device = select_device(device_arg, cpu_threads)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"No existe el dataset BC en '{dataset_path}'. "
            "Genera uno con 'python -m src.pc_sc2_client --collect-teacher' o 'python -m src.replay_dataset --extract-replays ...'"
        )

    train_ds, val_ds = load_bc_splits(dataset_path, val_ratio=0.15)
    sampler = train_ds.build_sampler()

    eff_workers = 0 if (os.name == "nt" and device.type == "cpu") else num_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, max(1, len(train_ds))),
        sampler=sampler,
        drop_last=len(train_ds) > batch_size,
        num_workers=eff_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(batch_size, max(1, len(val_ds))),
        shuffle=False,
        num_workers=eff_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = SC2HailoV2Actor(
        stem_channels=cfg.stem_channels,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        num_layers=cfg.num_layers,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
    ).to(device)

    print(
        f"[BC V2] Device={device} | Params={count_parameters(model):,} | "
        f"Train={len(train_ds)} | Val={len(val_ds)} | Threads={torch.get_num_threads()}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    sp_macro_t = torch.tensor(sorted(SPATIAL_MACRO_IDS), device=device, dtype=torch.long)
    sp_micro_t = torch.tensor(sorted(SPATIAL_MICRO_IDS), device=device, dtype=torch.long)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(os.path.abspath(cfg.actor_weights)), exist_ok=True)

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        tr_loss = tr_am = tr_ami = tr_steps = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, am, ami = compute_bc_loss(model, batch, device, sp_macro_t, sp_micro_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += float(loss.item())
            tr_am += am
            tr_ami += ami
            tr_steps += 1
        scheduler.step()

        model.eval()
        vl_loss = vl_am = vl_ami = vl_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                loss, am, ami = compute_bc_loss(model, batch, device, sp_macro_t, sp_micro_t)
                vl_loss += float(loss.item())
                vl_am += am
                vl_ami += ami
                vl_steps += 1

        tr_loss /= max(1, tr_steps)
        vl_loss /= max(1, vl_steps)
        dt = time.perf_counter() - t0
        print(
            f"Época [{ep:02d}/{epochs:02d}] ({dt:.1f}s) | "
            f"Train: {tr_loss:.4f} (Mac {100*tr_am/max(1,tr_steps):.1f}%, Mic {100*tr_ami/max(1,tr_steps):.1f}%) | "
            f"Val: {vl_loss:.4f} (Mac {100*vl_am/max(1,vl_steps):.1f}%, Mic {100*vl_ami/max(1,vl_steps):.1f}%)"
        )

        if vl_loss <= best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), cfg.actor_weights)
            torch.save(model.state_dict(), cfg.bc_anchor_weights)

    print(f"[BC V2] Mejores pesos guardados en: {cfg.actor_weights} y ancla KL en: {cfg.bc_anchor_weights}")

    # Guardar muestras reales para cuantización INT8 en Hailo DFC
    n_cal = min(cfg.calib_samples, len(val_ds))
    np.savez_compressed(
        cfg.calib_path,
        map_tensor=val_ds.map_obs[:n_cal].astype(np.float32),
        global_vec=val_ds.vec_obs[:n_cal].astype(np.float32),
    )
    print(f"[BC V2] Calibración INT8 guardada ({n_cal} muestras): {cfg.calib_path}")

    if export_onnx:
        model.load_state_dict(torch.load(cfg.actor_weights, map_location="cpu", weights_only=True))
        export_v2_to_onnx(model, onnx_path=cfg.onnx_path, opset_version=cfg.onnx_opset)


if __name__ == "__main__":
    cfg = V2Config()
    parser = argparse.ArgumentParser(description="Preentrenamiento Supervisado (BC) SC2-Hailo8 V2")
    parser.add_argument("--dataset", type=str, default=cfg.bc_dataset_path)
    parser.add_argument("--epochs", type=int, default=cfg.bc_epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.bc_lr)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=0, help="Hilos CPU PyTorch (0 = automático)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    train_bc(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_arg=args.device,
        cpu_threads=args.threads,
        num_workers=args.workers,
        export_onnx=not args.no_export,
        cfg=cfg,
    )
