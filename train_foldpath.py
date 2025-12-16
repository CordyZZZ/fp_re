"""Train FoldPath on PaintNet data.

This script is intentionally minimal and self-contained.
It reuses MaskPlanner's dataset I/O and PointNet++ primitives, but does NOT
depend on MaskPlanner-specific multi-stage losses or post-processing.

Dataset expectation (same as MaskPlanner/PaintNet):
  <root>/<sample>/<sample>.obj
  <root>/<sample>/trajectory.txt
  <root>/train_split.json, <root>/test_split.json

Example:
  python train_foldpath.py \
    --dataset cuboids-v2 \
    --data_root /path/to/PaintNet/cuboids-v2 \
    --out_dir runs/foldpath_cuboids \
    --epochs 200 --batch_size 24 --lr 3e-4

If you maintain multiple category roots, repeat --data_root.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.foldpath import FoldPath, FoldPathConfig
from utils.dataset.foldpath_dataset import FoldPathDataset, FoldPathDatasetConfig, foldpath_collate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, help="PaintNet category name, e.g. cuboids-v2")
    p.add_argument("--data_root", type=str, action="append", required=True,
                   help="Root folder containing train_split.json/test_split.json and samples")
    p.add_argument("--out_dir", type=str, default="runs/foldpath")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    # FoldPath core
    p.add_argument("--num_queries", type=int, default=40)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--tf_layers", type=int, default=4)
    p.add_argument("--tf_heads", type=int, default=4)
    p.add_argument("--head_layers", type=int, default=4)
    p.add_argument("--head_hidden", type=int, default=512)
    p.add_argument("--activation", type=str, default="relu", choices=["relu", "siren", "finer"])
    p.add_argument("--T_train", type=int, default=64)
    p.add_argument("--T_test", type=int, default=384)

    # Data
    p.add_argument("--pc_points", type=int, default=5120)
    p.add_argument("--normalization", type=str, default="per-mesh", choices=["none", "per-mesh", "per-dataset"])
    p.add_argument("--data_scale_factor", type=float, default=None)
    p.add_argument("--augment_roty", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str, model: torch.nn.Module, optim: torch.optim.Optimizer, epoch: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "epoch": epoch}, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)

    model_cfg = FoldPathConfig(
        num_queries=args.num_queries,
        d_model=args.d_model,
        head_hidden=args.head_hidden,
        head_layers=args.head_layers,
        tf_layers=args.tf_layers,
        tf_heads=args.tf_heads,
        T_train=args.T_train,
        T_test=args.T_test,
        activation=args.activation,
    )
    model = FoldPath(model_cfg).to(device)

    # Dataset
    aug = ["roty"] if args.augment_roty else []
    tr_ds = FoldPathDataset(FoldPathDatasetConfig(
        dataset=args.dataset,
        roots=args.data_root,
        split="train",
        pc_points=args.pc_points,
        normalization=args.normalization,
        data_scale_factor=args.data_scale_factor,
        augmentations=aug,
        num_queries=args.num_queries,
        T=args.T_train,
        sampling="uniform",
    ))
    te_ds = FoldPathDataset(FoldPathDatasetConfig(
        dataset=args.dataset,
        roots=args.data_root,
        split="test",
        pc_points=args.pc_points,
        normalization=args.normalization,
        data_scale_factor=args.data_scale_factor,
        augmentations=[],
        num_queries=args.num_queries,
        T=args.T_test,
        sampling="equispaced",
    ))

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, collate_fn=foldpath_collate, drop_last=True)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, collate_fn=foldpath_collate, drop_last=False)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-8)

    # Save config
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        import json
        json.dump({"args": vars(args), "model_cfg": asdict(model_cfg)}, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(tr_loader, desc=f"train {epoch}/{args.epochs}")
        for batch in pbar:
            pc = batch["pc"].to(device)
            s = batch["s"].to(device)
            y = batch["y"].to(device)
            fgt = batch["f"].to(device)

            y_hat, f_hat = model(pc, s)
            loss, logs = model.loss(y_hat, f_hat, y, fgt)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            pbar.set_postfix({"loss": f"{logs['loss']:.4f}", "lp": f"{logs['loss_points']:.4f}", "lc": f"{logs['loss_conf']:.4f}"})

        sched.step()

        # Lightweight evaluation: report only loss on test split.
        model.eval()
        te_losses = []
        with torch.no_grad():
            for batch in tqdm(te_loader, desc="eval", leave=False):
                pc = batch["pc"].to(device)
                s = batch["s"].to(device)
                y = batch["y"].to(device)
                fgt = batch["f"].to(device)
                y_hat, f_hat = model(pc, s)
                loss, _ = model.loss(y_hat, f_hat, y, fgt)
                te_losses.append(float(loss.detach().cpu()))
        mean_te = sum(te_losses) / max(1, len(te_losses))
        print(f"Epoch {epoch}: test_loss={mean_te:.6f} lr={sched.get_last_lr()[0]:.2e}")

        save_checkpoint(os.path.join(args.out_dir, "checkpoints", "last.pth"), model, optim, epoch)
        if epoch % 50 == 0:
            save_checkpoint(os.path.join(args.out_dir, "checkpoints", f"epoch_{epoch:04d}.pth"), model, optim, epoch)


if __name__ == "__main__":
    main()
