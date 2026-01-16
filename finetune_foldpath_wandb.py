"""Fine-tune FoldPath on target dataset using pre-trained model from source dataset.

Usage:
  python finetune_foldpath.py \
    --source_checkpoint /path/to/source/checkpoint.pth \
    --dataset target-dataset-name \
    --data_root /path/to/target/dataset \
    --out_dir runs/finetune_target \
    --epochs 100 \
    --lr 1e-4  # 通常微调使用更小的学习率
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from models.foldpath import FoldPath, FoldPathConfig
from datasets.foldpath_dataset import FoldPathDataset, FoldPathDatasetConfig, foldpath_collate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    
    # 预训练模型参数
    p.add_argument("--source_checkpoint", type=str, required=True,
                   help="Path to source pre-trained checkpoint")
    
    # 目标数据集参数
    p.add_argument("--dataset", type=str, required=True, 
                   help="Target PaintNet category name")
    p.add_argument("--data_root", type=str, action="append", required=True,
                   help="Root folder of target dataset")
    
    # 输出和训练参数
    p.add_argument("--out_dir", type=str, default="runs/finetune")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)  # 微调通常用更小的学习率
    p.add_argument("--num_queries", type=int, default=40)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--tf_layers", type=int, default=4)
    p.add_argument("--tf_heads", type=int, default=4)
    p.add_argument("--head_layers", type=int, default=4)
    p.add_argument("--head_hidden", type=int, default=512)
    p.add_argument("--activation", type=str, default="finer", choices=["relu", "siren", "finer"])
    p.add_argument("--T_train", type=int, default=64)
    p.add_argument("--T_test", type=int, default=384)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    
    # 微调策略参数
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze PointNet++ encoder during fine-tuning")
    p.add_argument("--freeze_decoder", action="store_true",
                   help="Freeze transformer decoder during fine-tuning")
    p.add_argument("--freeze_head", action="store_true",
                   help="Freeze MLP head during fine-tuning")
    p.add_argument("--finetune_all", action="store_true",
                   help="Fine-tune all parameters (default if no freeze flags are set)")
    
    # 数据增强参数
    p.add_argument("--pc_points", type=int, default=5120)
    p.add_argument("--normalization", type=str, default="per-mesh", 
                   choices=["none", "per-mesh", "per-dataset"])
    p.add_argument("--data_scale_factor", type=float, default=None)
    p.add_argument("--augment_roty", action="store_true")

    # wandb
    p.add_argument("--wandb_project", type=str, default="foldpath-finetune",
                   help="Wandb project name")
    p.add_argument("--wandb_name", type=str, default=None,
                   help="Wandb run name (default: auto-generated)")
    p.add_argument("--wandb_entity", type=str, default=None,
                   help="Wandb entity/team name")
    p.add_argument("--wandb_tags", type=str, nargs="+", default=[],
                   help="Tags for wandb run")
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable wandb logging")
    
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pretrained_model(checkpoint_path: str, model: FoldPath, device: torch.device) -> FoldPath:
    """加载预训练模型权重"""
    print(f"Loading pre-trained model from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 检查checkpoint格式
    if "model" in checkpoint:
        model_state_dict = checkpoint["model"]
    else:
        # 如果checkpoint直接保存了模型state_dict
        model_state_dict = checkpoint
    
    # 加载权重
    model.load_state_dict(model_state_dict, strict=False)
    
    print("Pre-trained model loaded successfully")
    return model


def setup_finetune_strategy(model: FoldPath, args: argparse.Namespace) -> None:
    """根据参数设置微调策略"""
    print("\nFine-tuning strategy:")
    
    if args.freeze_backbone or args.freeze_decoder or args.freeze_head:
        # 部分冻结策略
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = 0
        
        if args.freeze_backbone:
            for name, param in model.encoder.named_parameters():
                param.requires_grad = False
            print(f"  ✓ Frozen: PointNet++ encoder")
        
        if args.freeze_decoder:
            for name, param in model.decoder.named_parameters():
                param.requires_grad = False
            for name, param in model.query_embed.named_parameters():
                param.requires_grad = False
            print(f"  ✓ Frozen: Transformer decoder + query embeddings")
        
        if args.freeze_head:
            for name, param in model.head.named_parameters():
                param.requires_grad = False
            print(f"  ✓ Frozen: Modulated MLP head")
        
        # 计算可训练参数数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {frozen_params:,}")
        print(f"  Trainable ratio: {trainable_params/total_params*100:.1f}%")
    
    else:
        # 全部微调
        print(f"  Fine-tuning all parameters")
        for param in model.parameters():
            param.requires_grad = True


def save_checkpoint(path: str, model: torch.nn.Module, 
                    optim: torch.optim.Optimizer, 
                    epoch: int, 
                    metrics: dict = None) -> None:
    """保存checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {}
    }
    
    torch.save(checkpoint, path)


def save_finetune_config(args: argparse.Namespace, out_dir: str, model_cfg: FoldPathConfig) -> None:
    """保存微调配置"""
    config = {
        "args": vars(args),
        "model_cfg": asdict(model_cfg),
        "finetune_timestamp": str(torch.tensor(0).device),  # placeholder
    }
    
    # 获取预训练模型的配置信息
    source_checkpoint = torch.load(args.source_checkpoint, map_location="cpu")
    if "model_cfg" in source_checkpoint:
        config["source_model_cfg"] = source_checkpoint["model_cfg"]
    
    config_path = os.path.join(out_dir, "finetune_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    print(f"Fine-tune configuration saved to {config_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
    
    device = torch.device(args.device)
    
    # 初始化wandb
    if not args.no_wandb:
        run_name = args.wandb_name or f"finetune_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            entity=args.wandb_entity,
            tags=args.wandb_tags,
            config=vars(args),
            dir=args.out_dir,
        )
        
        # 添加额外的配置信息（使用不同的key名）
        wandb.config.model_type = "FoldPath"
        wandb.config.finetune_mode = "partial" if (args.freeze_backbone or args.freeze_decoder or args.freeze_head) else "full"
        wandb.config.target_dataset = args.dataset
        
        # 记录basename作为单独字段
        wandb.config.source_checkpoint_basename = os.path.basename(args.source_checkpoint)
    
    # 1. 从源checkpoint加载模型配置
    # 首先尝试从checkpoint中读取配置
    checkpoint = torch.load(args.source_checkpoint, map_location="cpu")
    
    if "model_cfg" in checkpoint:
        # checkpoint中包含配置
        model_cfg_dict = checkpoint["model_cfg"]
        model_cfg = FoldPathConfig(**model_cfg_dict)
        print(f"Loaded model config from checkpoint: {model_cfg_dict}")
    else:
        # 使用默认配置
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
        print("Using default model config")
    
    # 2. 创建模型
    model = FoldPath(model_cfg).to(device)
    
    # 3. 加载预训练权重
    model = load_pretrained_model(args.source_checkpoint, model, device)
    
    # 4. 设置微调策略
    setup_finetune_strategy(model, args)
    
    # 5. 准备数据集
    aug = ["roty"] if args.augment_roty else []
    
    train_config = FoldPathDatasetConfig(
        dataset=args.dataset,
        roots=args.data_root,
        split="train",
        pc_points=args.pc_points,
        normalization=args.normalization,
        data_scale_factor=args.data_scale_factor,
        augmentations=aug,
        num_queries=model_cfg.num_queries,
        T=model_cfg.T_train,
        sampling="uniform",
    )
    
    test_config = FoldPathDatasetConfig(
        dataset=args.dataset,
        roots=args.data_root,
        split="test",
        pc_points=args.pc_points,
        normalization=args.normalization,
        data_scale_factor=args.data_scale_factor,
        augmentations=[],
        num_queries=model_cfg.num_queries,
        T=model_cfg.T_test,
        sampling="equispaced",
    )
    
    print(f"\nDataset info:")
    print(f"  Target dataset: {args.dataset}")
    print(f"  Data roots: {args.data_root}")
    
    tr_ds = FoldPathDataset(train_config)
    te_ds = FoldPathDataset(test_config)
    
    print(f"  Train samples: {len(tr_ds)}")
    print(f"  Test samples: {len(te_ds)}")
    
    # 记录数据集信息到wandb（新增）
    if not args.no_wandb:
        wandb.config.update({
            "train_samples": len(tr_ds),
            "test_samples": len(te_ds),
            "batch_size": args.batch_size,
        })
    
    tr_loader = DataLoader(
        tr_ds, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        collate_fn=foldpath_collate, 
        drop_last=True
    )
    
    te_loader = DataLoader(
        te_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        collate_fn=foldpath_collate, 
        drop_last=False
    )
    
    # 6. 设置优化器（只优化需要梯度的参数）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters! Check your freeze settings.")
    
    optim = torch.optim.Adam(
        trainable_params, 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # 使用余弦退火学习率调度器
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, 
        T_max=args.epochs, 
        eta_min=args.lr * 0.01  # 最低学习率为初始学习率的1%
    )
    
    # 7. 保存配置
    save_finetune_config(args, args.out_dir, model_cfg)
    
    # 8. 训练循环
    print("\nStarting fine-tuning...")
    best_test_loss = float('inf')
    
    # for epoch in range(1, args.epochs + 1):
    #     # 训练阶段
    #     model.train()
    #     train_losses = []
    #     train_point_losses = []
    #     train_conf_losses = []
        
    #     # 新增：记录每个batch的损失用于wandb
    #     batch_step = 0
        
    #     pbar = tqdm(tr_loader, desc=f"Fine-tune {epoch}/{args.epochs}")
    #     for batch in pbar:
    #         pc = batch["pc"].to(device)
    #         s = batch["s"].to(device)
    #         y = batch["y"].to(device)
    #         fgt = batch["f"].to(device)
            
    #         y_hat, f_hat = model(pc, s)
    #         loss, logs = model.loss(y_hat, f_hat, y, fgt)
            
    #         optim.zero_grad(set_to_none=True)
    #         loss.backward()
            
    #         # 梯度裁剪
    #         torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
    #         optim.step()
            
    #         # 记录损失
    #         train_losses.append(logs["loss"])
    #         train_point_losses.append(logs["loss_points"])
    #         train_conf_losses.append(logs["loss_conf"])
            
    #         # 记录batch级别的损失到wandb（新增）
    #         if not args.no_wandb:
    #             batch_logs = {
    #                 "train/batch_loss": logs["loss"],
    #                 "train/batch_loss_points": logs["loss_points"],
    #                 "train/batch_loss_conf": logs["loss_conf"],
    #                 "train/batch_lr": sched.get_last_lr()[0],
    #             }
                
    #             # 添加详细的loss组件（如果有的话）
    #             if "loss_pos" in logs:
    #                 batch_logs.update({
    #                     "train/batch_loss_pos": logs["loss_pos"],
    #                     "train/batch_loss_ang": logs["loss_ang"],
    #                 })
    #             if "match_ratio" in logs:
    #                 batch_logs.update({
    #                     "train/match_ratio": logs["match_ratio"],
    #                     "train/num_real_paths": logs["num_real_paths"],
    #                     "train/num_matched": logs["num_matched"],
    #                 })
                
    #             wandb.log(batch_logs, step=epoch * len(tr_loader) + batch_step)
    #             batch_step += 1
            
    #         pbar.set_postfix({
    #             "loss": f"{logs['loss']:.4f}", 
    #             "lp": f"{logs['loss_points']:.4f}", 
    #             "lc": f"{logs['loss_conf']:.4f}"
    #         })
        
    #     sched.step()
        
    #     # 计算平均训练损失
    #     avg_train_loss = sum(train_losses) / len(train_losses)
    #     avg_train_point_loss = sum(train_point_losses) / len(train_point_losses)
    #     avg_train_conf_loss = sum(train_conf_losses) / len(train_conf_losses)
        
    
    # 在训练循环中添加梯度监控
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_pos_losses = []
        train_ang_losses = []
        train_conf_losses = []
        
        batch_step = 0
        
        pbar = tqdm(tr_loader, desc=f"Fine-tune {epoch}/{args.epochs}")
        for batch in pbar:
            pc = batch["pc"].to(device)
            s = batch["s"].to(device)
            y = batch["y"].to(device)
            fgt = batch["f"].to(device)
            
            y_hat, f_hat = model(pc, s)
            loss, logs = model.loss(y_hat, f_hat, y, fgt)
            
            # 检查梯度是否存在
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is None:
                    print(f"Warning: {name} has no gradient!")
            
            optim.zero_grad(set_to_none=True)
            loss.backward()
            
            # 梯度裁剪 - 使用更小的值
            clip_value = 0.1  # 降低梯度裁剪阈值
            torch.nn.utils.clip_grad_norm_(trainable_params, clip_value)
            
            # 检查梯度是否过大或过小
            total_grad_norm = 0
            for param in trainable_params:
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            # 如果梯度太小，可能是梯度消失
            if total_grad_norm < 1e-8:
                print(f"Warning: Gradient norm too small: {total_grad_norm:.6e}")
            
            optim.step()
            
            # 记录损失
            train_losses.append(logs["loss"])
            train_pos_losses.append(logs["loss_pos"])
            train_ang_losses.append(logs["loss_ang"])
            train_conf_losses.append(logs["loss_conf"])
            
            # 记录到wandb
            if not args.no_wandb:
                batch_logs = {
                    "train/batch_loss": logs["loss"],
                    "train/batch_loss_pos": logs["loss_pos"],
                    "train/batch_loss_ang": logs["loss_ang"],
                    "train/batch_loss_conf": logs["loss_conf"],
                    "train/match_ratio": logs["match_ratio"],
                    "train/pos_ratio": logs["pos_ratio"],
                    "train/conf_weight": logs["conf_weight"],
                    "train/gradient_norm": total_grad_norm,
                    "train/lr": sched.get_last_lr()[0],
                    "train/step": epoch * len(tr_loader) + batch_step,
                }
                
                # 记录预测的置信度分布
                if "avg_conf_pos" in logs:
                    batch_logs["train/avg_conf_pos"] = logs["avg_conf_pos"]
                    batch_logs["train/avg_conf_neg"] = logs["avg_conf_neg"]
                
                wandb.log(batch_logs)
                batch_step += 1
            
            pbar.set_postfix({
                "loss": f"{logs['loss']:.4f}", 
                "pos": f"{logs['loss_pos']:.4f}",
                "ang": f"{logs['loss_ang']:.4f}", 
                "conf": f"{logs['loss_conf']:.4f}",
                "grad": f"{total_grad_norm:.2e}"
            })
        
        sched.step()
        
        # 计算平均训练损失
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_pos_loss = sum(train_pos_losses) / len(train_pos_losses)
        avg_train_ang_loss = sum(train_ang_losses) / len(train_ang_losses)
        avg_train_conf_loss = sum(train_conf_losses) / len(train_conf_losses)
        
        # 测试阶段
        model.eval()
        test_losses = []
        test_point_losses = []
        test_conf_losses = []
        
        with torch.no_grad():
            for batch in tqdm(te_loader, desc="Evaluating", leave=False):
                pc = batch["pc"].to(device)
                s = batch["s"].to(device)
                y = batch["y"].to(device)
                fgt = batch["f"].to(device)
                
                y_hat, f_hat = model(pc, s)
                loss, logs = model.loss(y_hat, f_hat, y, fgt)
                test_losses.append(float(loss.detach().cpu()))
                test_point_losses.append(logs["loss_points"])
                test_conf_losses.append(logs["loss_conf"])
        
        avg_test_loss = sum(test_losses) / max(1, len(test_losses))
        avg_test_point_loss = sum(test_point_losses) / max(1, len(test_point_losses))
        avg_test_conf_loss = sum(test_conf_losses) / max(1, len(test_conf_losses))
        
        # 打印epoch结果
        print(f"\nEpoch {epoch:3d}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f} "
              f"(Point: {avg_train_pos_loss:.6f}, Angle:{avg_train_ang_loss:.6f}, Conf: {avg_train_conf_loss:.6f})")
        print(f"  Test Loss:  {avg_test_loss:.6f}")
        print(f"  Learning Rate: {sched.get_last_lr()[0]:.2e}")
        
        # 记录epoch级别的指标到wandb（新增）
        if not args.no_wandb:
            epoch_logs = {
                "epoch": epoch,
                "train/loss": avg_train_loss,
                "train/loss_points": avg_train_pos_loss,
                "train/loss_conf": avg_train_conf_loss,
                "test/loss": avg_test_loss,
                "test/loss_points": avg_test_point_loss,
                "test/loss_conf": avg_test_conf_loss,
                "lr": sched.get_last_lr()[0],
            }
            
            # 添加额外的调试信息（如果有的话）
            if "loss_pos" in logs:
                epoch_logs.update({
                    "train/loss_pos": sum([l.get("loss_pos", 0) for l in train_losses]) / len(train_losses) if hasattr(train_losses[0], 'get') else 0,
                    "train/loss_ang": sum([l.get("loss_ang", 0) for l in train_losses]) / len(train_losses) if hasattr(train_losses[0], 'get') else 0,
                })
            
            wandb.log(epoch_logs, step=epoch)
        
        # 保存checkpoint
        metrics = {
            "train_loss": avg_train_loss,
            "train_pos_loss": avg_train_pos_loss,
            "train_conf_loss": avg_train_conf_loss,
            "test_loss": avg_test_loss,
            "test_point_loss": avg_test_point_loss,
            "test_conf_loss": avg_test_conf_loss,
            "lr": sched.get_last_lr()[0]
        }
        
        # 保存最新的checkpoint
        save_checkpoint(
            os.path.join(args.out_dir, "checkpoints", "latest.pth"),
            model, optim, epoch, metrics
        )
        
        # 如果测试损失更好，保存最佳模型
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            save_checkpoint(
                os.path.join(args.out_dir, "checkpoints", "best.pth"),
                model, optim, epoch, metrics
            )
            print(f"  ✓ New best model saved (test loss: {best_test_loss:.6f})")
            
            # 记录最佳模型信息到wandb（新增）
            if not args.no_wandb:
                wandb.run.summary["best_test_loss"] = best_test_loss
                wandb.run.summary["best_epoch"] = epoch
        
        # 每10个epoch保存一次
        if epoch % 10 == 0:
            save_checkpoint(
                os.path.join(args.out_dir, "checkpoints", f"epoch_{epoch:04d}.pth"),
                model, optim, epoch, metrics
            )

    print(f"\nFine-tuning completed!")
    print(f"Best test loss: {best_test_loss:.6f}")
    print(f"Checkpoints saved in: {args.out_dir}/checkpoints/")
    
    # 结束wandb运行（新增）
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()