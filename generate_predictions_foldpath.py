#!/usr/bin/env python3
"""
Generate NPY predictions using trained FoldPath model.
"""

import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.foldpath import FoldPath, FoldPathConfig
from utils.dataset.foldpath_dataset import FoldPathDataset, FoldPathDatasetConfig, foldpath_collate


def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions using trained FoldPath model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file (e.g., last.pth)')
    parser.add_argument('--config', type=str, required=True, help='Path to config.json file')
    parser.add_argument('--dataset', type=str, default='windows-v2', help='Dataset name') 
    parser.add_argument('--data_root', type=str, default='/fileStore/windows-v2', help='Path to dataset root')
    parser.add_argument('--output_dir', type=str, default='./generated_predictions', help='Output directory for predictions')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='Data split to generate predictions for')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for inference')
    parser.add_argument('--save_all_batches', action='store_true', help='Save predictions for all batches')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract model configuration
    if 'model_cfg' in config:
        model_cfg_dict = config['model_cfg']
    else:
        # Fallback to default values
        model_cfg_dict = {
            'num_queries': 40,
            'd_model': 384,
            'head_hidden': 512,
            'head_layers': 4,
            'tf_layers': 4,
            'tf_heads': 4,
            'T_train': 64,
            'T_test': 384,
            'activation': 'siren'
        }
    
    # Extract training arguments
    if 'args' in config:
        train_args = config['args']
    else:
        train_args = {}
    
    return model_cfg_dict, train_args


def create_model_from_config(model_cfg_dict, device):
    """Create FoldPath model from configuration dictionary"""
    model_cfg = FoldPathConfig(
        num_queries=model_cfg_dict.get('num_queries', 40),
        d_model=model_cfg_dict.get('d_model', 384),
        head_hidden=model_cfg_dict.get('head_hidden', 512),
        head_layers=model_cfg_dict.get('head_layers', 4),
        tf_layers=model_cfg_dict.get('tf_layers', 4),
        tf_heads=model_cfg_dict.get('tf_heads', 4),
        T_train=model_cfg_dict.get('T_train', 64),
        T_test=model_cfg_dict.get('T_test', 384),
        activation=model_cfg_dict.get('activation', 'siren')
    )
    
    model = FoldPath(model_cfg).to(device)
    return model, model_cfg


def load_checkpoint(model, checkpoint_path, device):
    # Load model weights from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights directly")
    
    return model


def generate_predictions(model, data_loader, device, output_dir):
    # Generate predictions and save them as NPY files
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Generating predictions")):
            # Move data to device
            pc = batch["pc"].to(device)
            s = batch["s"].to(device)
            y = batch["y"].to(device)
            fgt = batch["f"].to(device)
            
            # Generate predictions
            y_hat, f_hat = model(pc, s)
            
            # Convert to numpy
            y_hat_np = y_hat.cpu().numpy()  # Shape: (batch_size, num_queries, seq_len, 6)
            f_hat_np = f_hat.cpu().numpy()  # Shape: (batch_size, num_queries)
            y_np = y.cpu().numpy()  # Ground truth
            f_np = fgt.cpu().numpy()  # Ground truth confidence
            
            # Save this batch's predictions
            batch_data = {
                'traj_pred': y_hat_np,
                'conf_pred': f_hat_np,
                'traj_gt': y_np,
                'conf_gt': f_np,
                'batch': batch_idx,
                'sample_names': batch.get('sample_names', [f'sample_{i}' for i in range(len(pc))])
            }
            
            # Collect for combined file
            all_predictions.append(y_hat_np)
            all_ground_truth.append(y_np)
    
    # Save combined predictions
    if all_predictions:
        combined_predictions = np.concatenate(all_predictions, axis=0)
        combined_ground_truth = np.concatenate(all_ground_truth, axis=0)
        
        combined_data = {
            'traj_pred': combined_predictions,
            'traj_gt': combined_ground_truth,
            'num_samples': len(combined_predictions)
        }
        
        combined_file = os.path.join(output_dir, 'all_predictions.npy')
        np.save(combined_file, combined_data)
        print(f"saved combined predictions to {combined_file}")
        # print(f"  - Shape: {combined_predictions.shape}")
    
    return combined_predictions if all_predictions else None


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    model_cfg_dict, train_args = load_config(args.config)
    
    # Create model
    print("Creating model...")
    model, model_cfg = create_model_from_config(model_cfg_dict, device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    model = load_checkpoint(model, args.checkpoint, device)
    
    # Create dataset
    print(f"Creating {args.split} dataset...")
    dataset_config = FoldPathDatasetConfig(
        dataset=args.dataset,
        roots=[args.data_root],
        split=args.split,
        pc_points=5120,  # Default from training
        normalization='per-mesh',  # Default from training
        data_scale_factor=None,
        augmentations=[],
        num_queries=model_cfg.num_queries,
        T=model_cfg.T_test,  # Use test-time sampling
        sampling="equispaced",
    )
    
    dataset = FoldPathDataset(dataset_config)
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        collate_fn=foldpath_collate, 
        drop_last=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(data_loader)}")
    
    # Generate predictions
    print(f"\nGenerating predictions for {args.split} split...")
    predictions = generate_predictions(model, data_loader, device, args.output_dir)


if __name__ == '__main__':
    main()