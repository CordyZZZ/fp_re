#!/usr/bin/env python3
"""
FoldPath Inference Script
"""

import argparse
import os
import json
import numpy as np
import torch
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.foldpath import FoldPath, FoldPathConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_point_cloud(obj_path: str, target_points: int = 5120) -> np.ndarray:
    """
    Load and normalize point cloud from OBJ file.
    
    Args:
        obj_path: Path to OBJ file
        target_points: Target number of points after resampling
        
    Returns:
        Normalized point cloud (N, 3) or None if failed
    """
    try:
        vertices = []
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        # Filter invalid values
                        if not (np.isnan(x) or np.isinf(x) or 
                                np.isnan(y) or np.isinf(y) or 
                                np.isnan(z) or np.isinf(z)):
                            vertices.append([x, y, z])
        
        if len(vertices) == 0:
            logger.error(f"No valid vertices in {obj_path}")
            return None
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # Normalize: center and scale
        center = vertices.mean(axis=0)
        vertices = vertices - center
        
        max_val = np.abs(vertices).max()
        if max_val > 0:
            scale = 1.0 / max_val
            vertices = vertices * scale
        
        # Resample to target points
        if len(vertices) > target_points:
            indices = np.random.choice(len(vertices), target_points, replace=False)
            vertices = vertices[indices]
        elif len(vertices) < target_points:
            indices = np.random.choice(len(vertices), target_points - len(vertices), replace=True)
            additional = vertices[indices]
            noise = np.random.normal(0, 0.001, additional.shape).astype(np.float32)
            vertices = np.vstack([vertices, additional + noise])
        
        return vertices
        
    except Exception as e:
        logger.error(f"Failed to load point cloud: {e}")
        return None


def load_model(checkpoint_path: str, device: torch.device) -> FoldPath:
    """
    Load FoldPath model with configuration.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Torch device (cpu or cuda)
        
    Returns:
        Loaded FoldPath model
    """
    try:
        # Load configuration
        config_path = os.path.join(os.path.dirname(checkpoint_path), '..', 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            model_cfg = config_data.get('model_cfg', {})
            args = config_data.get('args', {})
            
            cfg = FoldPathConfig(
                num_queries=model_cfg.get('num_queries', 40),
                d_model=model_cfg.get('d_model', 384),
                head_hidden=model_cfg.get('head_hidden', 512),
                head_layers=model_cfg.get('head_layers', 4),
                tf_layers=model_cfg.get('tf_layers', 4),
                tf_heads=model_cfg.get('tf_heads', 4),
                T_train=model_cfg.get('T_train', 64),
                T_test=model_cfg.get('T_test', 384),
                activation=args.get('activation', 'finer')
            )
        else:
            # Default configuration
            cfg = FoldPathConfig(
                num_queries=40,
                d_model=384,
                head_hidden=512,
                head_layers=4,
                tf_layers=4,
                tf_heads=4,
                T_train=64,
                T_test=384,
                activation='finer'
            )
        
        # Create and load model
        model = FoldPath(cfg)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        # Fix NaN/Inf in weights
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                logger.warning(f"Fixing invalid values in parameter: {name}")
                nan_mask = torch.isnan(param)
                inf_mask = torch.isinf(param)
                if nan_mask.any():
                    param.data[nan_mask] = torch.randn_like(param.data[nan_mask]) * 0.01
                if inf_mask.any():
                    param.data[inf_mask] = torch.randn_like(param.data[inf_mask]) * 0.01
        
        model = model.to(device)
        model.eval()
        
        # Disable gradients for inference
        for param in model.parameters():
            param.requires_grad = False
        
        logger.info(f"Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def run_inference(model: FoldPath, sample_dir: str, data_root: str, 
                  device: torch.device, num_points: int = 384) -> dict:
    """
    Run inference on a sample.
    
    Args:
        model: FoldPath model
        sample_dir: Sample directory name
        data_root: Root directory containing sample folders
        device: Torch device
        num_points: Number of points in output trajectories
        
    Returns:
        Dictionary containing predictions and metadata
    """
    try:
        # Find OBJ file
        obj_path = os.path.join(data_root, sample_dir, f"{sample_dir}.obj")
        if not os.path.exists(obj_path):
            obj_path = os.path.join(data_root, sample_dir, f"{sample_dir}_norm.obj")
        
        if not os.path.exists(obj_path):
            logger.error(f"OBJ file not found: {obj_path}")
            return None
        
        # Load point cloud
        point_cloud = load_point_cloud(obj_path)
        if point_cloud is None:
            return None
        
        # Prepare tensors
        pc_tensor = torch.from_numpy(point_cloud.astype(np.float32)).unsqueeze(0).to(device)
        s_tensor = torch.linspace(-1, 1, num_points, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            y_pred, f_pred = model(pc_tensor, s_tensor)
        
        # Convert to numpy
        y_np = y_pred[0].cpu().numpy()
        f_np = f_pred[0].cpu().numpy()
        
        # Fix NaN values
        f_np = np.nan_to_num(f_np, nan=0.0, posinf=1.0, neginf=0.0)
        y_np = np.nan_to_num(y_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Process paths
        all_paths = []
        path_info = []
        
        for i in range(len(f_np)):
            confidence = f_np[i]
            
            if confidence > 0.3:  # Confidence threshold
                path_positions = y_np[i, :, :3]
                
                # Remove invalid points
                mask = ~np.isnan(path_positions).any(axis=1) & ~np.isinf(path_positions).any(axis=1)
                valid_positions = path_positions[mask]
                
                if len(valid_positions) > 10:  # Minimum valid points
                    all_paths.append(valid_positions)
                    
                    # Calculate path length
                    if len(valid_positions) > 1:
                        path_length = np.sum(np.linalg.norm(np.diff(valid_positions, axis=0), axis=1))
                    else:
                        path_length = 0.0
                    
                    path_info.append({
                        'path_id': i,
                        'confidence': float(confidence),
                        'num_points': len(valid_positions),
                        'length': float(path_length)
                    })
        
        result = {
            'sample': sample_dir,
            'paths': all_paths,
            'path_info': path_info,
            'num_points': num_points,
            'point_cloud': point_cloud
        }
        
        logger.info(f"Generated {len(all_paths)} valid paths for {sample_dir}")
        return result
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='FoldPath Inference')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of dataset')
    parser.add_argument('--sample_dir', type=str, required=True,
                       help='Sample directory name')
    parser.add_argument('--output', type=str, default='predictions.npy',
                       help='Output file path')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for inference')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load model
    try:
        model = load_model(args.checkpoint, device)
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return
    
    # Run inference
    predictions = run_inference(model, args.sample_dir, args.data_root, device)
    
    if predictions:
        # Save results
        save_data = {
            'predictions': predictions,
            'metadata': {
                'sample': predictions['sample'],
                'num_paths': len(predictions['paths']),
                'device': str(device)
            }
        }
        
        np.save(args.output, save_data, allow_pickle=True)
        logger.info(f"Predictions saved to: {args.output}")
        
        # Save JSON summary
        json_path = args.output.replace('.npy', '.json')
        with open(json_path, 'w') as f:
            json.dump({
                'sample': predictions['sample'],
                'num_paths': len(predictions['paths']),
                'path_info': predictions['path_info']
            }, f, indent=2)
        
        logger.info(f"JSON summary saved to: {json_path}")
        
        # Print summary
        if predictions['paths']:
            logger.info(f"Generated {len(predictions['paths'])} paths:")
            for info in predictions['path_info'][:5]:  # Show first 5
                logger.info(f"  Path {info['path_id']}: conf={info['confidence']:.3f}, "
                           f"points={info['num_points']}, length={info['length']:.2f}")
            if len(predictions['paths']) > 5:
                logger.info(f"  ... and {len(predictions['paths']) - 5} more paths")
        else:
            logger.warning("No valid paths generated")
    else:
        logger.error("Inference failed")


if __name__ == "__main__":
    main()