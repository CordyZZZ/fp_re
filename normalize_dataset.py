#!/usr/bin/env python3
"""
Normalize PaintNet dataset for FoldPath visualization.

This script creates normalized copies of OBJ files and trajectory data.
It follows the same normalization logic as FoldPathDataset.

Usage:
    python normalize_dataset.py --data_root /path/to/windows-v2 --output_root /path/to/normalized_windows-v2
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import trimesh


def normalize_mesh_per_mesh(vertices):
    """Normalize mesh vertices per-mesh (center and unit sphere)"""
    # Center
    center = vertices.mean(axis=0)
    vertices_centered = vertices - center
    
    # Scale to unit sphere
    max_dist = np.linalg.norm(vertices_centered, axis=1).max()
    if max_dist > 0:
        scale = 1.0 / max_dist
    else:
        scale = 1.0
    
    vertices_normalized = vertices_centered * scale
    return vertices_normalized, center, scale


def normalize_trajectory(trajectory_points, center, scale):
    """Normalize trajectory points using the same center and scale"""
    # trajectory_points shape: (N, 6) where first 3 are position, last 3 are orientation
    positions = trajectory_points[:, :3]
    orientations = trajectory_points[:, 3:]
    
    # Normalize positions
    positions_normalized = (positions - center) * scale
    
    # Orientations remain unchanged (they're unit vectors)
    # But we need to renormalize just in case
    norms = np.linalg.norm(orientations, axis=1, keepdims=True)
    orientations_normalized = orientations / np.where(norms > 0, norms, 1.0)
    
    return np.concatenate([positions_normalized, orientations_normalized], axis=1)


def load_trajectory(traj_file):
    """Load trajectory from .txt file"""
    with open(traj_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    trajectory = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 6:
            # Position (x, y, z) and orientation (dx, dy, dz)
            point = [float(p) for p in parts[:6]]
            trajectory.append(point)
    
    return np.array(trajectory, dtype=np.float32)


def process_sample(sample_dir, output_dir, normalization='per-mesh'):
    """Process a single sample directory"""
    sample_id = os.path.basename(sample_dir)
    
    # Create output directory
    output_sample_dir = os.path.join(output_dir, sample_id)
    os.makedirs(output_sample_dir, exist_ok=True)
    
    # Paths
    obj_file = os.path.join(sample_dir, f"{sample_id}.obj")
    traj_file = os.path.join(sample_dir, "trajectory.txt")
    norm_obj_file = os.path.join(output_sample_dir, f"{sample_id}_norm.obj")
    norm_traj_file = os.path.join(output_sample_dir, "trajectory_norm.txt")
    params_file = os.path.join(output_sample_dir, "norm_params.json")
    
    # Skip if already processed
    if os.path.exists(norm_obj_file) and os.path.exists(norm_traj_file):
        return None
    
    try:
        # Load and normalize mesh
        mesh = trimesh.load(obj_file)
        vertices = mesh.vertices.astype(np.float32)
        
        if normalization == 'per-mesh':
            vertices_norm, center, scale = normalize_mesh_per_mesh(vertices)
        elif normalization == 'none':
            vertices_norm = vertices
            center = np.zeros(3, dtype=np.float32)
            scale = 1.0
        else:
            raise ValueError(f"Unsupported normalization: {normalization}")
        
        # Save normalized mesh
        mesh_norm = trimesh.Trimesh(vertices=vertices_norm, faces=mesh.faces)
        mesh_norm.export(norm_obj_file)
        
        # Load and normalize trajectory
        if os.path.exists(traj_file):
            trajectory = load_trajectory(traj_file)
            if trajectory.shape[0] > 0:
                trajectory_norm = normalize_trajectory(trajectory, center, scale)
                
                # Save normalized trajectory
                np.savetxt(norm_traj_file, trajectory_norm, fmt='%.6f')
            else:
                # Empty trajectory file
                open(norm_traj_file, 'w').close()
        else:
            # No trajectory file
            open(norm_traj_file, 'w').close()
            trajectory_norm = np.zeros((0, 6), dtype=np.float32)
        
        # Save normalization parameters
        params = {
            'center': center.tolist(),
            'scale': float(scale),
            'normalization': normalization,
            'original_obj': os.path.relpath(obj_file, output_dir),
            'original_traj': os.path.relpath(traj_file, output_dir) if os.path.exists(traj_file) else None
        }
        
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        return params
    
    except Exception as e:
        print(f"Error processing {sample_dir}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Normalize PaintNet dataset for visualization")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory of the dataset")
    parser.add_argument("--output_root", type=str, required=True,
                       help="Output directory for normalized data")
    parser.add_argument("--normalization", type=str, default="per-mesh",
                       choices=["none", "per-mesh", "per-dataset"],
                       help="Normalization method")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip samples that already have normalized files")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_root, exist_ok=True)
    
    # Copy split files
    for split_file in ["train_split.json", "test_split.json"]:
        src = os.path.join(args.data_root, split_file)
        dst = os.path.join(args.output_root, split_file)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, dst)
            print(f"Copied {split_file}")
    
    # Find all sample directories
    sample_dirs = []
    for item in os.listdir(args.data_root):
        item_path = os.path.join(args.data_root, item)
        if os.path.isdir(item_path):
            obj_file = os.path.join(item_path, f"{item}.obj")
            if os.path.exists(obj_file):
                sample_dirs.append(item_path)
    
    print(f"Found {len(sample_dirs)} samples to process")
    
    # Process each sample
    all_params = []
    for sample_dir in tqdm(sample_dirs, desc="Normalizing samples"):
        params = process_sample(sample_dir, args.output_root, args.normalization)
        if params:
            all_params.append(params)
    
    # Save global normalization info
    global_info = {
        'dataset_root': args.data_root,
        'normalized_root': args.output_root,
        'normalization': args.normalization,
        'num_samples': len(all_params),
        'samples': [os.path.basename(os.path.dirname(p['original_obj'])) for p in all_params]
    }
    
    info_file = os.path.join(args.output_root, "normalization_info.json")
    with open(info_file, 'w') as f:
        json.dump(global_info, f, indent=2)
    
    print(f"\nNormalization complete!")
    print(f"Original data: {args.data_root}")
    print(f"Normalized data: {args.output_root}")
    print(f"Total samples processed: {len(all_params)}")


if __name__ == "__main__":
    main()