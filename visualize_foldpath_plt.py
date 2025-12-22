#!/usr/bin/env python3
"""
Visualize FoldPath predictions with 3D point clouds using matplotlib.
"""

import argparse
import json
import logging
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_obj_vertices(obj_path):
    """
    Load vertices from an OBJ file.
    """
    vertices = []
    
    try:
        logging.info(f"Loading OBJ vertices from: {obj_path}")
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        try:
                            vertices.append([float(parts[1]), 
                                           float(parts[2]), 
                                           float(parts[3])])
                        except ValueError:
                            continue
        
        logging.info(f"Loaded {len(vertices)} vertices")
        return np.array(vertices) if vertices else None
        
    except Exception as e:
        logging.error(f"Failed to load OBJ: {e}")
        return None


def denormalize_points(points, norm_params):
    """
    Denormalize point coordinates using normalization parameters.
    """
    if norm_params is None or points is None:
        return points
    
    try:
        center = np.array(norm_params.get('center', [0, 0, 0]))
        scale = norm_params.get('scale', 1.0)
        
        denorm_points = points.copy()
        if denorm_points.shape[1] >= 3:
            denorm_points[:, :3] = denorm_points[:, :3] * scale + center
        
        return denorm_points
    except Exception as e:
        logging.error(f"Failed to denormalize: {e}")
        return points


def get_colormap_colors(colormap_name='tab10', num_colors=10):
    """
    Get colors from matplotlib colormap.
    """
    # Use the new matplotlib API
    if colormap_name in plt.colormaps:
        cmap = plt.colormaps[colormap_name]
    else:
        cmap = plt.colormaps.get_cmap(colormap_name)
    
    colors = []
    for i in range(num_colors):
        rgba = cmap(i / max(num_colors - 1, 1))
        colors.append(rgba[:3])  # RGB only
    
    return colors


def create_comparison_visualization(vertices, pred_trajectories, gt_trajectories,
                                   sample_name, output_path, max_trajectories=3):
    """
    Create comparison visualization of predictions vs ground truth.
    """
    try:
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Display mesh point cloud as background
        if vertices is not None and len(vertices) > 0:
            if len(vertices) > 5000:
                step = len(vertices) // 5000
                vertices_display = vertices[::step]
            else:
                vertices_display = vertices
            
            ax.scatter(vertices_display[:, 0], vertices_display[:, 1], 
                      vertices_display[:, 2], c='lightgray', s=0.5, 
                      alpha=0.1, marker='.')
        
        # Get colors from colormap
        pred_colors = get_colormap_colors('tab10', max_trajectories)
        
        # Display predicted trajectories
        for i in range(min(max_trajectories, len(pred_trajectories))):
            traj = pred_trajectories[i]
            if len(traj) > 0:
                color = pred_colors[i % len(pred_colors)]
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                       color=color, alpha=0.8, linewidth=2.5,
                       label=f'Pred Path {i}')
        
        # Display ground truth trajectories (white points)
        for i, traj in enumerate(gt_trajectories[:max_trajectories]):
            if len(traj) > 0:
                # Display as white points
                ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2],
                          c='black', s=20, alpha=0.8, marker='o',
                          edgecolors='none', label=f'GT Path {i}' if i == 0 else "")
        
        # Configure plot appearance
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        
        ax.set_title(f"Predictions vs Ground Truth: {sample_name}", 
                    fontsize=16, fontweight='bold')
        
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved comparison: {output_path}")
        plt.close(fig)
        
        return True
        
    except Exception as e:
        logging.error(f"Comparison visualization failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False


def create_confidence_visualization(vertices, trajectories, titles,
                                   output_path, sample_name):
    """
    Create visualization of trajectories colored by confidence.
    """
    try:
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Display mesh as point cloud background
        if vertices is not None and len(vertices) > 0:
            if len(vertices) > 10000:
                step = len(vertices) // 10000
                vertices_display = vertices[::step]
            else:
                vertices_display = vertices
            
            ax.scatter(vertices_display[:, 0], vertices_display[:, 1],
                      vertices_display[:, 2], c='lightgray', s=0.1,
                      alpha=0.2, marker='.')
        
        # Get colors from colormap
        colors = get_colormap_colors('tab10', len(trajectories))
        
        # Display trajectories
        for i, (traj, title, color) in enumerate(zip(trajectories, titles, colors)):
            if traj is not None and len(traj) > 0:
                # Draw trajectory line
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                       color=color, alpha=0.8, linewidth=2.0, label=title)
                
                # Draw start/end points
                if len(traj) > 1:
                    start_end_points = np.vstack([traj[0:1], traj[-1:]])
                    ax.scatter(start_end_points[:, 0], start_end_points[:, 1], 
                              start_end_points[:, 2], c=color, s=30, 
                              alpha=0.8, marker='o')
        
        # Configure plot appearance
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        
        ax.set_title(f"FoldPath Predictions: {sample_name}", 
                    fontsize=16, fontweight='bold')
        
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved: {output_path}")
        plt.close(fig)
        
        return True
        
    except Exception as e:
        logging.error(f"Visualization failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description='Visualize FoldPath predictions with matplotlib')
    
    parser.add_argument('--pred_dir', required=True, type=str, 
                       help='Directory containing all_predictions.npy')
    parser.add_argument('--normalized_root', default='/fileStore/windows-v2-normalized', 
                       type=str, help='Directory containing normalized OBJ files')
    parser.add_argument('--sample_dirs', nargs='+', default=['1_wr1fr_1'], 
                       help='Sample directory names')
    parser.add_argument('--output_dir', default=None, type=str, 
                       help='Output directory for visualizations')
    parser.add_argument('--max_samples', default=1, type=int, 
                       help='Maximum number of samples to visualize')
    parser.add_argument('--max_trajectories', default=10, type=int, 
                       help='Maximum number of trajectories to show')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.pred_dir, "plt_visualizations")
    
    logging.info("=" * 60)
    logging.info("FoldPath Matplotlib Visualizer")
    logging.info("=" * 60)
    logging.info(f"Predictions directory: {args.pred_dir}")
    logging.info(f"Normalized root: {args.normalized_root}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Samples: {args.sample_dirs}")
    logging.info(f"Max samples: {args.max_samples}")
    logging.info(f"Max trajectories: {args.max_trajectories}")
    logging.info("=" * 60)
    
    # Validate directories
    if not os.path.isdir(args.pred_dir):
        logging.error(f"Predictions directory does not exist: {args.pred_dir}")
        return
    
    if not os.path.isdir(args.normalized_root):
        logging.error(f"Normalized root does not exist: {args.normalized_root}")
        return
    
    # Load prediction data
    pred_file = os.path.join(args.pred_dir, "all_predictions.npy")
    if not os.path.exists(pred_file):
        logging.error(f"all_predictions.npy not found")
        return
    
    try:
        data = np.load(pred_file, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
        
        if not isinstance(data, dict) or 'traj_pred' not in data:
            logging.error("Invalid data format")
            return
        
        traj_pred = data['traj_pred']
        traj_gt = data.get('traj_gt', None)
        conf_pred = data.get('conf_pred', None)
        
        logging.info(f"Loaded predictions shape: {traj_pred.shape}")
        
        batch_size, num_queries, seq_len, features = traj_pred.shape
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Process each sample
        num_samples = min(batch_size, len(args.sample_dirs), args.max_samples)
        
        for sample_idx in range(num_samples):
            sample_dir = args.sample_dirs[sample_idx]
            logging.info(f"Processing sample {sample_idx+1}/{num_samples}: {sample_dir}")
            
            # Load mesh and normalization parameters
            vertices = None
            norm_params = None
            
            # Try to load OBJ file
            mesh_path = os.path.join(args.normalized_root, sample_dir, 
                                    f"{sample_dir}_norm.obj")
            if not os.path.exists(mesh_path):
                mesh_path = os.path.join(args.normalized_root, sample_dir, 
                                        f"{sample_dir}.obj")
            
            if os.path.exists(mesh_path):
                vertices = load_obj_vertices(mesh_path)
            
            # Load normalization parameters
            params_path = os.path.join(args.normalized_root, sample_dir, 
                                      "norm_params.json")
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    norm_params = json.load(f)
                logging.info(f"Loaded normalization parameters")
            
            # Denormalize mesh vertices
            if vertices is not None and norm_params is not None:
                vertices = denormalize_points(vertices, norm_params)
            
            # Extract predicted trajectories
            pred_trajectories = []
            for path_idx in range(min(args.max_trajectories, num_queries)):
                try:
                    trajectory = traj_pred[sample_idx, path_idx]
                    positions = trajectory[:, :3]  # Extract position coordinates
                    if norm_params is not None:
                        positions = denormalize_points(positions, norm_params)
                    pred_trajectories.append(positions)
                except Exception as e:
                    logging.warning(f"Error extracting pred path {path_idx}: {e}")
            
            # Extract ground truth trajectories
            gt_trajectories = []
            if traj_gt is not None:
                for path_idx in range(min(args.max_trajectories, traj_gt.shape[1])):
                    try:
                        trajectory = traj_gt[sample_idx, path_idx]
                        positions = trajectory[:, :3]
                        if norm_params is not None:
                            positions = denormalize_points(positions, norm_params)
                        gt_trajectories.append(positions)
                    except Exception as e:
                        logging.warning(f"Error extracting GT path {path_idx}: {e}")
            
            # Create comparison visualization with "_plt" suffix
            comparison_path = os.path.join(args.output_dir, 
                                         f"{sample_dir}_pred_vs_gt_plt.png")
            create_comparison_visualization(
                vertices=vertices,
                pred_trajectories=pred_trajectories,
                gt_trajectories=gt_trajectories,
                sample_name=sample_dir,
                output_path=comparison_path,
                max_trajectories=args.max_trajectories
            )
            
            # Create confidence-based visualization if available
            if conf_pred is not None and sample_idx < conf_pred.shape[0]:
                conf_scores = conf_pred[sample_idx]
                
                # Sort by confidence
                top_indices = np.argsort(conf_scores)[::-1][:args.max_trajectories]
                top_trajectories = []
                top_titles = []
                
                for i, idx in enumerate(top_indices):
                    if idx < len(pred_trajectories):
                        top_trajectories.append(pred_trajectories[idx])
                        top_titles.append(f"Path {idx} (conf={conf_scores[idx]:.3f})")
                
                if top_trajectories:
                    conf_path = os.path.join(args.output_dir, 
                                           f"{sample_dir}_top_conf_plt.png")
                    create_confidence_visualization(
                        vertices=vertices,
                        trajectories=top_trajectories,
                        titles=top_titles,
                        output_path=conf_path,
                        sample_name=sample_dir
                    )
    
    except Exception as e:
        logging.error(f"Error processing predictions: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    logging.info("\n" + "=" * 60)
    logging.info("Matplotlib Visualization completed!")
    logging.info(f"Visualizations saved in: {args.output_dir}")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()