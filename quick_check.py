#!/usr/bin/env python3
"""FoldPath Evaluation with DTW-based AP metrics (as per paper section IV.C)"""

import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch
from scipy.spatial.distance import euclidean
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.foldpath import FoldPath, FoldPathConfig
from utils.dataset.foldpath_dataset import FoldPathDataset, FoldPathDatasetConfig, foldpath_collate

try:
    from fastdtw import fastdtw
    HAS_DTW = True
except ImportError:
    print("Warning: fastdtw not installed. Run: pip install fastdtw")
    HAS_DTW = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate FoldPath with DTW-based AP metrics")
    
    # Required arguments
    p.add_argument("--dataset", type=str, required=True, help="Dataset name")
    p.add_argument("--data_root", type=str, action="append", required=True, help="Data root directories")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    
    # Model configuration
    p.add_argument("--num_queries", type=int, default=40)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--head_hidden", type=int, default=512)
    p.add_argument("--head_layers", type=int, default=4)
    p.add_argument("--tf_layers", type=int, default=4)
    p.add_argument("--tf_heads", type=int, default=4)
    p.add_argument("--activation", type=str, default="relu")
    
    # Evaluation parameters
    p.add_argument("--T", type=int, default=384, help="Number of waypoints per path")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--conf_thresh", type=float, default=0.35, help="Confidence threshold for filtering")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # DTW parameters (from paper IV.C)
    p.add_argument("--delta", type=float, default=0.025, help="Euclidean distance threshold (normalized)")
    p.add_argument("--theta", type=float, default=10.0, help="Angular threshold in degrees")
    p.add_argument("--ap_thresholds", type=str, default="0.5:0.95:0.05", 
                   help="AP thresholds in format start:stop:step")
    
    # Output
    p.add_argument("--out_dir", type=str, default="eval_results")
    p.add_argument("--save_predictions", action="store_true", help="Save predictions to file")
    
    return p.parse_args()


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> int:
    """Load checkpoint with flexible key handling"""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Try different possible keys
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and any(k.startswith("encoder") or k.startswith("decoder") for k in ckpt.keys()):
        state_dict = ckpt  # raw state dict
    else:
        raise ValueError("Could not find model weights in checkpoint")
    
    # Load with strict=False for flexibility
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    
    return ckpt.get("epoch", 0)


def extract_paths_from_predictions(predictions: dict, conf_thresh: float = 0.35):
    """Extract individual paths from model predictions based on stroke IDs"""
    traj_pred = predictions.get("traj_pred", np.zeros((0, 6)))
    stroke_ids = predictions.get("stroke_ids_pred", np.zeros(len(traj_pred), dtype=int))
    conf_scores = predictions.get("conf_pred", np.array([1.0]))
    
    if len(traj_pred) == 0:
        return []
    
    paths = []
    unique_stroke_ids = np.unique(stroke_ids)
    
    for stroke_id in unique_stroke_ids:
        mask = stroke_ids == stroke_id
        if np.sum(mask) == 0:
            continue
            
        path_points = traj_pred[mask]
        
        # Handle confidence scores
        if len(conf_scores) == len(traj_pred):
            path_confs = conf_scores[mask]
            avg_conf = np.mean(path_confs)
        elif len(conf_scores) == len(unique_stroke_ids):
            stroke_idx = np.where(unique_stroke_ids == stroke_id)[0][0]
            avg_conf = conf_scores[stroke_idx] if stroke_idx < len(conf_scores) else 1.0
        else:
            avg_conf = float(conf_scores[0]) if len(conf_scores) > 0 else 1.0
        
        # Filter by confidence
        if avg_conf < conf_thresh:
            continue
            
        # Sort by natural order (assuming stroke_id ordering is meaningful)
        paths.append({
            "points": path_points,
            "stroke_id": stroke_id,
            "confidence": float(avg_conf)
        })
    
    return paths


def extract_paths_from_ground_truth(gt_data: dict):
    """Extract individual paths from ground truth data"""
    traj_gt = gt_data.get("traj_as_pc", np.zeros((0, 6)))
    stroke_ids = gt_data.get("stroke_ids_as_pc", np.zeros(len(traj_gt), dtype=int))
    
    if len(traj_gt) == 0:
        return []
    
    paths = []
    unique_stroke_ids = np.unique(stroke_ids)
    
    for stroke_id in unique_stroke_ids:
        mask = stroke_ids == stroke_id
        if np.sum(mask) == 0:
            continue
            
        path_points = traj_gt[mask]
        paths.append({
            "points": path_points,
            "stroke_id": stroke_id
        })
    
    return paths


def dtw_f_score(pred_path: np.ndarray, gt_path: np.ndarray, delta: float = 0.025, theta: float = 10.0):
    """
    Calculate F-Score between two paths using DTW matching as described in paper IV.C
    
    Args:
        pred_path: (N, 6) predicted path [x, y, z, rx, ry, rz] or [x, y, z, qx, qy, qz, qw]
        gt_path: (M, 6) ground truth path
        delta: Euclidean distance threshold (normalized space)
        theta: Angular threshold in degrees
    
    Returns:
        f_score: F-Score between 0 and 1
        precision: Precision value
        recall: Recall value
    """
    if len(pred_path) == 0 or len(gt_path) == 0:
        return 0.0, 0.0, 0.0
    
    # Convert angle to radians
    theta_rad = np.deg2rad(theta)
    
    # Extract positions (first 3 columns)
    pred_pos = pred_path[:, :3]
    gt_pos = gt_path[:, :3]
    
    def path_distance(path1, path2):
        """Calculate DTW distance between two paths"""
        if not HAS_DTW:
            # Fallback: use simple nearest neighbor
            dist_matrix = np.linalg.norm(path1[:, None] - path2[None, :], axis=2)
            return dist_matrix.mean()
        
        # Use fastdtw for efficiency
        distance, path = fastdtw(path1, path2, dist=euclidean)
        return distance, path
    
    # Try forward and reverse order (as per paper)
    best_f_score = 0.0
    best_precision = 0.0
    best_recall = 0.0
    
    for reverse_pred in [False, True]:
        # Reverse predicted path if needed
        if reverse_pred:
            current_pred = np.flip(pred_path, axis=0)
            current_pred_pos = np.flip(pred_pos, axis=0)
        else:
            current_pred = pred_path
            current_pred_pos = pred_pos
        
        if HAS_DTW:
            # Use DTW to find optimal alignment
            _, alignment_path = fastdtw(current_pred_pos, gt_pos, dist=euclidean)
            
            # Count matches based on thresholds
            matched_pred = np.zeros(len(current_pred), dtype=bool)
            matched_gt = np.zeros(len(gt_path), dtype=bool)
            
            for pred_idx, gt_idx in alignment_path:
                # Position distance
                pos_dist = np.linalg.norm(current_pred_pos[pred_idx] - gt_pos[gt_idx])
                
                # Angle distance (if available)
                if current_pred.shape[1] >= 6 and gt_path.shape[1] >= 6:
                    # Extract orientation (assuming last 3 are Euler angles or rotation vector)
                    pred_rot = current_pred[pred_idx, 3:6]
                    gt_rot = gt_path[gt_idx, 3:6]
                    
                    # Normalize rotation vectors
                    pred_norm = np.linalg.norm(pred_rot)
                    gt_norm = np.linalg.norm(gt_rot)
                    
                    if pred_norm > 1e-6 and gt_norm > 1e-6:
                        pred_rot = pred_rot / pred_norm
                        gt_rot = gt_rot / gt_norm
                        angle = np.arccos(np.clip(np.dot(pred_rot, gt_rot), -1.0, 1.0))
                    else:
                        angle = 0.0
                else:
                    angle = 0.0
                
                # Check thresholds
                if pos_dist < delta and angle < theta_rad:
                    matched_pred[pred_idx] = True
                    matched_gt[gt_idx] = True
            
            # Calculate precision and recall
            precision = np.sum(matched_pred) / len(matched_pred)
            recall = np.sum(matched_gt) / len(matched_gt)
        else:
            # Fallback: simple nearest neighbor matching
            # For each pred point, find nearest gt point
            pred_to_gt_dist = np.min(np.linalg.norm(current_pred_pos[:, None] - gt_pos[None, :], axis=2), axis=1)
            pred_matched = pred_to_gt_dist < delta
            
            # For each gt point, find nearest pred point
            gt_to_pred_dist = np.min(np.linalg.norm(gt_pos[:, None] - current_pred_pos[None, :], axis=2), axis=1)
            gt_matched = gt_to_pred_dist < delta
            
            precision = np.mean(pred_matched)
            recall = np.mean(gt_matched)
        
        # Calculate F-Score
        if precision + recall > 0:
            f_score = 2 * precision * recall / (precision + recall)
        else:
            f_score = 0.0
        
        # Keep best F-Score
        if f_score > best_f_score:
            best_f_score = f_score
            best_precision = precision
            best_recall = recall
    
    return best_f_score, best_precision, best_recall


def calculate_ap_dtw(pred_paths_list, gt_paths_list, thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Calculate AP_DTW and AP_DTW_50 as per paper
    
    Args:
        pred_paths_list: List of lists of predicted paths for each sample
        gt_paths_list: List of lists of ground truth paths for each sample
        thresholds: F-Score thresholds for AP calculation
    
    Returns:
        Dictionary with AP metrics
    """
    all_f_scores = []
    
    # For each sample, find best matching paths
    for pred_paths, gt_paths in zip(pred_paths_list, gt_paths_list):
        if len(gt_paths) == 0:
            all_f_scores.append(0.0)
            continue
        
        # Find best F-Score for each ground truth path
        gt_f_scores = []
        for gt_path in gt_paths:
            best_gt_f = 0.0
            for pred_path in pred_paths:
                f_score, _, _ = dtw_f_score(pred_path["points"], gt_path["points"])
                best_gt_f = max(best_gt_f, f_score)
            gt_f_scores.append(best_gt_f)
        
        # Average F-Score across ground truth paths
        avg_f = np.mean(gt_f_scores) if gt_f_scores else 0.0
        all_f_scores.append(avg_f)
    
    # Calculate AP_DTW_50 (path correct if F-Score >= 0.5)
    ap50 = np.mean([1.0 if f >= 0.5 else 0.0 for f in all_f_scores])
    
    # Calculate AP_DTW (average across multiple thresholds)
    aps = []
    for thr in thresholds:
        ap = np.mean([1.0 if f >= thr else 0.0 for f in all_f_scores])
        aps.append(ap)
    
    ap_mean = np.mean(aps)
    
    return {
        "AP_DTW": float(ap_mean),
        "AP_DTW_50": float(ap50),
        "mean_F_score": float(np.mean(all_f_scores)),
        "std_F_score": float(np.std(all_f_scores)),
        "all_F_scores": [float(f) for f in all_f_scores]
    }


def calculate_chamfer_distance(pred_points, gt_points):
    """Calculate Chamfer Distance (for comparison)"""
    if len(pred_points) == 0 or len(gt_points) == 0:
        return 1.0
    
    # Sample for efficiency
    pred_sampled = pred_points[::max(1, len(pred_points) // 100)]
    gt_sampled = gt_points[::max(1, len(gt_points) // 100)]
    
    # Compute pairwise distances
    dist_matrix = np.linalg.norm(pred_sampled[:, None, :3] - gt_sampled[None, :, :3], axis=2)
    
    # Chamfer distance
    pred_to_gt = np.min(dist_matrix, axis=1)
    gt_to_pred = np.min(dist_matrix, axis=0)
    
    cd = 0.5 * (np.mean(pred_to_gt) + np.mean(gt_to_pred))
    return float(cd)


def evaluate_model(args):
    """Main evaluation function"""
    print("=" * 60)
    print("FoldPath Evaluation with DTW-based AP metrics")
    print("=" * 60)
    
    start_time = time.time()
    
    # Setup device
    device = torch.device(args.device)
    
    # Parse AP thresholds
    thr_start, thr_stop, thr_step = map(float, args.ap_thresholds.split(":"))
    ap_thresholds = np.arange(thr_start, thr_stop + thr_step/2, thr_step)
    
    # Create model
    model_cfg = FoldPathConfig(
        num_queries=args.num_queries,
        d_model=args.d_model,
        head_hidden=args.head_hidden,
        head_layers=args.head_layers,
        tf_layers=args.tf_layers,
        tf_heads=args.tf_heads,
        T_train=args.T,
        T_test=args.T,
        activation=args.activation,
    )
    model = FoldPath(model_cfg)
    
    # Load checkpoint
    print(f"[1/4] Loading checkpoint: {args.ckpt}")
    epoch = load_checkpoint(model, args.ckpt, device)
    model.eval()
    
    # Create dataset
    print(f"[2/4] Loading {args.split} dataset")
    ds_config = FoldPathDatasetConfig(
        dataset=args.dataset,
        roots=args.data_root,
        split=args.split,
        pc_points=5120,
        normalization="per-mesh",
        augmentations=[],
        num_queries=args.num_queries,
        T=args.T,
        sampling="equispaced",
    )
    dataset = FoldPathDataset(ds_config)
    
    # Create data loader
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=foldpath_collate,
        drop_last=False
    )
    
    # Inference and collect predictions
    print(f"[3/4] Running inference on {len(dataset)} samples")
    
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Inference")):
            pc = batch["pc"].to(device)
            
            # Get predictions
            predictions = model.infer(pc, T=args.T, max_paths=args.num_queries, conf_thresh=args.conf_thresh)
            
            # Process each sample in batch
            for i in range(len(pc)):
                sample_idx = batch_idx * args.batch_size + i
                
                # Debug print to understand prediction structure
                if batch_idx == 0 and i == 0:
                    print(f"\nDebug - First prediction structure:")
                    for key, value in predictions[i].items():
                        if isinstance(value, np.ndarray):
                            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                        else:
                            print(f"  {key}: {type(value)}")
                
                # Store predictions
                pred_dict = {
                    "traj_pred": predictions[i]["traj_pred"].astype(np.float32),
                    "stroke_ids_pred": predictions[i]["stroke_ids_pred"].astype(np.int64),
                    "conf_pred": predictions[i].get("conf_pred", np.array([1.0])).astype(np.float32),
                    "sample_idx": sample_idx,
                }
                all_predictions.append(pred_dict)
                
                # Store ground truth
                f_gt = batch["f"][i]
                y_gt = batch["y"][i]
                
                # Extract valid paths (f > 0.5)
                valid_mask = f_gt > 0.5
                if valid_mask.sum() > 0:
                    valid_paths = y_gt[valid_mask]
                    # Combine all valid paths into single array with stroke IDs
                    traj_gt_list = []
                    stroke_ids_list = []
                    for j, path in enumerate(valid_paths):
                        n_points = len(path)
                        traj_gt_list.append(path.numpy())
                        stroke_ids_list.append(np.full(n_points, j, dtype=np.int64))
                    
                    if traj_gt_list:
                        traj_gt = np.vstack(traj_gt_list)
                        stroke_ids_gt = np.concatenate(stroke_ids_list)
                    else:
                        traj_gt = np.zeros((0, 6), dtype=np.float32)
                        stroke_ids_gt = np.zeros(0, dtype=np.int64)
                else:
                    traj_gt = np.zeros((0, 6), dtype=np.float32)
                    stroke_ids_gt = np.zeros(0, dtype=np.int64)
                
                gt_dict = {
                    "traj_as_pc": traj_gt,
                    "stroke_ids_as_pc": stroke_ids_gt,
                    "sample_idx": sample_idx,
                }
                all_ground_truth.append(gt_dict)
    
    # Extract individual paths for AP calculation
    print(f"[4/4] Calculating metrics")
    
    pred_paths_by_sample = []
    gt_paths_by_sample = []
    chamfer_distances = []
    
    for pred, gt in zip(all_predictions, all_ground_truth):
        # Debug first sample
        if len(pred_paths_by_sample) == 0:
            print(f"\nDebug - First sample shapes:")
            print(f"  traj_pred shape: {pred['traj_pred'].shape}")
            print(f"  stroke_ids_pred shape: {pred['stroke_ids_pred'].shape}")
            print(f"  conf_pred shape: {pred['conf_pred'].shape}")
            print(f"  unique stroke_ids: {np.unique(pred['stroke_ids_pred'])}")
        
        # Extract paths
        pred_paths = extract_paths_from_predictions(pred, args.conf_thresh)
        gt_paths = extract_paths_from_ground_truth(gt)
        
        pred_paths_by_sample.append(pred_paths)
        gt_paths_by_sample.append(gt_paths)
        
        # Calculate Chamfer distance (for reference)
        cd = calculate_chamfer_distance(
            pred["traj_pred"] if len(pred["traj_pred"]) > 0 else np.zeros((0, 3)),
            gt["traj_as_pc"] if len(gt["traj_as_pc"]) > 0 else np.zeros((0, 3))
        )
        chamfer_distances.append(cd)
    
    # Calculate DTW-based AP metrics
    print("  Calculating DTW-based AP metrics...")
    ap_metrics = calculate_ap_dtw(pred_paths_by_sample, gt_paths_by_sample, ap_thresholds)
    
    # Additional statistics
    num_pred_paths = [len(p) for p in pred_paths_by_sample]
    num_gt_paths = [len(p) for p in gt_paths_by_sample]
    
    avg_pred_paths = np.mean(num_pred_paths) if num_pred_paths else 0.0
    avg_gt_paths = np.mean(num_gt_paths) if num_gt_paths else 0.0
    path_coverage = avg_pred_paths / max(avg_gt_paths, 1.0)
    
    # Calculate average confidence
    all_confidences = []
    for pred in all_predictions:
        if "conf_pred" in pred and len(pred["conf_pred"]) > 0:
            # Handle different conf_pred shapes
            if len(pred["conf_pred"].shape) == 0 or pred["conf_pred"].size == 1:
                all_confidences.append(float(pred["conf_pred"]))
            else:
                all_confidences.extend(pred["conf_pred"].flatten())
    
    avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
    
    # Compile final metrics
    final_metrics = {
        # DTW-based AP metrics (main metrics from paper)
        "AP_DTW": ap_metrics["AP_DTW"],
        "AP_DTW_50": ap_metrics["AP_DTW_50"],
        "mean_F_score": ap_metrics["mean_F_score"],
        "std_F_score": ap_metrics["std_F_score"],
        
        # Traditional distance metrics
        "Chamfer_Distance": float(np.mean(chamfer_distances)) if chamfer_distances else 1.0,
        "Chamfer_Distance_std": float(np.std(chamfer_distances)) if chamfer_distances else 0.0,
        
        # Path statistics
        "avg_predicted_paths": float(avg_pred_paths),
        "avg_ground_truth_paths": float(avg_gt_paths),
        "path_coverage": float(path_coverage),
        "avg_confidence": float(avg_confidence),
        
        # Evaluation info
        "num_samples": len(all_predictions),
        "checkpoint_epoch": epoch,
        "evaluation_time_seconds": time.time() - start_time,
        "dtw_available": HAS_DTW,
        "dtw_params": {
            "delta": args.delta,
            "theta": args.theta,
            "ap_thresholds": f"{thr_start}:{thr_stop}:{thr_step}"
        }
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"DTW-based AP Metrics (Paper Section IV.C):")
    print(f"  AP_DTW:       {final_metrics['AP_DTW']:.4f}")
    print(f"  AP_DTW_50:    {final_metrics['AP_DTW_50']:.4f}")
    print(f"  Mean F-Score: {final_metrics['mean_F_score']:.4f} ± {final_metrics['std_F_score']:.4f}")
    
    print(f"Distance Metrics:")
    print(f"  Chamfer Distance: {final_metrics['Chamfer_Distance']:.4f} ± {final_metrics['Chamfer_Distance_std']:.4f}")
    
    print(f"Path Statistics:")
    print(f"  Avg predicted paths:  {final_metrics['avg_predicted_paths']:.1f}")
    print(f"  Avg ground truth paths: {final_metrics['avg_ground_truth_paths']:.1f}")
    print(f"  Path coverage:        {final_metrics['path_coverage']:.3f}")
    print(f"  Avg confidence:       {final_metrics['avg_confidence']:.3f}")
    
    print(f"Configuration:")
    print(f"  Samples evaluated: {final_metrics['num_samples']}")
    print(f"  Checkpoint epoch:  {final_metrics['checkpoint_epoch']}")
    print(f"  Time: {final_metrics['evaluation_time_seconds']:.1f} seconds")
    print(f"  DTW available: {final_metrics['dtw_available']}")
    print("=" * 60)
    
    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save metrics
    metrics_file = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")
    
    # Save predictions if requested
    if args.save_predictions:
        predictions_file = os.path.join(args.out_dir, "predictions.json")
        predictions_data = {
            "predictions": all_predictions,
            "ground_truth": all_ground_truth,
            "config": vars(args)
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            return obj
        
        with open(predictions_file, "w") as f:
            json.dump(predictions_data, f, indent=2, default=convert_numpy)
        print(f"Predictions saved to: {predictions_file}")
    
    return final_metrics


def main():
    args = parse_args()
    
    if not HAS_DTW:
        print("Warning: fastdtw is required for accurate DTW-based AP calculation.")
        print("Install with: pip install fastdtw")
        print("Falling back to simpler distance metrics.")
    
    metrics = evaluate_model(args)


if __name__ == "__main__":
    main()
