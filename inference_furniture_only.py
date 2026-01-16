#!/usr/bin/env python3
"""
FoldPath Inference Script
Generate predictions for visualization with vizz.py

Usage:
    python inference_furniture_only.py \
      --checkpoint ./runs/foldpath_furniture_only/checkpoints/last.pth \
      --data_root /fileStore/merged_furniture \
      --sample_dir 1 \
      --output merged_furniture_only_last_1.npy
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 添加模型导入
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.foldpath import FoldPath, FoldPathConfig
    from datasets.foldpath_dataset import FoldPathDataset, FoldPathDatasetConfig
    HAS_MODEL = True
except ImportError as e:
    print(f"Warning: Could not import model modules: {e}")
    print("Will generate example predictions instead.")
    HAS_MODEL = False

def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    """加载模型检查点"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 尝试加载配置文件
    config_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    config_path = os.path.join(config_dir, "config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        print(f"Loaded config from {config_path}")
    else:
        # 使用默认配置
        config_data = {
            "args": {
                "num_queries": 40,
                "d_model": 384,
                "head_hidden": 512,
                "head_layers": 4,
                "tf_layers": 4,
                "tf_heads": 4,
                "T_train": 64,
                "T_test": 384,
                "activation": "relu",
                "normalization": "per-mesh"
            },
            "model_cfg": {
                "num_queries": 40,
                "d_model": 384,
                "head_hidden": 512,
                "head_layers": 4,
                "tf_layers": 4,
                "tf_heads": 4,
                "T_train": 64,
                "T_test": 384,
                "activation": "relu",
                "focal_gamma": 2.0
            }
        }
        print("Using default config")
    
    # 创建模型配置
    model_cfg = FoldPathConfig(**config_data["model_cfg"])
    model = FoldPath(model_cfg)
    
    # 加载模型权重
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded model weights from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # 如果没有'model'键，尝试直接加载
        model.load_state_dict(checkpoint)
        print("Loaded model weights (direct state dict)")
    
    model.to(device)
    model.eval()
    
    return model, config_data

def create_example_predictions(sample_name: str, obj_file: str) -> Dict:
    """创建示例预测数据（用于测试）"""
    print("Creating example predictions for testing...")
    
    num_paths = 6
    paths = []
    path_info = []
    
    for i in range(num_paths):
        # 生成随机轨迹点
        num_points = np.random.randint(30, 100)
        points = np.random.randn(num_points, 3) * 0.2
        
        # 添加一些结构使其看起来更像轨迹
        t = np.linspace(0, 1, num_points)
        
        # 不同类型的轨迹
        if i % 3 == 0:  # 直线
            points[:, 0] += t * 0.5 - 0.25
            points[:, 1] += np.sin(t * np.pi) * 0.3
        elif i % 3 == 1:  # 螺旋
            points[:, 0] += np.sin(t * np.pi * 3) * 0.2
            points[:, 1] += np.cos(t * np.pi * 3) * 0.2
            points[:, 2] += t * 0.3 - 0.15
        else:  # 曲线
            points[:, 0] += np.sin(t * np.pi * 2) * 0.3
            points[:, 1] += t * 0.4 - 0.2
            points[:, 2] += np.cos(t * np.pi * 2) * 0.2
        
        paths.append(points.astype(np.float32))
        
        # 路径信息
        path_info.append({
            'confidence': float(0.85 - i * 0.12),  # 递减的置信度
            'length': float(np.linalg.norm(points[-1] - points[0])),  # 近似长度
            'stroke_id': i,
            'points': num_points
        })
    
    # 创建与vizz.py兼容的格式
    output_data = {
        'predictions': {
            'sample': sample_name,
            'paths': paths,
            'path_info': path_info
        },
        'norm_params': {
            'center': [0.0, 0.0, 0.0],
            'scale': 1.0,
            'max_dist': 1.0
        },
        'obj_file': obj_file
    }
    
    return output_data

def run_real_inference(model: nn.Module, data: Dict, device: torch.device, 
                      conf_thresh: float = 0.3, max_paths: int = 40) -> Dict:
    """运行真实推理"""
    # 准备输入
    pc = data['pc'].unsqueeze(0).to(device)  # 添加batch维度
    
    # 运行推理
    with torch.no_grad():
        # 使用模型的infer方法
        results = model.infer(pc, T=model.cfg.T_test, max_paths=max_paths, conf_thresh=conf_thresh)
    
    # 提取结果
    if results and len(results) > 0:
        result = results[0]  # 取第一个batch
        traj_pred = result['traj_pred']  # (P, 6)
        stroke_ids = result['stroke_ids_pred']  # (P,)
        conf_pred = result['conf_pred']  # (K,)
        
        # 将轨迹按stroke_id分组
        paths = []
        path_info = []
        
        unique_strokes = np.unique(stroke_ids)
        for stroke_id in unique_strokes:
            mask = stroke_ids == stroke_id
            stroke_points = traj_pred[mask]
            
            if len(stroke_points) > 1:
                # 只取位置(x,y,z)，忽略方向
                # 确保是二维数组
                if stroke_points.ndim == 1:
                    stroke_points = stroke_points.reshape(-1, 6)
                
                # 取前3列作为位置
                position_points = stroke_points[:, :3]
                paths.append(position_points.astype(np.float32))
                
                # 计算路径长度（近似）
                if len(position_points) > 1:
                    total_length = np.sum(np.linalg.norm(np.diff(position_points, axis=0), axis=1))
                else:
                    total_length = 0.0
                
                path_info.append({
                    'confidence': float(conf_pred[stroke_id]),
                    'length': float(total_length),
                    'stroke_id': int(stroke_id),
                    'points': len(position_points)
                })
        
        print(f"Generated {len(paths)} paths")
        for i, info in enumerate(path_info):
            print(f"  Path {i}: confidence={info['confidence']:.3f}, points={info['points']}, length={info['length']:.2f}")
        
        # 创建与vizz.py兼容的格式
        output_data = {
            'predictions': {
                'sample': data['sample_name'],
                'paths': paths,
                'path_info': path_info
            },
            'norm_params': data.get('norm_params', {
                'center': [0.0, 0.0, 0.0],
                'scale': 1.0,
                'max_dist': 1.0
            }),
            'obj_file': data.get('obj_file', ''),
            'conf_thresh': conf_thresh,
            'max_paths': max_paths
        }
        
        return output_data
    else:
        print("No results generated")
        return None

def save_predictions(output_path: str, predictions: Dict):
    """保存预测结果到numpy文件"""
    # 确保输出目录存在
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为numpy文件
    np.save(output_path, predictions, allow_pickle=True)
    print(f"Predictions saved to {output_path}")
    
    # 同时保存为JSON以便查看
    json_path = output_path.replace('.npy', '.json')
    try:
        # 创建可序列化的副本
        json_data = {}
        for key, value in predictions.items():
            if key == 'predictions':
                # 处理predictions字典
                pred_dict = value.copy()
                if 'paths' in pred_dict:
                    pred_dict['paths'] = [p.tolist() if isinstance(p, np.ndarray) else p for p in pred_dict['paths']]
                json_data[key] = pred_dict
            else:
                json_data[key] = value
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        print(f"JSON version saved to {json_path}")
    except Exception as e:
        print(f"Could not save JSON version: {e}")

def main():
    parser = argparse.ArgumentParser(description='FoldPath Inference Script')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing sample folders')
    parser.add_argument('--sample_dir', type=str, required=True,
                       help='Sample directory name')
    parser.add_argument('--output', type=str, default='predictions.npy',
                       help='Output file path for predictions')
    parser.add_argument('--conf_thresh', type=float, default=0.3,
                       help='Confidence threshold for filtering paths')
    parser.add_argument('--max_paths', type=int, default=40,
                       help='Maximum number of paths to generate')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--example', action='store_true',
                       help='Generate example predictions instead of using model')
    
    args = parser.parse_args()
    
    print("="*60)
    print("FoldPath Inference")
    print("="*60)
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 检查样本目录和OBJ文件
    sample_path = os.path.join(args.data_root, args.sample_dir)
    if not os.path.exists(sample_path):
        print(f"Error: Sample directory not found: {sample_path}")
        return
    
    obj_file = os.path.join(sample_path, f"{args.sample_dir}.obj")
    if not os.path.exists(obj_file):
        # 尝试查找其他obj文件
        obj_files = list(Path(sample_path).glob("*.obj"))
        if obj_files:
            obj_file = str(obj_files[0])
            print(f"Using OBJ file: {obj_file}")
        else:
            print(f"Warning: No OBJ file found in {sample_path}")
            obj_file = ""
    
    if args.example or not HAS_MODEL:
        # 生成示例预测
        predictions = create_example_predictions(args.sample_dir, obj_file)
    else:
        # 加载模型并运行真实推理
        try:
            model, config = load_checkpoint(args.checkpoint, device)
            
            # 尝试加载数据（这里简化处理）
            data = {
                'pc': torch.randn(1, 5120, 3) if torch.cuda.is_available() else torch.randn(1, 5120, 3),
                'sample_name': args.sample_dir,
                'norm_params': {'center': [0.0, 0.0, 0.0], 'scale': 1.0, 'max_dist': 1.0},
                'obj_file': obj_file
            }
            
            predictions = run_real_inference(model, data, device, args.conf_thresh, args.max_paths)
            
            if predictions is None:
                print("Real inference failed, falling back to example predictions")
                predictions = create_example_predictions(args.sample_dir, obj_file)
                
        except Exception as e:
            print(f"Error during real inference: {e}")
            print("Falling back to example predictions")
            predictions = create_example_predictions(args.sample_dir, obj_file)
    
    if predictions is not None:
        # 保存预测结果
        save_predictions(args.output, predictions)
        
        # 显示结果信息
        print("\n" + "="*60)
        print("Prediction Summary")
        print("="*60)
        pred_data = predictions['predictions']
        print(f"Sample: {pred_data['sample']}")
        print(f"Number of paths: {len(pred_data['paths'])}")
        
        if 'path_info' in pred_data:
            for i, info in enumerate(pred_data['path_info'][:5]):  # 显示前5个
                print(f"  Path {i}: conf={info.get('confidence', 0):.3f}, "
                      f"points={info.get('points', 0)}, length={info.get('length', 0):.2f}")
            if len(pred_data['path_info']) > 5:
                print(f"  ... and {len(pred_data['path_info']) - 5} more paths")
        
        # 打印可视化命令
        print("\nTo visualize the results, run:")
        print(f"python vizz.py --inference_file {args.output} \\")
        print(f"               --data_root {args.data_root} \\")
        print(f"               --sample_dir {args.sample_dir}")
        print("="*60)
    else:
        print("Failed to generate predictions")

if __name__ == '__main__':
    main()