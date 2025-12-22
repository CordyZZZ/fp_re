#!/usr/bin/env python3
"""
FoldPath VTK Visualizer - Clean version
Shows full 3D object mesh with predicted (colored polylines) and ground truth (white points)
"""

import argparse
import json
import logging
import os
import sys
import numpy as np
import vtk
from typing import List, Optional, Dict, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ===================== Configuration =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== OBJ Mesh Loading Functions =====================
def load_obj_mesh(obj_path: str, norm_params: Dict) -> Optional[vtk.vtkPolyData]:
    """
    Load OBJ mesh and denormalize it
    """
    try:
        # Load OBJ file
        obj_reader = vtk.vtkOBJReader()
        obj_reader.SetFileName(obj_path)
        obj_reader.Update()
        
        polydata = obj_reader.GetOutput()
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            logger.warning(f"OBJ file has no valid vertices: {obj_path}")
            return None
        
        points = polydata.GetPoints()
        n_points = points.GetNumberOfPoints()
        
        # Get vertices as numpy array
        vertices_np = np.zeros((n_points, 3), dtype=np.float32)
        for i in range(n_points):
            vertices_np[i] = points.GetPoint(i)
        
        # Denormalize vertices
        center = np.array(norm_params.get('center', [0.0, 0.0, 0.0]), dtype=np.float32)
        scale = float(norm_params.get('scale', 1.0))
        
        vertices_denorm = vertices_np.copy()
        vertices_denorm[:, 0] = vertices_denorm[:, 0] * scale + center[0]
        vertices_denorm[:, 1] = vertices_denorm[:, 1] * scale + center[1]
        vertices_denorm[:, 2] = vertices_denorm[:, 2] * scale + center[2]
        
        # Update VTK PolyData with denormalized points
        new_points = vtk.vtkPoints()
        for i in range(len(vertices_denorm)):
            new_points.InsertPoint(i, vertices_denorm[i])
        polydata.SetPoints(new_points)
        polydata.Modified()
        
        return polydata
        
    except Exception as e:
        logger.error(f"Failed to load OBJ mesh: {e}")
        return None

def create_mesh_actor(polydata: vtk.vtkPolyData) -> vtk.vtkActor:
    """
    Create VTK actor for OBJ mesh
    """
    try:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.85, 0.85, 0.85)  # Light gray
        actor.GetProperty().SetOpacity(0.9)
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(20)
        actor.GetProperty().SetDiffuse(0.7)
        actor.GetProperty().SetAmbient(0.3)
        
        # Show mesh edges
        actor.GetProperty().SetEdgeColor(0.5, 0.5, 0.5)
        actor.GetProperty().SetEdgeVisibility(True)
        actor.GetProperty().SetLineWidth(0.5)
        
        return actor
        
    except Exception as e:
        logger.error(f"Failed to create mesh actor: {e}")
        return None

# ===================== Data Loading and Processing =====================
def load_predictions(pred_file: str) -> Dict[str, Any]:
    """
    Load prediction data from npy file
    """
    try:
        data = np.load(pred_file, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
        
        if not isinstance(data, dict) or 'traj_pred' not in data:
            logger.error("Invalid data format in predictions file")
            return {}
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to load predictions: {e}")
        return {}

def denormalize_points(points: np.ndarray, norm_params: Dict) -> np.ndarray:
    """
    Denormalize point coordinates
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
        logger.error(f"Failed to denormalize points: {e}")
        return points

def extract_trajectories(data: Dict[str, Any], sample_idx: int, 
                        norm_params: Optional[Dict] = None,
                        trajectory_type: str = 'pred',
                        max_trajectories: int = 10) -> List[np.ndarray]:
    """
    Extract trajectories from predictions data
    """
    trajectories = []
    
    try:
        if trajectory_type == 'pred':
            traj_key = 'traj_pred'
        elif trajectory_type == 'gt':
            traj_key = 'traj_gt'
        else:
            return trajectories
        
        if traj_key not in data:
            return trajectories
        
        traj_data = data[traj_key]
        
        if traj_data.ndim != 4:
            return trajectories
        
        batch_size, num_queries, _, _ = traj_data.shape
        
        if sample_idx >= batch_size:
            return trajectories
        
        # Extract trajectories
        for path_idx in range(min(num_queries, max_trajectories)):
            try:
                trajectory = traj_data[sample_idx, path_idx]
                positions = trajectory[:, :3]  # Extract position coordinates
                
                # Denormalize if parameters provided
                if norm_params is not None:
                    positions = denormalize_points(positions, norm_params)
                
                # Filter invalid values
                mask = ~np.isnan(positions).any(axis=1) & ~np.isinf(positions).any(axis=1)
                positions = positions[mask]
                
                if len(positions) > 1:  # Need at least 2 points for a line
                    trajectories.append(positions)
                    
            except Exception:
                continue
        
        return trajectories
        
    except Exception:
        return trajectories

def get_colormap_colors(colormap_name: str = 'tab10', num_colors: int = 10) -> List[Tuple[float, float, float]]:
    """
    get color from matplotlib colormap
    """
    # get colormap
    cmap = plt.colormaps[colormap_name] if colormap_name in plt.colormaps else plt.colormaps.get_cmap(colormap_name)
    
    colors = []
    for i in range(num_colors):
        rgba = cmap(i / max(num_colors - 1, 1))
        colors.append(rgba[:3])    
    return colors

# ===================== Trajectory Visualization Functions =====================
def create_trajectory_actor(trajectory: np.ndarray,
                           color: Tuple[float, float, float],
                           line_width: float = 3.0,
                           opacity: float = 0.8) -> Optional[vtk.vtkActor]:
    """
    Create VTK actor for trajectory polyline
    """
    try:
        if trajectory is None or len(trajectory) < 2:
            return None
        
        # Create points
        points = vtk.vtkPoints()
        for point in trajectory:
            points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
        
        # Create polyline
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(trajectory))
        for i in range(len(trajectory)):
            polyline.GetPointIds().SetId(i, i)
        
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyline)
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(cells)
        
        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color[0], color[1], color[2])
        actor.GetProperty().SetLineWidth(line_width)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetLighting(False)
        
        return actor
        
    except Exception:
        return None

def create_trajectory_points_actor(trajectory: np.ndarray,
                                 color: Tuple[float, float, float],
                                 point_size: float = 8.0,
                                 opacity: float = 1.0,
                                 is_ground_truth: bool = False) -> Optional[vtk.vtkActor]:
    """
    Create VTK actor for trajectory points
    """
    try:
        if trajectory is None or len(trajectory) == 0:
            return None
        
        if is_ground_truth:
            # Ground truth: 显示所有点，但采样避免过多
            if len(trajectory) > 30:  # 减少采样点
                step = len(trajectory) // 30
                display_points = trajectory[::step]
            else:
                display_points = trajectory
        else:
            # 预测轨迹：只显示起点和终点
            display_points = np.vstack([trajectory[0:1], trajectory[-1:]])
        
        # Create VTK points
        vtk_points = vtk.vtkPoints()
        vtk_vertices = vtk.vtkCellArray()
        
        for i, point in enumerate(display_points):
            point_id = vtk_points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
            vtk_vertices.InsertNextCell(1)
            vtk_vertices.InsertCellPoint(point_id)
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetVerts(vtk_vertices)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color[0], color[1], color[2])
        actor.GetProperty().SetPointSize(point_size)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetLighting(False)
        
        return actor
        
    except Exception:
        return None

# ===================== Scene Setup and Camera =====================
def calculate_scene_bounds(mesh_polydata: Optional[vtk.vtkPolyData],
                          pred_trajectories: List[np.ndarray],
                          gt_trajectories: List[np.ndarray]) -> List[float]:
    """
    Calculate bounds that encompass entire scene
    """
    all_points = []
    
    # Add mesh points
    if mesh_polydata:
        mesh_points = mesh_polydata.GetPoints()
        if mesh_points:
            n_points = mesh_points.GetNumberOfPoints()
            mesh_vertices = np.zeros((n_points, 3))
            for i in range(n_points):
                mesh_vertices[i] = mesh_points.GetPoint(i)
            all_points.append(mesh_vertices)
    
    # Add trajectory points
    for traj in pred_trajectories:
        if traj is not None and len(traj) > 0:
            all_points.append(traj)
    
    for traj in gt_trajectories:
        if traj is not None and len(traj) > 0:
            all_points.append(traj)
    
    if not all_points:
        return [-1, 1, -1, 1, -1, 1]
    
    # Combine all points
    all_points_concat = np.vstack([p for p in all_points if len(p) > 0])
    
    # Calculate bounds with padding
    bounds = [
        float(np.min(all_points_concat[:, 0])),
        float(np.max(all_points_concat[:, 0])),
        float(np.min(all_points_concat[:, 1])),
        float(np.max(all_points_concat[:, 1])),
        float(np.min(all_points_concat[:, 2])),
        float(np.max(all_points_concat[:, 2]))
    ]
    
    # Add padding to ensure everything is visible
    padding_x = (bounds[1] - bounds[0]) * 0.1
    padding_y = (bounds[3] - bounds[2]) * 0.1
    padding_z = (bounds[5] - bounds[4]) * 0.1
    
    return [
        bounds[0] - padding_x, bounds[1] + padding_x,
        bounds[2] - padding_y, bounds[3] + padding_y,
        bounds[4] - padding_z, bounds[5] + padding_z
    ]

def setup_camera(renderer: vtk.vtkRenderer, bounds: List[float]) -> None:
    """
    Setup camera to show entire scene
    """
    try:
        camera = renderer.GetActiveCamera()
        
        # Calculate center of scene
        center = [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ]
        
        # Calculate maximum dimension
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        dz = bounds[5] - bounds[4]
        max_dim = max(dx, dy, dz)
        
        # Set camera distance
        distance = max_dim * 0.5
        
        # Set camera position to show isometric view
        camera.SetPosition(
            center[0] + distance * 0.5,
            center[1] - distance * 0.5,
            center[2] + distance * 0.5
        )
        camera.SetFocalPoint(center[0], center[1], center[2])
        camera.SetViewUp(0, 0, 1)
        
        # Set clipping planes
        near_clip = 0.1
        far_clip = distance * 3
        camera.SetClippingRange(near_clip, far_clip)
        
        # Reset camera to ensure everything is visible
        renderer.ResetCamera(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
        
        # Zoom out a bit more for safety
        camera.Zoom(1.2)
        
    except Exception:
        pass

# ===================== Visualization Functions =====================
def create_visualization(mesh_polydata: Optional[vtk.vtkPolyData],
                        pred_trajectories: List[np.ndarray],
                        gt_trajectories: List[np.ndarray],
                        sample_name: str,
                        output_path: str,
                        max_trajectories: int = 10) -> bool:
    """
    Create visualization showing mesh with trajectories
    - Prediction: Colored polylines with start/end points
    - Ground Truth: White points only
    """
    try:
        # Create renderer with light background
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.95, 0.95, 0.95)
        
        # Add lighting
        light = vtk.vtkLight()
        light.SetPosition(5, 5, 10)
        light.SetFocalPoint(0, 0, 0)
        light.SetIntensity(0.8)
        renderer.AddLight(light)
        
        fill_light = vtk.vtkLight()
        fill_light.SetPosition(-5, -5, 5)
        fill_light.SetFocalPoint(0, 0, 0)
        fill_light.SetIntensity(0.4)
        renderer.AddLight(fill_light)
        
        # Add mesh actor (if available)
        if mesh_polydata:
            mesh_actor = create_mesh_actor(mesh_polydata)
            if mesh_actor:
                renderer.AddActor(mesh_actor)
        
        # Get colors for predictions (tab10 colormap)
        pred_colors = get_colormap_colors('tab10', max_trajectories)
        
        # Add predicted trajectories (colored polylines)
        for i, traj in enumerate(pred_trajectories[:max_trajectories]):
            if len(traj) > 1:
                color = pred_colors[i % len(pred_colors)]
                
                # Add trajectory polyline
                line_actor = create_trajectory_actor(
                    traj,
                    color=color,
                    line_width=10.0,
                    opacity=0.8
                )
                if line_actor:
                    renderer.AddActor(line_actor)
                
                # Add start/end points for prediction
                points_actor = create_trajectory_points_actor(
                    traj,
                    color=color,
                    point_size=10.0,
                    opacity=0.8,
                    is_ground_truth=False
                )
                if points_actor:
                    renderer.AddActor(points_actor)
        
        # Add ground truth trajectories (white points only)
        for i, traj in enumerate(gt_trajectories[:max_trajectories]):
            if len(traj) > 0:
                color = (1.0, 1.0, 1.0)

                points_actor = create_trajectory_points_actor(
                    traj,
                    color=color,
                    point_size=10.0,
                    opacity=1.0,
                    is_ground_truth=True
                )
                if points_actor:
                    renderer.AddActor(points_actor)
        
        # Calculate scene bounds and setup camera
        bounds = calculate_scene_bounds(mesh_polydata, 
                                       pred_trajectories[:max_trajectories], 
                                       gt_trajectories[:max_trajectories])
        setup_camera(renderer, bounds)
        
        # Create render window
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(1920, 1080)
        render_window.SetMultiSamples(8)  # Anti-aliasing
        render_window.OffScreenRenderingOn()
        
        # Render
        render_window.Render()
        
        # Save to high-quality PNG
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(render_window)
        window_to_image.SetScale(2)  # High resolution
        window_to_image.SetInputBufferTypeToRGB()
        window_to_image.ReadFrontBufferOff()
        window_to_image.Update()
        
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(output_path)
        writer.SetInputConnection(window_to_image.GetOutputPort())
        writer.Write()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")
        return False

# ===================== Main Function =====================
def main():
    parser = argparse.ArgumentParser(description='FoldPath VTK Visualizer')
    
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
        args.output_dir = os.path.join(args.pred_dir, "visualizations")
    
    logger.info("FoldPath VTK Visualizer")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Validate directories
    if not os.path.isdir(args.pred_dir):
        logger.error(f"Predictions directory does not exist: {args.pred_dir}")
        return
    
    if not os.path.isdir(args.normalized_root):
        logger.error(f"Normalized root does not exist: {args.normalized_root}")
        return
    
    # Load prediction data
    pred_file = os.path.join(args.pred_dir, "all_predictions.npy")
    if not os.path.exists(pred_file):
        logger.error(f"all_predictions.npy not found: {pred_file}")
        return
    
    data = load_predictions(pred_file)
    if not data:
        logger.error("Failed to load prediction data")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each sample
    num_samples = min(len(args.sample_dirs), args.max_samples)
    
    for sample_idx in range(num_samples):
        sample_dir = args.sample_dirs[sample_idx]
        logger.info(f"Processing sample: {sample_dir}")
        
        # Load normalization parameters
        params_path = os.path.join(args.normalized_root, sample_dir, "norm_params.json")
        if not os.path.exists(params_path):
            norm_params = {'center': [0.0, 0.0, 0.0], 'scale': 1.0}
        else:
            with open(params_path, 'r') as f:
                norm_params = json.load(f)
        
        # Load OBJ mesh
        mesh_path = os.path.join(args.normalized_root, sample_dir, f"{sample_dir}_norm.obj")
        if not os.path.exists(mesh_path):
            mesh_path = os.path.join(args.normalized_root, sample_dir, f"{sample_dir}.obj")
        
        mesh_polydata = None
        if os.path.exists(mesh_path):
            mesh_polydata = load_obj_mesh(mesh_path, norm_params)
        
        # Extract trajectories
        pred_trajectories = extract_trajectories(
            data, sample_idx, norm_params, 'pred', args.max_trajectories
        )
        
        gt_trajectories = extract_trajectories(
            data, sample_idx, norm_params, 'gt', args.max_trajectories
        )
        
        # Create visualization
        if pred_trajectories or gt_trajectories:
            output_path = os.path.join(args.output_dir, f"{sample_dir}_pred_vs_gt.png")
            
            success = create_visualization(
                mesh_polydata=mesh_polydata,
                pred_trajectories=pred_trajectories,
                gt_trajectories=gt_trajectories,
                sample_name=sample_dir,
                output_path=output_path,
                max_trajectories=args.max_trajectories
            )
            
            if success:
                logger.info(f"  Saved: {output_path}")
            else:
                logger.error(f"  Failed to create visualization")
        else:
            logger.warning(f"  No valid trajectories found")
    
    logger.info("Visualization completed!")

if __name__ == '__main__':
    main()