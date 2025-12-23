#!/usr/bin/env python3
"""
FoldPath Static Visualizer
Static visualization of predicted folding paths on object meshes.
Saves high-quality PNG images without interactive window.

Usage:
    python visualize_foldpath_static.py --inference_file predictions.npy \
           --data_root ./dataset --sample_dir sample_name --output_dir ./visualizations
"""

import argparse
import logging
import os
import numpy as np
import vtk
from typing import List, Optional, Dict, Tuple
import matplotlib.pyplot as plt

# ===================== Configuration =====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ===================== Data Loading Functions =====================
def load_obj_vertices(obj_path: str) -> np.ndarray:
    """
    Load vertices from OBJ file.
    
    Args:
        obj_path: Path to OBJ file
        
    Returns:
        numpy array of vertices (N, 3) or empty array if failed
    """
    vertices = []
    try:
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        vertices.append([x, y, z])
        
        return np.array(vertices, dtype=np.float32) if vertices else np.zeros((0, 3))
        
    except Exception as e:
        logger.error(f"Failed to load OBJ vertices: {e}")
        return np.zeros((0, 3))


def calculate_normalization_params(vertices: np.ndarray) -> Dict:
    """
    Calculate per-mesh normalization parameters.
    
    Args:
        vertices: Mesh vertices (N, 3)
        
    Returns:
        Dictionary with 'center' and 'scale' keys
    """
    if len(vertices) == 0:
        return {'center': [0.0, 0.0, 0.0], 'scale': 1.0}
    
    center = vertices.mean(axis=0)
    centered = vertices - center
    max_dist = np.linalg.norm(centered, axis=1).max()
    scale = 1.0 / max_dist if max_dist > 0 else 1.0
    
    logger.info(f"Normalization: center={center}, scale={scale:.6f}")
    
    return {'center': center.tolist(), 'scale': float(scale)}


def load_obj_mesh(obj_path: str, norm_params: Dict) -> Optional[vtk.vtkPolyData]:
    """
    Load OBJ mesh and apply denormalization if needed.
    
    Args:
        obj_path: Path to OBJ file
        norm_params: Normalization parameters
        
    Returns:
        VTK PolyData object or None if failed
    """
    try:
        obj_reader = vtk.vtkOBJReader()
        obj_reader.SetFileName(obj_path)
        obj_reader.Update()
        
        polydata = obj_reader.GetOutput()
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            return None
        
        points = polydata.GetPoints()
        n_points = points.GetNumberOfPoints()
        
        # Extract vertices
        vertices_np = np.zeros((n_points, 3), dtype=np.float32)
        for i in range(n_points):
            vertices_np[i] = points.GetPoint(i)
        
        center = np.array(norm_params.get('center', [0.0, 0.0, 0.0]))
        scale = float(norm_params.get('scale', 1.0))
        
        # Check if mesh is normalized (heuristic)
        avg_vertex_range = np.abs(vertices_np).max()
        
        if avg_vertex_range < 2.0:  # Likely normalized
            vertices_denorm = vertices_np.copy()
            vertices_denorm[:, 0] = vertices_denorm[:, 0] / scale + center[0]
            vertices_denorm[:, 1] = vertices_denorm[:, 1] / scale + center[1]
            vertices_denorm[:, 2] = vertices_denorm[:, 2] / scale + center[2]
        else:
            vertices_denorm = vertices_np
        
        # Update VTK points
        new_points = vtk.vtkPoints()
        for i in range(len(vertices_denorm)):
            new_points.InsertNextPoint(vertices_denorm[i])
        polydata.SetPoints(new_points)
        polydata.Modified()
        
        logger.info(f"Mesh loaded: {n_points} vertices")
        
        return polydata
        
    except Exception as e:
        logger.error(f"Failed to load OBJ mesh: {e}")
        return None


def load_inference_results(pred_file: str) -> Optional[Dict]:
    """
    Load inference results from numpy file.
    
    Args:
        pred_file: Path to predictions.npy file
        
    Returns:
        Dictionary with predictions or None if failed
    """
    try:
        data = np.load(pred_file, allow_pickle=True).item()
        return data.get('predictions') if 'predictions' in data else None
    except Exception as e:
        logger.error(f"Failed to load inference results: {e}")
        return None


def denormalize_trajectory(trajectory: np.ndarray, norm_params: Dict) -> np.ndarray:
    """
    Denormalize trajectory from normalized space to original object space.
    
    Args:
        trajectory: Normalized trajectory points (N, 3) in [-1, 1] range
        norm_params: Normalization parameters
        
    Returns:
        Denormalized trajectory in original object space
    """
    if trajectory is None or len(trajectory) == 0:
        return trajectory
    
    try:
        center = np.array(norm_params.get('center', [0.0, 0.0, 0.0]))
        scale = float(norm_params.get('scale', 1.0))
        
        if scale == 0:
            scale = 1.0
        
        denorm_trajectory = trajectory.copy()
        
        # Denormalization: v_original = v_normalized / scale + center
        for i in range(len(denorm_trajectory)):
            for j in range(3):
                denorm_trajectory[i, j] = denorm_trajectory[i, j] / scale + center[j]
        
        return denorm_trajectory
        
    except Exception as e:
        logger.error(f"Failed to denormalize trajectory: {e}")
        return trajectory


# ===================== VTK Actor Creation =====================
def create_mesh_actor(polydata: vtk.vtkPolyData) -> Optional[vtk.vtkActor]:
    """
    Create VTK actor for gray object mesh.
    
    Args:
        polydata: VTK PolyData object
        
    Returns:
        VTK Actor for mesh or None if failed
    """
    try:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.85, 0.85, 0.85)  # Light gray
        actor.GetProperty().SetOpacity(0.85)
        
        # Show edges for better depth perception
        actor.GetProperty().SetEdgeColor(0.5, 0.5, 0.5)
        actor.GetProperty().SetEdgeVisibility(True)
        actor.GetProperty().SetLineWidth(0.5)
        
        return actor
    except Exception:
        return None


def create_trajectory_actor(trajectory: np.ndarray, color: Tuple, 
                           line_width: float = 12.0) -> Optional[vtk.vtkActor]:
    """
    Create VTK actor for trajectory line.
    
    Args:
        trajectory: Trajectory points (N, 3)
        color: RGB color tuple
        line_width: Width of trajectory line
        
    Returns:
        VTK Actor for trajectory or None if failed
    """
    try:
        if len(trajectory) < 2:
            return None
        
        points = vtk.vtkPoints()
        for pt in trajectory:
            points.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))
        
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(trajectory))
        for i in range(len(trajectory)):
            polyline.GetPointIds().SetId(i, i)
        
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyline)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(cells)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(line_width)
        actor.GetProperty().SetLighting(False)
        
        return actor
    except Exception:
        return None


def create_path_markers(trajectory: np.ndarray, color: Tuple, 
                       point_size: float = 30.0) -> Optional[vtk.vtkActor]:
    """
    Create VTK actor for trajectory start and end markers.
    
    Args:
        trajectory: Trajectory points (N, 3)
        color: RGB color tuple
        point_size: Size of marker points
        
    Returns:
        VTK Actor for markers or None if failed
    """
    try:
        if len(trajectory) < 2:
            return None
        
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        
        # Start point
        idx1 = points.InsertNextPoint(float(trajectory[0, 0]), 
                                     float(trajectory[0, 1]), 
                                     float(trajectory[0, 2]))
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(idx1)
        
        # End point
        idx2 = points.InsertNextPoint(float(trajectory[-1, 0]), 
                                     float(trajectory[-1, 1]), 
                                     float(trajectory[-1, 2]))
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(idx2)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetVerts(vertices)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetPointSize(point_size)
        actor.GetProperty().SetLighting(False)
        
        return actor
    except Exception:
        return None


def get_colormap_colors(colormap_name: str = 'tab10', num_colors: int = 10) -> List[Tuple]:
    """
    Get colors from matplotlib colormap.
    
    Args:
        colormap_name: Name of matplotlib colormap
        num_colors: Number of colors to extract
        
    Returns:
        List of RGB color tuples
    """
    cmap = plt.colormaps[colormap_name]
    colors = []
    for i in range(num_colors):
        rgba = cmap(i / max(num_colors - 1, 1))
        colors.append(rgba[:3])
    return colors


# ===================== Main Visualization Function =====================
def create_visualization(mesh_polydata, trajectories, path_info, 
                        output_path: str, max_trajectories: int = 6) -> bool:
    """
    Create and save static visualization.
    
    Args:
        mesh_polydata: VTK PolyData of object mesh
        trajectories: List of denormalized trajectory arrays
        path_info: List of trajectory information dictionaries
        output_path: Path to save PNG image
        max_trajectories: Maximum number of trajectories to display
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create renderer with light gray background
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.95, 0.95, 0.95)
        
        # Add lighting
        light = vtk.vtkLight()
        light.SetPosition(10, 10, 10)
        light.SetFocalPoint(0, 0, 0)
        light.SetIntensity(1.0)
        renderer.AddLight(light)
        
        # Add gray mesh
        if mesh_polydata:
            mesh_actor = create_mesh_actor(mesh_polydata)
            if mesh_actor:
                renderer.AddActor(mesh_actor)
                logger.info("Added object mesh")
        
        # Add colored trajectories
        colors = get_colormap_colors('tab10', max_trajectories)
        
        for i, (traj, info) in enumerate(zip(trajectories[:max_trajectories], 
                                           path_info[:max_trajectories])):
            if len(traj) > 1:
                color = colors[i % len(colors)]
                
                # Add trajectory line
                line_actor = create_trajectory_actor(traj, color, line_width=15.0)
                if line_actor:
                    renderer.AddActor(line_actor)
                
                # Add start and end markers
                markers_actor = create_path_markers(traj, color, point_size=30.0)
                if markers_actor:
                    renderer.AddActor(markers_actor)
                
                logger.info(f"Path {i}: conf={info['confidence']:.3f}, length={info['length']:.2f}")
        
        # Setup camera
        renderer.ResetCamera()
        renderer.GetActiveCamera().Zoom(1.1)
        
        # Create render window
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(1600, 1200)
        render_window.SetMultiSamples(8)  # Anti-aliasing
        render_window.OffScreenRenderingOn()
        
        # Render scene
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
        
        logger.info(f"Saved visualization: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")
        return False


# ===================== Main Function =====================
def main():
    """Main function for static visualization."""
    parser = argparse.ArgumentParser(description='FoldPath Static Visualizer')
    
    parser.add_argument('--inference_file', type=str, required=True,
                       help='Inference output file (e.g., predictions.npy)')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing sample folders')
    parser.add_argument('--sample_dir', type=str, required=True,
                       help='Sample directory name')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for PNG images')
    parser.add_argument('--max_trajectories', type=int, default=6,
                       help='Maximum number of trajectories to display')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("FoldPath Static Visualizer")
    logger.info("=" * 60)
    
    # Load inference results
    if not os.path.exists(args.inference_file):
        logger.error(f"Inference file not found: {args.inference_file}")
        return
    
    predictions = load_inference_results(args.inference_file)
    if predictions is None:
        logger.error("Failed to load inference results")
        return
    
    sample_name = predictions.get('sample', args.sample_dir)
    paths = predictions.get('paths', [])
    path_info = predictions.get('path_info', [])
    
    logger.info(f"Loaded {len(paths)} predicted paths")
    
    # Load OBJ file and calculate normalization
    obj_path = os.path.join(args.data_root, args.sample_dir, f"{args.sample_dir}.obj")
    if not os.path.exists(obj_path):
        logger.error(f"OBJ file not found: {obj_path}")
        return
    
    # Calculate normalization parameters
    logger.info(f"Calculating normalization parameters...")
    vertices = load_obj_vertices(obj_path)
    if len(vertices) == 0:
        logger.error("Failed to load vertices from OBJ file")
        return
    
    norm_params = calculate_normalization_params(vertices)
    
    # Load and process mesh
    mesh_polydata = load_obj_mesh(obj_path, norm_params)
    if not mesh_polydata:
        logger.error("Failed to load mesh")
        return
    
    # Process trajectories
    valid_paths = []
    for i, path in enumerate(paths[:args.max_trajectories]):
        if path is not None and len(path) > 1:
            # Ensure proper shape
            if path.ndim == 1:
                path = path.reshape(-1, 3)
            
            # Denormalize trajectory
            denorm_path = denormalize_trajectory(path, norm_params)
            
            if denorm_path is not None:
                valid_paths.append(denorm_path)
                logger.info(f"Path {i}: denormalized to object space")
    
    logger.info(f"Processed {len(valid_paths)} trajectories")
    
    # Create and save visualization
    output_path = os.path.join(args.output_dir, f"{sample_name}_foldpath.png")
    
    success = create_visualization(
        mesh_polydata=mesh_polydata,
        trajectories=valid_paths,
        path_info=path_info[:args.max_trajectories],
        output_path=output_path,
        max_trajectories=args.max_trajectories
    )
    
    if success:
        logger.info("=" * 60)
        logger.info(f"Visualization saved to: {output_path}")
        logger.info("=" * 60)
    else:
        logger.error("Failed to create visualization")


if __name__ == '__main__':
    main()