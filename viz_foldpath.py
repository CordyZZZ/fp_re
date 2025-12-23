#!/usr/bin/env python3
"""
FoldPath Inference 3D Visualizer
Interactive 3D visualization of predicted folding paths on object meshes.
Supports rotation, zoom, pan, and multiple viewing angles.

Usage:
    python visualize_foldpath_inference.py --inference_file predictions.npy \
           --data_root ./dataset --sample_dir sample_name
"""

import argparse
import logging
import os
import numpy as np
import vtk
from typing import List, Optional, Dict, Tuple
import matplotlib.pyplot as plt
import datetime

# ===================== Configuration =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== OBJ Loading Functions =====================
def load_obj_vertices(obj_path: str) -> np.ndarray:
    """
    Load vertices from OBJ file.
    
    Args:
        obj_path: Path to OBJ file
        
    Returns:
        numpy array of vertices (N, 3)
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
        
        if len(vertices) == 0:
            logger.warning(f"No vertices found in {obj_path}")
            return np.zeros((0, 3))
        
        return np.array(vertices, dtype=np.float32)
        
    except Exception as e:
        logger.error(f"Failed to load OBJ vertices: {e}")
        return np.zeros((0, 3))


def calculate_normalization_params(vertices: np.ndarray) -> Dict:
    """
    Calculate per-mesh normalization parameters (center and scale).
    
    Args:
        vertices: Mesh vertices (N, 3)
        
    Returns:
        Dictionary with center and scale parameters
    """
    if len(vertices) == 0:
        return {'center': [0.0, 0.0, 0.0], 'scale': 1.0}
    
    # Calculate center
    center = vertices.mean(axis=0)
    
    # Calculate max distance for unit sphere scaling
    centered = vertices - center
    max_dist = np.linalg.norm(centered, axis=1).max()
    
    if max_dist > 0:
        scale = 1.0 / max_dist
    else:
        scale = 1.0
    
    logger.info(f"Calculated normalization: center={center}, max_dist={max_dist:.3f}, scale={scale:.6f}")
    
    return {
        'center': center.tolist(),
        'max_dist': float(max_dist),
        'scale': float(scale)
    }


def load_obj_mesh(obj_path: str, norm_params: Dict) -> Optional[vtk.vtkPolyData]:
    """
    Load OBJ mesh and apply denormalization if needed.
    
    Args:
        obj_path: Path to OBJ file
        norm_params: Normalization parameters dictionary
        
    Returns:
        VTK PolyData object or None if failed
    """
    try:
        # Load OBJ file using VTK reader
        obj_reader = vtk.vtkOBJReader()
        obj_reader.SetFileName(obj_path)
        obj_reader.Update()
        
        polydata = obj_reader.GetOutput()
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            return None
        
        points = polydata.GetPoints()
        n_points = points.GetNumberOfPoints()
        
        # Extract vertices to numpy array
        vertices_np = np.zeros((n_points, 3), dtype=np.float32)
        for i in range(n_points):
            vertices_np[i] = points.GetPoint(i)
        
        center = np.array(norm_params.get('center', [0.0, 0.0, 0.0]))
        scale = float(norm_params.get('scale', 1.0))
        
        # Check if mesh is normalized and needs denormalization
        avg_vertex_range = np.abs(vertices_np).max()
        obj_size = os.path.getsize(obj_path)
        
        if avg_vertex_range < 2.0 and obj_size < 1000000:
            logger.info("Mesh appears to be normalized, applying denormalization...")
            vertices_denorm = vertices_np.copy()
            vertices_denorm[:, 0] = vertices_denorm[:, 0] / scale + center[0]
            vertices_denorm[:, 1] = vertices_denorm[:, 1] / scale + center[1]
            vertices_denorm[:, 2] = vertices_denorm[:, 2] / scale + center[2]
        else:
            logger.info("Mesh appears to be in original coordinates, keeping as-is")
            vertices_denorm = vertices_np
        
        # Update VTK points with (possibly) denormalized vertices
        new_points = vtk.vtkPoints()
        for i in range(len(vertices_denorm)):
            new_points.InsertNextPoint(vertices_denorm[i])
        polydata.SetPoints(new_points)
        polydata.Modified()
        
        logger.info(f"Mesh: {n_points} vertices")
        logger.info(f"Bounds: X[{vertices_denorm[:,0].min():.1f}, {vertices_denorm[:,0].max():.1f}], "
                   f"Y[{vertices_denorm[:,1].min():.1f}, {vertices_denorm[:,1].max():.1f}], "
                   f"Z[{vertices_denorm[:,2].min():.1f}, {vertices_denorm[:,2].max():.1f}]")
        
        return polydata
        
    except Exception as e:
        logger.error(f"Failed to load OBJ mesh: {e}")
        return None


# ===================== Actor Creation Functions =====================
def create_mesh_actor(polydata: vtk.vtkPolyData) -> vtk.vtkActor:
    """
    Create VTK actor for gray object mesh.
    
    Args:
        polydata: VTK PolyData object
        
    Returns:
        VTK Actor for the mesh
    """
    try:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.85, 0.85, 0.85)  # Light gray
        actor.GetProperty().SetOpacity(0.85)
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


def create_trajectory_actor(trajectory: np.ndarray, color: Tuple, 
                           line_width: float = 10.0, opacity: float = 1.0) -> Optional[vtk.vtkActor]:
    """
    Create VTK actor for trajectory line.
    
    Args:
        trajectory: Trajectory points (N, 3)
        color: RGB color tuple
        line_width: Width of trajectory line
        opacity: Opacity of trajectory (0.0-1.0)
        
    Returns:
        VTK Actor for trajectory line or None if failed
    """
    try:
        if trajectory is None or len(trajectory) < 2:
            return None
        
        # Create points for trajectory
        points = vtk.vtkPoints()
        for pt in trajectory:
            points.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))
        
        # Create polyline
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
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetLighting(False)  # Disable lighting for brighter colors
        
        return actor
    except Exception:
        return None


def create_path_markers(trajectory: np.ndarray, color: Tuple, 
                       point_size: float = 20.0, opacity: float = 1.0) -> Optional[vtk.vtkActor]:
    """
    Create VTK actor for trajectory start and end markers.
    
    Args:
        trajectory: Trajectory points (N, 3)
        color: RGB color tuple
        point_size: Size of marker points
        opacity: Opacity of markers (0.0-1.0)
        
    Returns:
        VTK Actor for markers or None if failed
    """
    try:
        if trajectory is None or len(trajectory) < 2:
            return None
        
        # Start and end points
        start_point = trajectory[0]
        end_point = trajectory[-1]
        
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        
        # Start point
        idx1 = points.InsertNextPoint(float(start_point[0]), float(start_point[1]), float(start_point[2]))
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(idx1)
        
        # End point
        idx2 = points.InsertNextPoint(float(end_point[0]), float(end_point[1]), float(end_point[2]))
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
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetLighting(False)
        
        return actor
    except Exception:
        return None


def get_colormap_colors(colormap_name: str = 'tab10', num_colors: int = 10) -> List[Tuple[float, float, float]]:
    """
    Get colors from matplotlib colormap.
    
    Args:
        colormap_name: Name of matplotlib colormap
        num_colors: Number of colors to extract
        
    Returns:
        List of RGB color tuples
    """
    cmap = plt.colormaps[colormap_name] if colormap_name in plt.colormaps else plt.colormaps.get_cmap(colormap_name)
    
    colors = []
    for i in range(num_colors):
        rgba = cmap(i / max(num_colors - 1, 1))
        colors.append(rgba[:3])    
    return colors


# ===================== Data Processing =====================
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
        
        if 'predictions' not in data:
            logger.error("Invalid inference format - no 'predictions' key")
            return None
        
        return data['predictions']
        
    except Exception as e:
        logger.error(f"Failed to load inference results: {e}")
        return None


def denormalize_inference_trajectory(trajectory: np.ndarray, norm_params: Dict) -> np.ndarray:
    """
    Denormalize inference trajectory from normalized space to original object space.
    
    Args:
        trajectory: Normalized trajectory points (N, 3) in [-1, 1] range
        norm_params: Normalization parameters dictionary
        
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
        
        # Denormalization formula: v_original = v_normalized / scale + center
        for i in range(len(denorm_trajectory)):
            for j in range(3):
                denorm_trajectory[i, j] = denorm_trajectory[i, j] / scale + center[j]
        
        return denorm_trajectory
        
    except Exception as e:
        logger.error(f"Failed to denormalize trajectory: {e}")
        return trajectory


# ===================== Interactive Visualization =====================
class InteractiveVisualizer:
    """
    Interactive 3D visualizer for FoldPath predictions.
    
    Features:
    - Mouse drag to rotate
    - Scroll to zoom
    - Right-click drag to pan
    - Keyboard shortcuts for different views
    - Screenshot saving
    """
    
    def __init__(self, mesh_polydata, trajectories, path_info, max_trajectories=6):
        """
        Initialize visualizer.
        
        Args:
            mesh_polydata: VTK PolyData of object mesh
            trajectories: List of denormalized trajectory arrays
            path_info: List of trajectory information dictionaries
            max_trajectories: Maximum number of trajectories to display
        """
        self.mesh_polydata = mesh_polydata
        self.trajectories = trajectories
        self.path_info = path_info
        self.max_trajectories = max_trajectories
        
        # Create renderer with light gray background
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.95, 0.95, 0.95)
        
        # Setup lighting and scene
        self.setup_lighting()
        self.setup_scene()
        
        # Create render window
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1200, 800)
        self.render_window.SetWindowName("FoldPath 3D Visualizer - Drag to rotate, Scroll to zoom")
        
        # Create interactor
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        
        # Setup interaction
        self.setup_interaction()
        
    
    def setup_lighting(self):
        """Setup scene lighting for better 3D perception."""
        # Main light from top-right
        light1 = vtk.vtkLight()
        light1.SetPosition(10, 10, 10)
        light1.SetFocalPoint(0, 0, 0)
        light1.SetIntensity(1.0)
        self.renderer.AddLight(light1)
        
        # Fill light from bottom-left
        light2 = vtk.vtkLight()
        light2.SetPosition(-10, -10, 10)
        light2.SetFocalPoint(0, 0, 0)
        light2.SetIntensity(0.5)
        self.renderer.AddLight(light2)
    
    
    def setup_scene(self):
        """Setup 3D scene with mesh and trajectories."""
        # Add gray object mesh
        if self.mesh_polydata:
            mesh_actor = create_mesh_actor(self.mesh_polydata)
            if mesh_actor:
                self.renderer.AddActor(mesh_actor)
                logger.info("Added gray mesh")
        
        # Get colors for trajectories
        colors = get_colormap_colors('tab10', self.max_trajectories)
        
        # Add colored trajectories
        for i, (traj, info) in enumerate(zip(self.trajectories[:self.max_trajectories], 
                                           self.path_info[:self.max_trajectories])):
            if traj is not None and len(traj) > 1:
                color = colors[i % len(colors)]
                
                # Add trajectory line
                line_actor = create_trajectory_actor(traj, color, line_width=12.0)
                if line_actor:
                    self.renderer.AddActor(line_actor)
                
                # Add start and end markers
                markers_actor = create_path_markers(traj, color, point_size=25.0)
                if markers_actor:
                    self.renderer.AddActor(markers_actor)
                
                logger.info(f"  Path {i}: conf={info['confidence']:.3f}, length={info['length']:.2f}")
        
        # Reset camera to fit entire scene
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().Zoom(1.1)  # Zoom out slightly
    
    
    def setup_interaction(self):
        """Setup mouse and keyboard interaction."""
        # Use trackball camera for intuitive rotation/zoom/pan
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        # Add keyboard shortcuts
        self.add_keyboard_shortcuts()
    
    
    def add_keyboard_shortcuts(self):
        """Add keyboard shortcuts for view control."""
        def key_press(obj, event):
            key = obj.GetKeySym()
            camera = self.renderer.GetActiveCamera()
            
            if key == 'r' or key == 'R':
                # Reset view to default
                self.renderer.ResetCamera()
                camera.Zoom(1.1)
                self.render_window.Render()
                logger.info("View reset")
            
            elif key == 'w' or key == 'W':
                # Toggle wireframe mode
                for actor in self.renderer.GetActors():
                    if actor.GetProperty().GetRepresentation() == 0:  # Surface
                        actor.GetProperty().SetRepresentationToWireframe()
                    else:
                        actor.GetProperty().SetRepresentationToSurface()
                self.render_window.Render()
                logger.info("Toggled wireframe mode")
            
            elif key == 's' or key == 'S':
                # Save screenshot
                self.save_screenshot()
            
            elif key == '1':
                # Top view (X-Y plane)
                camera.SetPosition(0, 0, 10)
                camera.SetFocalPoint(0, 0, 0)
                camera.SetViewUp(0, 1, 0)
                self.render_window.Render()
                logger.info("Switched to top view")
            
            elif key == '2':
                # Front view (X-Z plane)
                camera.SetPosition(0, 10, 0)
                camera.SetFocalPoint(0, 0, 0)
                camera.SetViewUp(0, 0, 1)
                self.render_window.Render()
                logger.info("Switched to front view")
            
            elif key == '3':
                # Side view (Y-Z plane)
                camera.SetPosition(10, 0, 0)
                camera.SetFocalPoint(0, 0, 0)
                camera.SetViewUp(0, 0, 1)
                self.render_window.Render()
                logger.info("Switched to side view")
            
            elif key == 'q' or key == 'Escape':
                # Quit application
                self.interactor.ExitCallback()
        
        self.interactor.AddObserver("KeyPressEvent", key_press)
    
    
    def save_screenshot(self, filename: str = None):
        """
        Save screenshot of current view.
        
        Args:
            filename: Output filename (optional, auto-generated if None)
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"foldpath_screenshot_{timestamp}.png"
        
        # Capture window content
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(self.render_window)
        window_to_image.SetScale(2)  # High resolution
        window_to_image.SetInputBufferTypeToRGB()
        window_to_image.ReadFrontBufferOff()  # Read from back buffer for consistency
        window_to_image.Update()
        
        # Save as PNG
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(window_to_image.GetOutputPort())
        writer.Write()
        
        logger.info(f"Screenshot saved: {filename}")
    
    
    def run(self):
        """Run interactive visualization."""
        # Display controls information
        print("\n" + "="*60)
        print("FoldPath Interactive 3D Visualizer")
        print("="*60)
        print("Controls:")
        print("  Mouse Drag:      Rotate view")
        print("  Scroll Wheel:    Zoom in/out")
        print("  Right-click Drag: Pan view")
        print("\nKeyboard Shortcuts:")
        print("  R: Reset view")
        print("  W: Toggle wireframe mode")
        print("  S: Save screenshot")
        print("  1: Top view (X-Y)")
        print("  2: Front view (X-Z)")
        print("  3: Side view (Y-Z)")
        print("  Q/ESC: Exit")
        print("="*60 + "\n")
        
        # Start interactive session
        self.interactor.Initialize()
        self.render_window.Render()
        self.interactor.Start()


# ===================== Main Function =====================
def main():
    """Main function for FoldPath inference visualization."""
    parser = argparse.ArgumentParser(description='FoldPath Interactive 3D Visualizer')
    
    parser.add_argument('--inference_file', type=str, required=True,
                       help='Inference output file (e.g., predictions.npy)')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing sample folders')
    parser.add_argument('--sample_dir', type=str, required=True,
                       help='Sample directory name')
    parser.add_argument('--max_trajectories', type=int, default=6,
                       help='Maximum number of trajectories to display')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("FoldPath Interactive 3D Visualizer")
    logger.info("="*60)
    logger.info(f"Sample: {args.sample_dir}")
    
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
    
    # Load OBJ file and calculate normalization parameters
    obj_path = os.path.join(args.data_root, args.sample_dir, f"{args.sample_dir}.obj")
    if not os.path.exists(obj_path):
        logger.error(f"OBJ file not found: {obj_path}")
        return
    
    # Calculate normalization parameters from original mesh
    logger.info(f"Calculating normalization parameters from {obj_path}")
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
    
    # Process trajectories: denormalize to original object space
    valid_paths = []
    for i, path in enumerate(paths[:args.max_trajectories]):
        if path is not None and len(path) > 1:
            # Ensure proper shape
            if path.ndim == 1:
                path = path.reshape(-1, 3)
            
            logger.info(f"Path {i}: original range [{path.min():.3f}, {path.max():.3f}]")
            
            # Denormalize trajectory
            denorm_path = denormalize_inference_trajectory(path, norm_params)
            
            if denorm_path is not None:
                valid_paths.append(denorm_path)
                logger.info(f"  denormalized range [{denorm_path.min():.1f}, {denorm_path.max():.1f}]")
    
    logger.info(f"Processed {len(valid_paths)} trajectories")
    
    # Create and run interactive visualizer
    visualizer = InteractiveVisualizer(
        mesh_polydata=mesh_polydata,
        trajectories=valid_paths,
        path_info=path_info[:args.max_trajectories],
        max_trajectories=args.max_trajectories
    )
    
    visualizer.run()


if __name__ == '__main__':
    main()