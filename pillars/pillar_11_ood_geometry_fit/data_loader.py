import torch
import numpy as np
import random
from pathlib import Path

def load_data(test_id, batch_size=4):
    """
    Loads real geometry data from processed data.
    Uses the processed data from data/pillar_11_processed/.
    """
    print(f"  - (Pillar 11) Loading real geometry data for test {test_id}.")
    
    # Paths to processed data
    geometry_dir = Path("data/pillar_11_processed")
    labels_file = geometry_dir / "manifold_labels.npy"
    
    if not geometry_dir.exists() or not labels_file.exists():
        raise FileNotFoundError(f"Geometry data files not found in {geometry_dir}")
    
    # Load manifold labels
    manifold_labels = np.load(labels_file)
    
    # Get all geometry sample files
    geometry_files = list(geometry_dir.glob("geometry_sample_*.npy"))
    if not geometry_files:
        raise FileNotFoundError(f"No geometry sample files found in {geometry_dir}")
    
    # Randomly sample batch_size files
    selected_files = random.sample(geometry_files, min(batch_size, len(geometry_files)))
    
    # Load geometry data and corresponding labels
    geometry_batch = []
    label_batch = []
    
    for geometry_file in selected_files:
        # Extract sample index from filename
        sample_idx = int(geometry_file.stem.split('_')[-1])
        
        # Load geometry points
        points = np.load(geometry_file)
        geometry_batch.append(points)
        
        # Get corresponding label
        if sample_idx < len(manifold_labels):
            label_batch.append(manifold_labels[sample_idx])
        else:
            label_batch.append(0)  # Default label
    
    # Convert to tensors
    # Note: geometry data has variable shapes, so we'll pad to the largest size
    max_points = max(points.shape[0] for points in geometry_batch)
    max_dims = max(points.shape[1] for points in geometry_batch)
    
    padded_geometry = []
    for points in geometry_batch:
        # Pad to max size
        padded = np.zeros((max_points, max_dims))
        padded[:points.shape[0], :points.shape[1]] = points
        padded_geometry.append(padded)
    
    geometry_tensors = torch.tensor(padded_geometry, dtype=torch.float32)
    label_tensors = torch.tensor(label_batch, dtype=torch.long)
    
    print(f"  - Loaded real geometry batch. Shape: {geometry_tensors.shape}")
    print(f"  - Manifold labels shape: {label_tensors.shape}")
    
    return geometry_tensors, label_tensors
