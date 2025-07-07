import torch
import numpy as np
import random
from pathlib import Path

def load_data(test_id, batch_size=4):
    """
    Loads real sensor drift data from processed data.
    Uses the processed data from data/pillar_9_processed/.
    """
    print(f"  - (Pillar 9) Loading real sensor drift data for test {test_id}.")
    
    # Paths to processed data
    sensor_file = Path("data/pillar_9_processed/sensor_data.npy")
    labels_file = Path("data/pillar_9_processed/drift_labels.npy")
    
    if not sensor_file.exists() or not labels_file.exists():
        raise FileNotFoundError(f"Sensor drift data files not found in data/pillar_9_processed/")
    
    # Load sensor data and drift labels
    sensor_data = np.load(sensor_file)
    drift_labels = np.load(labels_file)
    
    # Randomly sample batch_size samples
    n_samples = len(sensor_data)
    if batch_size > n_samples:
        batch_size = n_samples
    
    indices = random.sample(range(n_samples), batch_size)
    selected_data = sensor_data[indices]
    selected_labels = drift_labels[indices]
    
    # Convert to tensors
    data_tensors = torch.tensor(selected_data, dtype=torch.float32)
    label_tensors = torch.tensor(selected_labels, dtype=torch.float32).unsqueeze(1)
    
    print(f"  - Loaded real sensor drift batch. Shape: {data_tensors.shape}")
    print(f"  - Drift labels shape: {label_tensors.shape}")
    
    return data_tensors, label_tensors
