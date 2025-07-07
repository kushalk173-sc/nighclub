import torch
import os
import random
from pathlib import Path
from utils.dev import get_device, to_device

def load_data(test_id, batch_size=4):
    """
    Loads real time series data and corresponding labels from processed data.
    Uses the processed data from data/pillar_3_processed/.
    """
    print(f"  - (Pillar 3) Loading real time series data for test {test_id}.")
    
    # Paths to processed data
    data_dir = Path("data/pillar_3_processed/data")
    labels_dir = Path("data/pillar_3_processed/labels")
    
    # Get all available data files
    data_files = list(data_dir.glob("*.pt"))
    if not data_files:
        raise FileNotFoundError(f"No time series data files found in {data_dir}")
    
    # Randomly sample batch_size files
    selected_files = random.sample(data_files, min(batch_size, len(data_files)))
    
    # Load time series tensors and labels
    data_batch = []
    label_batch = []
    
    for data_file in selected_files:
        # Load data tensor with weights_only=True to suppress warnings
        data_tensor = torch.load(data_file, weights_only=True)
        # Move to device immediately
        data_tensor = to_device(data_tensor)
        data_batch.append(data_tensor)
        
        # Load corresponding label
        label_file = labels_dir / f"{data_file.stem}.pt"
        if not label_file.exists():
            raise FileNotFoundError(f"Label file {label_file} not found for data file {data_file}")
        
        label_tensor = torch.load(label_file, weights_only=True)
        # Move to device immediately
        label_tensor = to_device(label_tensor)
        label_batch.append(label_tensor)
    
    # Stack tensors into batches
    data = torch.stack(data_batch)
    labels = torch.stack(label_batch).squeeze()  # Remove extra dimension if present
    
    # Normalize each time series to handle scaling issues
    # Compute mean and std for each sample in the batch
    data_mean = data.mean(dim=-1, keepdim=True)
    data_std = data.std(dim=-1, keepdim=True)
    # Avoid division by zero
    data_std = torch.clamp(data_std, min=1e-8)
    # Normalize data
    data = (data - data_mean) / data_std
    
    # Normalize labels using the same statistics
    labels = (labels - data_mean.squeeze(-1)) / data_std.squeeze(-1)
    
    # Move to the correct device
    data = to_device(data)
    labels = to_device(labels)
    
    print(f"  - Loaded real time series batch. Shape: {data.shape}")
    print(f"  - Labels shape: {labels.shape}")
    
    return data, labels
