import torch
import os
import random
from pathlib import Path
from utils.dev import get_device

def load_data(test_id, batch_size=4):
    """
    Loads real sensor drift data from processed data.
    Uses the processed data from data/pillar_9_processed/.
    """
    print(f"  - (Pillar 9) Loading real sensor drift data for test {test_id}.")
    
    # Paths to processed data
    data_dir = Path("data/pillar_9_processed/data")
    labels_dir = Path("data/pillar_9_processed/labels")
    
    # Get all available data files
    data_files = list(data_dir.glob("*.pt"))
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    # Randomly sample batch_size files
    selected_files = random.sample(data_files, min(batch_size, len(data_files)))
    
    # Load data tensors and labels
    data_batch = []
    label_batch = []
    
    for data_file in selected_files:
        # Load data tensor with weights_only=True to suppress warnings
        data_tensor = torch.load(data_file, weights_only=True)
        # Ensure tensor is on CPU initially
        data_tensor = data_tensor.cpu()
        data_batch.append(data_tensor)
        
        # Load corresponding label
        label_file = labels_dir / f"{data_file.stem}.pt"
        if not label_file.exists():
            raise FileNotFoundError(f"Label file {label_file} not found for data file {data_file}")
        
        label_tensor = torch.load(label_file, weights_only=True)
        # Ensure tensor is on CPU initially
        label_tensor = label_tensor.cpu()
        label_batch.append(label_tensor)
    
    # Stack tensors into batches
    data = torch.stack(data_batch)
    labels = torch.stack(label_batch).squeeze()  # Remove extra dimension if present
    
    # Move to the correct device
    device = get_device()
    data = data.to(device)
    labels = labels.to(device)
    
    print(f"  - Loaded real sensor drift batch. Shape: {data.shape}")
    print(f"  - Labels shape: {labels.shape}")
    
    return data, labels
