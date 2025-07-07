import torch
import os
import random
from pathlib import Path

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
    
    # Load time series data and labels
    data_batch = []
    label_batch = []
    
    for data_file in selected_files:
        # Load time series tensor with weights_only=True to suppress warnings
        time_series_tensor = torch.load(data_file, weights_only=True)
        # Ensure tensor is on CPU
        time_series_tensor = time_series_tensor.cpu()
        data_batch.append(time_series_tensor)
        
        # Load corresponding label
        label_file = labels_dir / f"{data_file.stem}.pt"
        if label_file.exists():
            label_tensor = torch.load(label_file, weights_only=True)
            # Ensure tensor is on CPU
            label_tensor = label_tensor.cpu()
            label_batch.append(label_tensor)
        else:
            # Fallback random label if missing
            label_batch.append(torch.randn(1))
    
    # Stack tensors into batches
    # Handle variable lengths by padding or truncating
    max_length = max(tensor.shape[-1] for tensor in data_batch)
    padded_data = []
    
    for tensor in data_batch:
        if tensor.shape[-1] < max_length:
            # Pad with zeros
            padded = torch.zeros(tensor.shape[:-1] + (max_length,))
            padded[..., :tensor.shape[-1]] = tensor
            padded_data.append(padded)
        else:
            padded_data.append(tensor)
    
    data = torch.stack(padded_data)
    labels = torch.stack(label_batch).squeeze()  # Remove extra dimension if present
    
    print(f"  - Loaded real time series batch. Shape: {data.shape}")
    print(f"  - Labels shape: {labels.shape}")
    
    return data, labels
