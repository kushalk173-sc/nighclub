import torch
import os
import random
from pathlib import Path

def load_data(test_id, batch_size=4):
    """
    Loads real image data and corresponding labels from processed data.
    Uses the processed data from data/pillar_2_processed/.
    """
    print(f"  - (Pillar 2) Loading real image data for test {test_id}.")
    
    # Paths to processed data
    images_dir = Path("data/pillar_2_processed/images")
    labels_dir = Path("data/pillar_2_processed/labels")
    
    # Get all available image files
    image_files = list(images_dir.glob("*.pt"))
    if not image_files:
        raise FileNotFoundError(f"No image files found in {images_dir}")
    
    # Randomly sample batch_size files
    selected_files = random.sample(image_files, min(batch_size, len(image_files)))
    
    # Load image tensors and labels
    image_batch = []
    label_batch = []
    
    for image_file in selected_files:
        # Load image tensor with weights_only=True to suppress warnings
        image_tensor = torch.load(image_file, weights_only=True)
        # Ensure tensor is on CPU
        image_tensor = image_tensor.cpu()
        image_batch.append(image_tensor)
        
        # Load corresponding label
        label_file = labels_dir / f"{image_file.stem}.pt"
        if not label_file.exists():
            raise FileNotFoundError(f"Label file {label_file} not found for image file {image_file}")
        
        label_tensor = torch.load(label_file, weights_only=True)
        # Ensure tensor is on CPU
        label_tensor = label_tensor.cpu()
        label_batch.append(label_tensor)
    
    # Stack tensors into batches
    images = torch.stack(image_batch)
    labels = torch.stack(label_batch).squeeze()  # Remove extra dimension if present
    
    print(f"  - Loaded real image batch. Shape: {images.shape}")
    print(f"  - Labels shape: {labels.shape}")
    
    return images, labels
