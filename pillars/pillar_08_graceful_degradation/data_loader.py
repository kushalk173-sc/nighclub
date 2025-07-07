import torch
import os
import random
from pathlib import Path

def load_data(test_id, batch_size=4):
    """
    Loads real image data and corresponding labels from processed data.
    Uses the processed data from data/pillar_8_processed/ or falls back to pillar_2_processed/.
    """
    print(f"  - (Pillar 8) Loading real image data for test {test_id}.")
    
    # Try pillar 8 data first
    images_dir = Path("data/pillar_8_processed/images")
    labels_dir = Path("data/pillar_8_processed/labels")
    
    # If pillar 8 data doesn't exist, fall back to pillar 2 data
    if not images_dir.exists():
        print(f"  - Pillar 8 data not found, using pillar 2 data as fallback.")
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
        if label_file.exists():
            label_tensor = torch.load(label_file, weights_only=True)
            # Ensure tensor is on CPU
            label_tensor = label_tensor.cpu()
            label_batch.append(label_tensor)
        else:
            # Fallback random label if missing
            label_batch.append(torch.randint(0, 1000, (1,)))
    
    # Stack tensors into batches
    images = torch.stack(image_batch)
    labels = torch.stack(label_batch).squeeze()  # Remove extra dimension if present
    
    print(f"  - Loaded real image batch. Shape: {images.shape}")
    print(f"  - Labels shape: {labels.shape}")
    
    return images, labels

def apply_degradation(images, test_id):
    """
    Applies different types of degradation to the images based on test_id.
    """
    degraded = images.clone()
    
    # Different degradation types for different tests
    if test_id == 71:  # Noise degradation
        noise = torch.randn_like(degraded) * 0.1
        degraded = degraded + noise
    elif test_id == 72:  # Blur degradation
        # Simple blur by averaging neighboring pixels
        # Handle 3-channel images properly
        degraded_blurred = degraded.clone()
        for c in range(degraded.shape[1]):  # Apply to each channel
            kernel = torch.ones(1, 1, 3, 3) / 9
            channel_data = degraded[:, c:c+1, :, :]
            blurred_channel = torch.nn.functional.conv2d(channel_data, kernel, padding=1)
            degraded_blurred[:, c:c+1, :, :] = blurred_channel
        degraded = degraded_blurred
    elif test_id == 73:  # Brightness degradation
        degraded = degraded * 0.5  # Reduce brightness
    elif test_id == 74:  # Contrast degradation
        degraded = (degraded - degraded.mean()) * 0.5 + degraded.mean()
    elif test_id == 75:  # Resolution degradation
        # Downsample and upsample
        degraded = torch.nn.functional.interpolate(degraded, scale_factor=0.5, mode='bilinear')
        degraded = torch.nn.functional.interpolate(degraded, scale_factor=2.0, mode='bilinear')
    else:
        # Default: slight noise
        noise = torch.randn_like(degraded) * 0.05
        degraded = degraded + noise
    
    return degraded
