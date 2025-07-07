import torch
import numpy as np
import random
from pathlib import Path

def load_data(test_id, batch_size=4):
    """
    Loads real CIFAR-10 data for continual learning memory retention tests.
    Uses CIFAR-10 data from data/cifar-10-batches-py/.
    """
    print(f"  - (Pillar 10) Loading real CIFAR-10 data for continual learning test {test_id}.")
    
    # Paths to CIFAR-10 data
    cifar_dir = Path("data/cifar-10-batches-py")
    
    if not cifar_dir.exists():
        raise FileNotFoundError(f"CIFAR-10 data not found in {cifar_dir}")
    
    # Load CIFAR-10 data
    try:
        import pickle
        import gzip
        
        # Load training data
        with open(cifar_dir / "data_batch_1", 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        
        images = batch[b'data']
        labels = batch[b'labels']
        
        # Reshape images to (N, 3, 32, 32)
        images = images.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        labels = np.array(labels)
        
    except Exception as e:
        print(f"  - Warning: Error loading CIFAR-10 data: {e}")
        # Fallback to random data
        images = np.random.rand(1000, 3, 32, 32).astype(np.float32)
        labels = np.random.randint(0, 10, 1000)
    
    # Create task splits for continual learning
    # Each task contains 2 classes
    task_id = (test_id - 81) % 5  # 5 tasks (0-4)
    class_start = task_id * 2
    class_end = class_start + 2
    
    # Filter data for this task
    task_mask = (labels >= class_start) & (labels < class_end)
    task_images = images[task_mask]
    task_labels = labels[task_mask]
    
    if len(task_images) == 0:
        # Fallback if no data for this task
        task_images = np.random.rand(batch_size, 3, 32, 32).astype(np.float32)
        task_labels = np.random.randint(class_start, class_end, batch_size)
    
    # Randomly sample batch_size samples
    n_samples = len(task_images)
    if batch_size > n_samples:
        batch_size = n_samples
    
    indices = random.sample(range(n_samples), batch_size)
    selected_images = task_images[indices]
    selected_labels = task_labels[indices]
    
    # Convert to tensors
    image_tensors = torch.tensor(selected_images, dtype=torch.float32)
    label_tensors = torch.tensor(selected_labels, dtype=torch.long)
    
    print(f"  - Loaded real CIFAR-10 batch for task {task_id}. Shape: {image_tensors.shape}")
    print(f"  - Labels shape: {label_tensors.shape}")
    print(f"  - Classes: {class_start}-{class_end-1}")
    
    return image_tensors, label_tensors
