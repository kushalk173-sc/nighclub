import torch

def load_data(test_id):
    """
    Mocks the data loading process for a given test in Pillar 2 (Vision).
    This now returns a realistic tensor for a batch of images.
    """
    print(f"  - (Pillar 2) Loading mock data for test {test_id}.")
    
    # Simulate a batch of 4 images of size 3x224x224 (C, H, W)
    batch_size = 4
    mock_images = torch.randn(batch_size, 3, 224, 224)
    
    # Simulate corresponding ground-truth labels for the batch
    ground_truth_labels = torch.randint(0, 1000, (batch_size,))
    
    return mock_images, ground_truth_labels
