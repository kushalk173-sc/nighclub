
import torch

def load_data(test_id):
    """
    Mocks data loading for Pillar 3. Returns a generic tensor.
    """
    print(f"  - (Pillar 3) Loading mock numeric data for test {test_id}.")
    batch_size = 4
    # Generic tensor of shape (batch, features)
    mock_data = torch.randn(batch_size, 10)
    # Generic regression target
    ground_truth = torch.randn(batch_size, 1)
    return mock_data, ground_truth
