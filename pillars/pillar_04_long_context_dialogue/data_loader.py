import numpy as np
import random

def load_data(test_id):
    """
    Mocks the data loading process for a given test in Pillar 4.
    This now returns a list of strings representing a batch of dialogues.
    """
    print(f"  - (Pillar 4) Loading mock data for test {test_id}.")
    
    batch_size = 4
    mock_dialogues = [
        f"Character A: Hello, how are you? Character B: I am fine. This is dialogue number {i} for test {test_id}."
        for i in range(batch_size)
    ]
    
    # For a coherence task, the ground truth is whether the dialogue is coherent (1) or not (0).
    ground_truth_labels = np.random.randint(0, 2, size=batch_size)
    
    return mock_dialogues, ground_truth_labels
