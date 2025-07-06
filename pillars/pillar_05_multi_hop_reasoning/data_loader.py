import numpy as np
import random

def load_data(test_id):
    """
    Mocks the data loading process for a given test in Pillar 5.
    This returns a list of strings representing a batch of QA contexts.
    """
    print(f"  - (Pillar 5) Loading mock data for test {test_id}.")
    
    batch_size = 4
    mock_qa_contexts = [
        f"Context: The Eiffel Tower is in Paris. Paris is the capital of France. Question: Where is the Eiffel Tower located? This is QA context #{i}."
        for i in range(batch_size)
    ]
    
    # For a QA task, the ground truth could be whether the answer is correct (1) or not (0).
    ground_truth_labels = np.random.randint(0, 2, size=batch_size)
    
    return mock_qa_contexts, ground_truth_labels
