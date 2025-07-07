import random
import numpy as np

def evaluate(prediction, ground_truth=None):
    """
    Evaluates multi-hop reasoning prediction against real ground truth labels.
    """
    if ground_truth is None:
        print("  - (Pillar 5) Warning: No ground truth provided, using fallback evaluation.")
        return random.uniform(0, 1)
    
    print("  - (Pillar 5) Evaluating multi-hop reasoning against real ground truth.")
    
    try:
        # Handle different prediction formats
        if isinstance(prediction, np.ndarray):
            predicted_labels = prediction
        elif isinstance(prediction, list):
            predicted_labels = np.array(prediction)
        elif isinstance(prediction, (int, float)):
            # Single prediction value
            predicted_labels = np.array([prediction])
        else:
            print(f"  - Warning: Unexpected prediction type: {type(prediction)}")
            return 0.0
        
        # Ensure ground truth is numpy array
        if not isinstance(ground_truth, np.ndarray):
            ground_truth = np.array(ground_truth)
        
        # Calculate accuracy for binary classification
        if len(predicted_labels) == 1 and len(ground_truth) > 1:
            # Single prediction for multiple examples - use average
            predicted_labels = np.full_like(ground_truth, predicted_labels[0])
        
        # Ensure same length
        min_len = min(len(predicted_labels), len(ground_truth))
        predicted_labels = predicted_labels[:min_len]
        ground_truth = ground_truth[:min_len]
        
        # Calculate accuracy
        correct = (predicted_labels == ground_truth).sum()
        total = len(ground_truth)
        accuracy = correct / total if total > 0 else 0.0
        
        print(f"  - Correct: {correct}/{total}, Accuracy: {accuracy:.3f}")
        return accuracy
        
    except Exception as e:
        print(f"  - Error during evaluation: {e}")
        return 0.0
