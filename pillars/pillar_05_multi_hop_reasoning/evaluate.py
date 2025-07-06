from sklearn.metrics import accuracy_score
import torch
import numpy as np

def evaluate(prediction_logits, ground_truth_labels, metric="accuracy"):
    """
    Calculates the accuracy score for a batch of binary classification predictions.
    This serves as a proxy for "Exact-Match" in our baseline.
    
    Args:
        prediction_logits (torch.Tensor): The output from the model (batch_size, num_classes).
        ground_truth_labels (np.array): The true labels.
        metric (str): The metric to compute.
    """
    if metric == "accuracy":
        print(f"  - (Pillar 5) Evaluating prediction using '{metric}'.")
        
        predicted_labels = torch.argmax(prediction_logits, dim=1).cpu().numpy()
        
        score = accuracy_score(ground_truth_labels, predicted_labels)
        return score
    else:
        print(f"  - Metric '{metric}' not implemented. Returning 0.0.")
        return 0.0
