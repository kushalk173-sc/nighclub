from sklearn.metrics import f1_score
import torch
import numpy as np

def evaluate(prediction_logits, ground_truth_labels, metric="f1"):
    """
    Calculates the F1 score for a batch of binary classification predictions.
    
    Args:
        prediction_logits (torch.Tensor): The output from the model (batch_size, num_classes).
        ground_truth_labels (np.array): The true labels.
        metric (str): The metric to compute. Currently only 'f1' is supported.
    """
    if metric == "f1":
        print(f"  - (Pillar 4) Evaluating prediction using '{metric}'.")
        
        # Get the predicted class (0 or 1) by finding the index of the max logit
        predicted_labels = torch.argmax(prediction_logits, dim=1).cpu().numpy()
        
        # Calculate the F1 score
        score = f1_score(ground_truth_labels, predicted_labels, average='binary', zero_division=0.0)
        return score
    else:
        print(f"  - Metric '{metric}' not implemented. Returning 0.0.")
        return 0.0
