import torch

def evaluate(prediction_logits, ground_truth_labels, metric="accuracy"):
    """
    Calculates the top-1 accuracy for a batch of predictions.
    
    Args:
        prediction_logits (torch.Tensor): The output from the model (batch_size, num_classes).
        ground_truth_labels (torch.Tensor): The true labels (batch_size).
        metric (str): The metric to compute. Currently only 'accuracy' is supported.
    """
    if metric == "accuracy":
        print(f"  - (Pillar 2) Evaluating mock prediction using '{metric}'.")
        # Get the index of the max logit as the predicted class
        predicted_labels = torch.argmax(prediction_logits, dim=1)
        
        # Compare predicted labels to ground truth
        correct_predictions = (predicted_labels == ground_truth_labels).sum().item()
        
        # Calculate accuracy
        accuracy = correct_predictions / len(ground_truth_labels)
        return accuracy
    else:
        print(f"  - Metric '{metric}' not implemented. Returning 0.0.")
        return 0.0
