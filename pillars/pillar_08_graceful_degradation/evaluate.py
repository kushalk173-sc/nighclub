import torch
import numpy as np
from sklearn.metrics import mean_absolute_error

def evaluate(prediction, ground_truth, metric="accuracy"):
    """
    Evaluates prediction for Pillar 8 (Graceful Degradation).
    """
    if metric == "accuracy":
        print(f"  - (Pillar 8) Evaluating prediction using '{metric}'.")
        
        # Handle different prediction formats
        if isinstance(prediction, torch.Tensor):
            # If prediction is a tensor of logits, get the predicted class
            if prediction.dim() > 1:
                predicted_labels = torch.argmax(prediction, dim=-1)
            else:
                predicted_labels = prediction
        elif isinstance(prediction, list):
            predicted_labels = torch.tensor(prediction)
        else:
            print(f"  - Warning: Unexpected prediction type: {type(prediction)}")
            return 0.0
        
        # Ensure ground truth is a tensor
        if not isinstance(ground_truth, torch.Tensor):
            ground_truth = torch.tensor(ground_truth)
        
        # Calculate accuracy
        correct = (predicted_labels == ground_truth).sum().item()
        total = len(ground_truth)
        accuracy = (correct / total) * 100.0 if total > 0 else 0.0
        
        print(f"  - Correct: {correct}/{total}, Accuracy: {accuracy:.2f}%")
        return accuracy
        
    elif metric == "mae":
        print(f"  - (Pillar 8) Evaluating prediction using '{metric}'.")
        pred_np = prediction.cpu().numpy()
        gt_np = ground_truth.cpu().numpy()
        score = mean_absolute_error(gt_np, pred_np)
        return score
    else:
        print(f"  - Metric '{metric}' not implemented. Returning 0.0.")
        return 0.0
