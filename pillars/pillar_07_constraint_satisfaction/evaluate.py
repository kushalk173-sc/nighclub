import torch
import numpy as np
from sklearn.metrics import mean_absolute_error

def evaluate(prediction, ground_truth, metric="mae"):
    """
    Evaluates prediction for Pillar 7 (Constraint Satisfaction).
    """
    if metric == "mae":
        print(f"  - (Pillar 7) Evaluating prediction using '{metric}'.")
        
        # Handle 3D tensors (batch, 9, 9) by flattening
        if prediction.dim() == 3:
            pred_np = prediction.cpu().numpy().reshape(prediction.shape[0], -1)
        else:
            pred_np = prediction.cpu().numpy()
            
        if ground_truth.dim() == 3:
            gt_np = ground_truth.cpu().numpy().reshape(ground_truth.shape[0], -1)
        else:
            gt_np = ground_truth.cpu().numpy()
        
        score = mean_absolute_error(gt_np, pred_np)
        return score
    else:
        print(f"  - Metric '{metric}' not implemented. Returning 0.0.")
        return 0.0
