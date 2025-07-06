
import torch
from sklearn.metrics import mean_absolute_error

def evaluate(prediction, ground_truth, metric="mae"):
    """
    Calculates Mean Absolute Error for Pillar 11.
    """
    if metric == "mae":
        print(f"  - (Pillar 11) Evaluating prediction using '{metric}'.")
        pred_np = prediction.cpu().numpy()
        gt_np = ground_truth.cpu().numpy()
        score = mean_absolute_error(gt_np, pred_np)
        return score
    else:
        print(f"  - Metric '{metric}' not implemented. Returning 0.0.")
        return 0.0
