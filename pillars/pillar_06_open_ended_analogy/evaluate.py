import torch
from sklearn.metrics import mean_absolute_error

def evaluate(prediction, ground_truth, metric="mae"):
    """
    Calculates Mean Absolute Error for Pillar 6.
    """
    if metric == "mae":
        print(f"  - (Pillar 6) Evaluating prediction using '{metric}'.")
        # Flatten to 2D: (batch, -1)
        pred_np = prediction.cpu().numpy().reshape(prediction.shape[0], -1)
        gt_np = ground_truth.cpu().numpy().reshape(ground_truth.shape[0], -1)
        score = mean_absolute_error(gt_np, pred_np)
        return score
    else:
        print(f"  - Metric '{metric}' not implemented. Returning 0.0.")
        return 0.0
