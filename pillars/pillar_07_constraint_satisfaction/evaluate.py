import torch
import numpy as np
from sklearn.metrics import mean_absolute_error

def evaluate(prediction, ground_truth, metric="mae"):
    """
    Evaluates prediction for Pillar 7 (Constraint Satisfaction).
    """
    if metric == "mae":
        print(f"  - (Pillar 7) Evaluating prediction using '{metric}'.")
        
        # Convert to numpy for evaluation
        pred_np = prediction.cpu().numpy()
        gt_np = ground_truth.cpu().numpy()
        
        # Handle different output shapes
        if pred_np.shape != gt_np.shape:
            print(f"  - Shape mismatch: prediction {pred_np.shape} vs ground truth {gt_np.shape}")
            
            # If prediction is 1D and ground truth is 2D, reshape prediction
            if len(pred_np.shape) == 1 and len(gt_np.shape) == 2:
                # Assume prediction should be reshaped to match ground truth
                if pred_np.size == gt_np.size:
                    pred_np = pred_np.reshape(gt_np.shape)
                else:
                    # Pad or truncate to match
                    target_size = gt_np.size
                    if pred_np.size < target_size:
                        # Pad with zeros
                        padded = np.zeros(target_size)
                        padded[:pred_np.size] = pred_np.flatten()
                        pred_np = padded.reshape(gt_np.shape)
                    else:
                        # Truncate
                        pred_np = pred_np.flatten()[:target_size].reshape(gt_np.shape)
            elif len(pred_np.shape) == 2 and len(gt_np.shape) == 1:
                # If prediction is 2D and ground truth is 1D, flatten prediction
                pred_np = pred_np.flatten()
                if pred_np.size != gt_np.size:
                    # Pad or truncate
                    if pred_np.size < gt_np.size:
                        padded = np.zeros(gt_np.size)
                        padded[:pred_np.size] = pred_np
                        pred_np = padded
                    else:
                        pred_np = pred_np[:gt_np.size]
        
        # Handle 3D tensors (batch, 9, 9) by flattening
        if len(pred_np.shape) == 3:
            pred_np = pred_np.reshape(pred_np.shape[0], -1)
        if len(gt_np.shape) == 3:
            gt_np = gt_np.reshape(gt_np.shape[0], -1)
        
        score = mean_absolute_error(gt_np, pred_np)
        print(f"  - Real Score (MAE): {score:.4f}")
        return score
    else:
        print(f"  - Metric '{metric}' not implemented. Returning 0.0.")
        return 0.0
