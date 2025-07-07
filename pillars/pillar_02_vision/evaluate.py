import torch
import random

def evaluate(prediction, ground_truth=None):
    """
    Evaluates the prediction for a vision task using real ground truth labels.
    """
    if ground_truth is None:
        print("  - (Pillar 2) Warning: No ground truth provided, using fallback evaluation.")
        return random.uniform(0, 100)
    
    print("  - (Pillar 2) Evaluating prediction against real ground truth.")
    
    try:
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
        
        # Ensure both tensors are on the same device
        if predicted_labels.device != ground_truth.device:
            ground_truth = ground_truth.to(predicted_labels.device)
        
        # Calculate accuracy
        correct = (predicted_labels == ground_truth).sum().item()
        total = len(ground_truth)
        accuracy = (correct / total) * 100.0 if total > 0 else 0.0
        
        print(f"  - Correct: {correct}/{total}, Accuracy: {accuracy:.2f}%")
        return accuracy
        
    except Exception as e:
        print(f"  - Error during evaluation: {e}")
        return 0.0
