from .data_loader import load_data
from .evaluate import evaluate
import torch

def run_all_tests(model):
    """
    Runs all 10 tests for a generic pillar and returns the results.
    """
    pillar_id = 7
    pillar_name = "Constraint Satisfaction"
    metric_name = "mae"
    
    print("==================================================")
    print(f"  Running Pillar 7: Constraint Satisfaction")
    print("==================================================")
    
    results = {}
    for i in range(10):
        test_id = (pillar_id - 1) * 10 + 1 + i
        print(f"--- Running Pillar {pillar_id} Test #{test_id}: {pillar_name} Test #{i+1} ---")
        data, ground_truth = load_data(test_id)
        # Reshape data to match baseline model's expected input shape [B, 10]
        if data.dim() == 3:  # [B, H, W] -> [B, H*W]
            data = data.view(data.shape[0], -1)
        if data.shape[1] != 10:  # If not 10 features, pad or truncate
            if data.shape[1] > 10:
                data = data[:, :10]  # Truncate to 10 features
            else:
                # Pad with zeros to 10 features
                padding = torch.zeros(data.shape[0], 10 - data.shape[1])
                data = torch.cat([data, padding], dim=1)
        # Do not move data to device here; model will handle it

        prediction = model.predict(data, pillar_id=pillar_id)
        score = evaluate(prediction, ground_truth, metric=metric_name)
        results[test_id] = score
        
        # Determine format based on metric
        if metric_name == "mae":
            print(f"  - Real Score ({metric_name.upper()}): {score:.4f}")
        else:
            print(f"  - Real Score ({metric_name.upper()}): {score:.2f}%")
            
        print(f"--- Test #{test_id} Complete ---")
        
    valid_scores = [s for s in results.values() if s is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    results[f'average_{metric_name}'] = avg_score
    
    print("-" * 45)
    if metric_name == "mae":
        print(f"Pillar {pillar_id} Average Score ({metric_name.upper()}): {avg_score:.4f}")
    else:
        print(f"Pillar {pillar_id} Average Score ({metric_name.upper()}): {avg_score:.2f}%")
    print("-" * 45)
    
    return results

if __name__ == '__main__':
    # This would require loading the model to test standalone
    # from architecture.fluid_network import FluidNetwork
    # model = FluidNetwork()
    run_all_tests()
