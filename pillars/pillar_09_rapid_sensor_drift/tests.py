from .data_loader import load_data
from .evaluate import evaluate
import torch

def run_all_tests(model):
    """
    Runs all 10 tests for a generic pillar and returns the results.
    """
    pillar_id = 9
    pillar_name = "Rapid Sensor Drift"
    metric_name = "mae"
    
    print("==================================================")
    print(f"  Running Pillar 9: Rapid Sensor Drift")
    print("==================================================")
    
    results = {}
    for i in range(10):
        test_id = (pillar_id - 1) * 10 + 1 + i
        print(f"--- Running Pillar {pillar_id} Test #{test_id}: {pillar_name} Test #{i+1} ---")
        data, ground_truth = load_data(test_id)
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
    run_all_tests()
