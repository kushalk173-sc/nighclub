from .data_loader import load_data
from .evaluate import evaluate
import torch

def run_all_tests(model):
    """
    Runs all 10 tests for a generic numeric pillar and returns the results.
    """
    pillar_id = 3
    pillar_name = "Regime-Switching Time-Series (M4)"
    metric_name = "mae"
    
    print("==================================================")
    print(f"  Running Pillar {pillar_id}: {pillar_name}")
    print("==================================================")
    
    results = {}
    for i in range(10):
        test_id = (pillar_id - 1) * 10 + 1 + i
        print(f"--- Running Pillar {pillar_id} Test #{test_id}: {pillar_name} Test #{i+1} ---")
        data, _ = load_data(test_id)
        device = next(model.parameters()).device
        data = data.to(device)

        prediction = model.predict(data, pillar_id=pillar_id)
        score = evaluate(prediction)
        results[test_id] = score
        print(f"  - Score ({metric_name.upper()}): {score:.4f}")
        print(f"--- Test #{test_id} Complete ---")
        
    valid_scores = [s for s in results.values() if s is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    results[f'average_{metric_name}'] = avg_score
    
    print("-" * 45)
    print(f"Pillar {pillar_id} Average Score ({metric_name.upper()}): {avg_score:.4f}")
    print("-" * 45)
    
    return results 