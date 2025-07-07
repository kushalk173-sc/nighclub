import time
from .data_loader import load_data
from .evaluate import evaluate
import torch

def run_test(test_id, description, model):
    print(f"--- Running Pillar 3 Test #{test_id}: {description} ---")

    print("Loading data...")
    data, labels = load_data(test_id)

    print("Getting model prediction...")
    if model:
        prediction = model.predict(data, pillar_id=3)
    else:
        prediction = None
        time.sleep(0.2)

    print("Evaluating...")
    if prediction is not None:
        score = evaluate(prediction, labels, metric="mae")
    else:
        score = 0.0

    print(f"  - Score (MAE): {score:.4f}")
    print(f"--- Test #{test_id} Complete ---\n")
    return score

def run_pillar_3_tests(model=None):
    print(f"{'='*50}")
    print(f"  Running Pillar 3: Regime-Switching Time-Series (M4)")
    print(f"{'='*50}\n")

    test_definitions = {'21': 'M4 Time-Series Test #1', '22': 'M4 Time-Series Test #2', '23': 'M4 Time-Series Test #3', '24': 'M4 Time-Series Test #4', '25': 'M4 Time-Series Test #5', '26': 'M4 Time-Series Test #6', '27': 'M4 Time-Series Test #7', '28': 'M4 Time-Series Test #8', '29': 'M4 Time-Series Test #9', '30': 'M4 Time-Series Test #10'}
    results = {}
    for test_id, description in test_definitions.items():
        results[test_id] = run_test(test_id, description, model)

    avg_score = sum(results.values()) / len(results) if results else 0
    print("---------------------------------------------")
    print(f"Pillar 3 Average Score (MAE): {avg_score:.4f}")
    print("---------------------------------------------")
    return results

def run_all_tests(model):
    """
    Runs all 10 tests for a generic pillar and returns the results.
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
    # This would require loading the model to test standalone
    # from architecture.fluid_network import FluidNetwork
    # model = FluidNetwork()
    run_all_tests()
