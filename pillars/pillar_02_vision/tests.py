from .data_loader import load_data
from .evaluate import evaluate
import torch

def run_all_tests(model):
    """
    Runs all 10 tests for the Vision pillar and returns the results.
    """
    print("==================================================")
    print("  Running Pillar 2: Instant Domain Shift (Vision)")
    print("==================================================")
    
    test_descriptions = {
        11: "Gaussian noise @ σ=0.5", 12: "Shot noise", 13: "Impulse noise",
        14: "Defocus blur", 15: "Glass blur", 16: "Motion blur",
        17: "Zoom blur", 18: "Snow", 19: "Frost", 20: "Fog"
    }
    
    results = {}
    for test_id, description in test_descriptions.items():
        print(f"--- Running Pillar 2 Test #{test_id}: {description} ---")
        data, ground_truth = load_data(test_id)
        device = next(model.parameters()).device
        data = data.to(device)

        prediction = model.predict(data, pillar_id=2)
        score = evaluate(prediction, ground_truth)
        results[test_id] = score
        print(f"  - Real Accuracy: {score:.2f}%")
        print(f"--- Test #{test_id} Complete ---")
        
    valid_scores = [s for s in results.values() if s is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    results['average_accuracy'] = avg_score
    
    print("-" * 45)
    print(f"Pillar 2 Average Accuracy: {avg_score:.2f}%")
    print("-" * 45)
    
    return results

if __name__ == '__main__':
    # This would require loading the model to test standalone
    # from architecture.fluid_network import FluidNetwork
    # model = FluidNetwork()
    run_all_tests()
