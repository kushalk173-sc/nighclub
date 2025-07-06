
import time
from .data_loader import load_data
from .evaluate import evaluate

def run_test(test_id, description, model):
    print(f"--- Running Pillar 11 Test #{test_id}: {description} ---")

    print("Loading data...")
    data, labels = load_data(test_id)

    print("Getting model prediction...")
    if model:
        prediction = model.predict(data, pillar_id=11)
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

def run_pillar_11_tests(model=None):
    print(f"{'='*50}")
    print(f"  Running Pillar 11: OOD Geometry Fit (Synthetic manifolds)")
    print(f"{'='*50}\n")

    test_definitions = {'101': 'OOD Geometry Test #1', '102': 'OOD Geometry Test #2', '103': 'OOD Geometry Test #3', '104': 'OOD Geometry Test #4', '105': 'OOD Geometry Test #5', '106': 'OOD Geometry Test #6', '107': 'OOD Geometry Test #7', '108': 'OOD Geometry Test #8', '109': 'OOD Geometry Test #9', '110': 'OOD Geometry Test #10'}
    results = {}
    for test_id, description in test_definitions.items():
        results[test_id] = run_test(test_id, description, model)

    avg_score = sum(results.values()) / len(results) if results else 0
    print("---------------------------------------------")
    print(f"Pillar 11 Average Score (MAE): {avg_score:.4f}")
    print("---------------------------------------------")
    return results

if __name__ == '__main__':
    run_pillar_11_tests()
