
import time
from .data_loader import load_data
from .evaluate import evaluate

def run_test(test_id, description, model):
    print(f"--- Running Pillar 8 Test #{test_id}: {description} ---")

    print("Loading data...")
    data, labels = load_data(test_id)

    print("Getting model prediction...")
    if model:
        prediction = model.predict(data, pillar_id=8)
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

def run_pillar_8_tests(model=None):
    print(f"{'='*50}")
    print(f"  Running Pillar 8: Graceful Degradation / Weight Drop")
    print(f"{'='*50}\n")

    test_definitions = {'71': 'Weight Drop Test #1', '72': 'Weight Drop Test #2', '73': 'Weight Drop Test #3', '74': 'Weight Drop Test #4', '75': 'Weight Drop Test #5', '76': 'Weight Drop Test #6', '77': 'Weight Drop Test #7', '78': 'Weight Drop Test #8', '79': 'Weight Drop Test #9', '80': 'Weight Drop Test #10'}
    results = {}
    for test_id, description in test_definitions.items():
        results[test_id] = run_test(test_id, description, model)

    avg_score = sum(results.values()) / len(results) if results else 0
    print("---------------------------------------------")
    print(f"Pillar 8 Average Score (MAE): {avg_score:.4f}")
    print("---------------------------------------------")
    return results

if __name__ == '__main__':
    run_pillar_8_tests()
