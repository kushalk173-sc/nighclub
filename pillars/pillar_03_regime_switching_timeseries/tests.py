
import time
from .data_loader import load_data
from .evaluate import evaluate

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

if __name__ == '__main__':
    run_pillar_3_tests()
