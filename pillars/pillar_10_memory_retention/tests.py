
import time
from .data_loader import load_data
from .evaluate import evaluate

def run_test(test_id, description, model):
    print(f"--- Running Pillar 10 Test #{test_id}: {description} ---")

    print("Loading data...")
    data, labels = load_data(test_id)

    print("Getting model prediction...")
    if model:
        prediction = model.predict(data, pillar_id=10)
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

def run_pillar_10_tests(model=None):
    print(f"{'='*50}")
    print(f"  Running Pillar 10: Memory Retention under Continual Learning")
    print(f"{'='*50}\n")

    test_definitions = {'91': 'Continual Learning Test #1', '92': 'Continual Learning Test #2', '93': 'Continual Learning Test #3', '94': 'Continual Learning Test #4', '95': 'Continual Learning Test #5', '96': 'Continual Learning Test #6', '97': 'Continual Learning Test #7', '98': 'Continual Learning Test #8', '99': 'Continual Learning Test #9', '100': 'Continual Learning Test #10'}
    results = {}
    for test_id, description in test_definitions.items():
        results[test_id] = run_test(test_id, description, model)

    avg_score = sum(results.values()) / len(results) if results else 0
    print("---------------------------------------------")
    print(f"Pillar 10 Average Score (MAE): {avg_score:.4f}")
    print("---------------------------------------------")
    return results

if __name__ == '__main__':
    run_pillar_10_tests()
