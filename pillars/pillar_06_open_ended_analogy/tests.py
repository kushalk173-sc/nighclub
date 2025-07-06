
import time
from .data_loader import load_data
from .evaluate import evaluate

def run_test(test_id, description, model):
    print(f"--- Running Pillar 6 Test #{test_id}: {description} ---")

    print("Loading data...")
    data, labels = load_data(test_id)

    print("Getting model prediction...")
    if model:
        prediction = model.predict(data, pillar_id=6)
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

def run_pillar_6_tests(model=None):
    print(f"{'='*50}")
    print(f"  Running Pillar 6: Open-Ended Analogy (ConceptARC)")
    print(f"{'='*50}\n")

    test_definitions = {'51': 'ConceptARC Analogy Test #1', '52': 'ConceptARC Analogy Test #2', '53': 'ConceptARC Analogy Test #3', '54': 'ConceptARC Analogy Test #4', '55': 'ConceptARC Analogy Test #5', '56': 'ConceptARC Analogy Test #6', '57': 'ConceptARC Analogy Test #7', '58': 'ConceptARC Analogy Test #8', '59': 'ConceptARC Analogy Test #9', '60': 'ConceptARC Analogy Test #10'}
    results = {}
    for test_id, description in test_definitions.items():
        results[test_id] = run_test(test_id, description, model)

    avg_score = sum(results.values()) / len(results) if results else 0
    print("---------------------------------------------")
    print(f"Pillar 6 Average Score (MAE): {avg_score:.4f}")
    print("---------------------------------------------")
    return results

if __name__ == '__main__':
    run_pillar_6_tests()
