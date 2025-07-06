
import time
from .data_loader import load_data
from .evaluate import evaluate

def run_test(test_id, description, model):
    print(f"--- Running Pillar 7 Test #{test_id}: {description} ---")

    print("Loading data...")
    data, labels = load_data(test_id)

    print("Getting model prediction...")
    if model:
        prediction = model.predict(data, pillar_id=7)
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

def run_pillar_7_tests(model=None):
    print(f"{'='*50}")
    print(f"  Running Pillar 7: Constraint Satisfaction")
    print(f"{'='*50}\n")

    test_definitions = {'61': 'Constraint Satisfaction Test #1', '62': 'Constraint Satisfaction Test #2', '63': 'Constraint Satisfaction Test #3', '64': 'Constraint Satisfaction Test #4', '65': 'Constraint Satisfaction Test #5', '66': 'Constraint Satisfaction Test #6', '67': 'Constraint Satisfaction Test #7', '68': 'Constraint Satisfaction Test #8', '69': 'Constraint Satisfaction Test #9', '70': 'Constraint Satisfaction Test #10'}
    results = {}
    for test_id, description in test_definitions.items():
        results[test_id] = run_test(test_id, description, model)

    avg_score = sum(results.values()) / len(results) if results else 0
    print("---------------------------------------------")
    print(f"Pillar 7 Average Score (MAE): {avg_score:.4f}")
    print("---------------------------------------------")
    return results

if __name__ == '__main__':
    run_pillar_7_tests()
