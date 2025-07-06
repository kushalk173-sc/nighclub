
import time
from .data_loader import load_data
from .evaluate import evaluate

def run_test(test_id, description, model):
    print(f"--- Running Pillar 9 Test #{test_id}: {description} ---")

    print("Loading data...")
    data, labels = load_data(test_id)

    print("Getting model prediction...")
    if model:
        prediction = model.predict(data, pillar_id=9)
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

def run_pillar_9_tests(model=None):
    print(f"{'='*50}")
    print(f"  Running Pillar 9: Rapid Sensor Drift (Robotics sim)")
    print(f"{'='*50}\n")

    test_definitions = {'81': 'Robotics Sensor Drift Test #1', '82': 'Robotics Sensor Drift Test #2', '83': 'Robotics Sensor Drift Test #3', '84': 'Robotics Sensor Drift Test #4', '85': 'Robotics Sensor Drift Test #5', '86': 'Robotics Sensor Drift Test #6', '87': 'Robotics Sensor Drift Test #7', '88': 'Robotics Sensor Drift Test #8', '89': 'Robotics Sensor Drift Test #9', '90': 'Robotics Sensor Drift Test #10'}
    results = {}
    for test_id, description in test_definitions.items():
        results[test_id] = run_test(test_id, description, model)

    avg_score = sum(results.values()) / len(results) if results else 0
    print("---------------------------------------------")
    print(f"Pillar 9 Average Score (MAE): {avg_score:.4f}")
    print("---------------------------------------------")
    return results

if __name__ == '__main__':
    run_pillar_9_tests()
