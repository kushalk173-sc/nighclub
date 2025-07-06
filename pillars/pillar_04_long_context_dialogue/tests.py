import time
from .data_loader import load_data
from .evaluate import evaluate

def run_test(test_id, description, model):
    """
    A helper function to run a single test for Pillar 4.
    """
    print(f"--- Running Pillar 4 Test #{test_id}: {description} ---")

    # 1. Load data
    print("Loading data...")
    # This now returns a list of strings and corresponding labels
    dialogues, labels = load_data(test_id)

    # 2. Get model prediction
    print("Getting model prediction...")
    if model:
        prediction = model.predict(dialogues, pillar_id=4)
    else:
        prediction = None
        time.sleep(0.2)

    # 3. Evaluate
    print("Evaluating...")
    if prediction is not None:
        # The metric is Coherence F1 for this pillar
        score = evaluate(prediction, labels, metric="f1")
    else:
        score = 0.0

    print(f"  - Score: {score:.2f}")
    print(f"--- Test #{test_id} Complete ---\n")
    return score

def run_pillar_4_tests(model=None):
    """
    Runs all tests for Pillar 4: Long-Context Dialogue Coherence.
    """
    print(f"{'='*50}")
    print(f"  Running Pillar 4: Long-Context Dialogue Coherence")
    print(f"{'='*50}\n")

    test_definitions = {
        '31': '5 characters, 3 locations',
        '32': '10 characters, complex plot',
        '33': 'Long-term dependency check (10k tokens)',
        '34': 'Short-term dependency check',
        '35': 'Contradiction detection',
        '36': 'Multi-session coherence',
        '37': 'Emotional arc consistency',
        '38': 'Style consistency (formal vs. informal)',
        '39': '20 characters, flashbacks',
        '40': 'Ground-truth entity state tracking'
    }

    results = {}
    for test_id, description in test_definitions.items():
        results[test_id] = run_test(test_id, description, model)

    if not results:
        print("No tests defined for this pillar.")
        return {}

    avg_score = sum(results.values()) / len(results)
    print("---------------------------------------------")
    print(f"Pillar 4 Average Score: {avg_score:.2f}")
    print("---------------------------------------------")

    return results

if __name__ == '__main__':
    run_pillar_4_tests()
