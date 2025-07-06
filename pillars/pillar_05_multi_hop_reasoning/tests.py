import time
from .data_loader import load_data
from .evaluate import evaluate

def run_test(test_id, description, model):
    """
    A helper function to run a single test for Pillar 5.
    """
    print(f"--- Running Pillar 5 Test #{test_id}: {description} ---")

    # 1. Load data
    print("Loading data...")
    qa_contexts, labels = load_data(test_id)

    # 2. Get model prediction
    print("Getting model prediction...")
    if model:
        prediction = model.predict(qa_contexts, pillar_id=5)
    else:
        prediction = None
        time.sleep(0.2)

    # 3. Evaluate
    print("Evaluating...")
    if prediction is not None:
        # The metric is Exact-Match, but we use accuracy as a proxy
        score = evaluate(prediction, labels, metric="accuracy")
    else:
        score = 0.0

    print(f"  - Score: {score:.2%}")
    print(f"--- Test #{test_id} Complete ---\n")
    return score

def run_pillar_5_tests(model=None):
    """
    Runs all tests for Pillar 5: Multi-Hop Reasoning.
    """
    print(f"{'='*50}")
    print(f"  Running Pillar 5: Multi-Hop Reasoning")
    print(f"{'='*50}\n")

    test_definitions = {
        '41': '2-hop factual',
        '42': '3-hop factual',
        '43': '4-hop factual',
        '44': '2-hop causal chain',
        '45': '3-hop causal chain',
        '46': '4-hop temporal reasoning',
        '47': '5-hop compositional reasoning',
        '48': '6-hop comparison',
        '49': 'Bridge entity reasoning',
        '50': '6-hop causal chain'
    }

    results = {}
    for test_id, description in test_definitions.items():
        results[test_id] = run_test(test_id, description, model)

    if not results:
        print("No tests defined for this pillar.")
        return {}

    avg_score = sum(results.values()) / len(results)
    print("---------------------------------------------")
    print(f"Pillar 5 Average Score: {avg_score:.2%}")
    print("---------------------------------------------")

    return results

if __name__ == '__main__':
    run_pillar_5_tests()
