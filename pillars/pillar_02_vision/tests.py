import time
from .data_loader import load_data
from .evaluate import evaluate

def run_test(test_id, description, model):
    """
    A helper function to run a single test for Pillar 2.
    """
    print(f"--- Running Pillar 2 Test #{test_id}: {description} ---")

    # 1. Load data
    print("Loading data...")
    # This now returns a batch of image tensors and labels
    image_batch, labels = load_data(test_id)
    print(f"  - Loaded image batch of size: {image_batch.shape}")

    # 2. Get model prediction
    print("Getting model prediction...")
    if model:
        prediction = model.predict(image_batch, pillar_id=2)
    else:
        # This case is now less relevant but kept for robustness
        print("  - No model provided, skipping prediction.")
        prediction = None
        time.sleep(0.2)

    # 3. Evaluate
    print("Evaluating...")
    if prediction is not None:
        # The metric is Top-1 accuracy for this pillar
        score = evaluate(prediction, labels, metric="accuracy")
    else:
        score = 0.0

    print(f"  - Score: {score:.2%}")
    print(f"--- Test #{test_id} Complete ---\n")
    return score

def run_pillar_2_tests(model=None):
    """
    Runs all tests for Pillar 2: Instant Domain Shift (Vision).
    """
    print(f"{'='*50}")
    print(f"  Running Pillar 2: Instant Domain Shift (Vision)")
    print(f"{'='*50}\n")

    test_definitions = {
        '11': 'Gaussian noise @ σ=0.5',
        '12': 'Shot noise',
        '13': 'Impulse noise',
        '14': 'Defocus blur',
        '15': 'Glass blur',
        '16': 'Motion blur',
        '17': 'Zoom blur',
        '18': 'Snow',
        '19': 'Frost',
        '20': 'Fog'
    }

    results = {}
    for test_id, description in test_definitions.items():
        results[test_id] = run_test(test_id, description, model)

    if not results:
        print("No tests defined for this pillar.")
        return {}

    avg_score = sum(results.values()) / len(results)
    print("---------------------------------------------")
    print(f"Pillar 2 Average Score: {avg_score:.2%}")
    print("---------------------------------------------")

    return results

if __name__ == '__main__':
    # This would require loading the model to test standalone
    # from architecture.fluid_network import FluidNetwork
    # model = FluidNetwork()
    run_pillar_2_tests()
