import time
import random

# Mock data loader and evaluator for demonstration purposes
from .data_loader import load_data
from .evaluate import evaluate_wer
import torch

# Test descriptions for Pillar 1
TESTS = {
    1: "Clean studio recording + US accent",
    2: "Clean studio recording + UK accent",
    3: "Clean studio recording + Indian accent",
    4: "Slightly noisy room + US accent",
    5: "Noisy cafe + US accent",
    6: "Low-quality microphone + US accent",
    7: "Outdoor environment + US accent",
    8: "Whispering + US accent",
    9: "Smartphone recording + Australian accent",
    10: "Smartphone in car + Irish accent"
}

def run_test(model, test_id):
    """
    Runs a single test for the Audio-ASR pillar.
    """
    print(f"--- Running Test #{test_id}: {TESTS.get(test_id, 'Unknown Test')} ---")
    
    # 1. Load Data
    print("Loading data...")
    try:
        # The new data loader returns a batch
        audio_batch, ground_truth_batch = load_data(test_id)
        device = next(model.parameters()).device
        audio_batch = audio_batch.to(device)
    except Exception as e:
        print(f"Error loading data for test {test_id}: {e}")
        return {"error": str(e)}

    # 2. Get Model Prediction for the batch
    print("Getting model prediction...")
    try:
        # The model's transcribe function should handle a batch
        predictions = model.transcribe(audio_batch)
        print(f"  - Ground truth: {ground_truth_batch[0]} (showing first item)")
        print(f"  - Predicted: {predictions[0]} (showing first item)")
    except Exception as e:
        print(f"Error during model prediction for test {test_id}: {e}")
        return {"error": str(e)}

    # 3. Evaluate
    print("Evaluating...")
    try:
        score = evaluate_wer(predictions, ground_truth_batch)
        print(f"  - WER: {score:.2f}%")
    except Exception as e:
        print(f"Error during evaluation for test {test_id}: {e}")
        return {"error": str(e)}
        
    print(f"--- Test #{test_id} Complete ---")
    return {"wer": score}

def run_all_tests(model):
    """
    Runs all 10 tests for the Audio-ASR pillar and returns the results.
    """
    print("==================================================")
    print("  Running Pillar 1: Instant Domain Shift (Audio-ASR) ")
    print("==================================================")
  
    results = {}
    for test_id in TESTS:
        test_result = run_test(model, test_id)
        # Store the raw score from the result dictionary
        if 'wer' in test_result:
            results[test_id] = test_result['wer']
        else:
            results[test_id] = None # Indicate error

    # Calculate average and add it to the results dictionary
    valid_scores = [s for s in results.values() if s is not None]
    avg_wer = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    results['average_wer'] = avg_wer
    
    print("-" * 45)
    print(f"Pillar 1 Average WER: {avg_wer:.2f}%")
    print("-" * 45)

    return results

if __name__ == '__main__':
    # This allows running the tests directly for stand-alone testing
    run_all_tests() 