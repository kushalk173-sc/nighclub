import time
import random
import sys

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
    """Helper to run a single ASR test."""
    print(f"--- Running Test #{test_id}: {TESTS[test_id]} ---")
    
    print("Loading data...")
    # Load both audio data and ground truth transcripts
    data, ground_truth = load_data(test_id)
    print(f"  - Loaded real audio batch. Shape: {data.shape}")

    print("Getting model prediction...")
    try:
        # Do not move data to device here; model will handle it
        
        # Fix input shape for wav2vec2: expects [batch, time] not [batch, channels, time]
        if data.dim() == 3 and data.shape[1] == 1:
            data = data.squeeze(1)  # Remove channel dimension: [B, 1, T] -> [B, T]
        
        # The model's transcribe method handles the forward pass
        prediction = model.transcribe(data)
        
        # Evaluate against real ground truth
        score = evaluate_wer(prediction, ground_truth)
        print(f"  - Real WER: {score:.2f}%")
    except Exception as e:
        print(f"Error during model prediction for test {test_id}: {e}", file=sys.stderr)
        score = 100.0 # Return max WER on error
    
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