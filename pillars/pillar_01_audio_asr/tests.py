import time
import random

# Mock data loader and evaluator for demonstration purposes
from .data_loader import load_audio_data
from .evaluate import calculate_wer

def run_test(test_id, description, model):
    """
    A helper function to run a single ASR test.
    """
    print(f"--- Running Test #{test_id}: {description} ---")

    # 1. Load data
    print("Loading data...")
    audio_tensor, ground_truth_transcript = load_audio_data(test_id)
    print(f"  - Ground truth: '{ground_truth_transcript}'")

    # 2. Get model prediction
    print("Getting model prediction...")
    if model:
        predicted_transcript = model.transcribe(audio_tensor)
    else:
        # Fallback for stand-alone testing without a model
        predicted_transcript = "this is a dummy fallback transcript"
        time.sleep(1)

    print(f"  - Predicted: '{predicted_transcript}'")

    # 3. Evaluate
    print("Evaluating...")
    # For the first test, we note the requirement but calculate full WER
    if test_id == 1:
        wer = calculate_wer(predicted_transcript, ground_truth_transcript, duration=3)
    else:
        wer = calculate_wer(predicted_transcript, ground_truth_transcript)

    print(f"  - WER: {wer:.2%}")
    print(f"--- Test #{test_id} Complete ---\n")
    return wer

def run_pillar_1_tests(model=None):
    """
    Runs all 10 tests for Pillar 1: Instant Domain Shift (Audio-ASR).
    """
    print("=============================================")
    print("  Running Pillar 1: Instant Domain Shift (Audio-ASR) ")
    print("=============================================\n")

    test_definitions = {
        1: "Studio mic + US accent",
        2: "Laptop mic + British accent",
        3: "Headset mic + Australian accent",
        4: "Conference mic + Indian accent",
        5: "Smartphone outdoors + US accent (South)",
        6: "Smartphone in cafe + Scottish accent",
        7: "In-car system + German accent",
        8: "Stadium noise + US accent (announcer)",
        9: "Child speaking + US accent",
        10: "Smartphone in car + Irish accent",
    }

    results = {}
    for test_id, description in test_definitions.items():
        results[test_id] = run_test(test_id, description, model)

    avg_wer = sum(results.values()) / len(results)
    print("---------------------------------------------")
    print(f"Pillar 1 Average WER: {avg_wer:.2%}")
    print("---------------------------------------------")

    return results

if __name__ == '__main__':
    # This allows running the tests directly for stand-alone testing
    run_pillar_1_tests() 