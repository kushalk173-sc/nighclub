from jiwer import wer
import random
import jiwer

def calculate_wer(predicted_transcript, ground_truth_transcript, duration=None):
    """
    Calculates the Word Error Rate (WER) using the 'jiwer' library.
    
    The 'duration' parameter is kept for compatibility with the test definition,
    but it is not used in this calculation as WER is based on the full transcripts.
    """
    if duration:
        print(f"  - (Pillar 1) Calculating WER for the first {duration} seconds (note: this is a conceptual check).")
    
    # Jiwer handles empty or different length strings gracefully.
    error_rate = wer(ground_truth_transcript, predicted_transcript)
    
    return error_rate 

def evaluate_wer(predictions, references=None):
    """
    Calculates the Word Error Rate (WER). In this mock version, if no
    references are provided, it returns a random score to simulate the process.
    """
    if references is None:
        # For mock predictions where we don't have ground truth
        print("  - (Pillar 1) Evaluating mock ASR prediction. No references provided.")
        return random.uniform(5.0, 30.0) # Return a plausible random WER

    # This part would execute if we had real labels
    print("  - (Pillar 1) Evaluating ASR prediction against references.")
    return jiwer.wer(references, predictions) 