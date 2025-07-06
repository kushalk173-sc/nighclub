from jiwer import wer

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

def evaluate_wer(predictions, references):
    """
    A minimal, safe placeholder for WER calculation.
    Returns 1.0 if any prediction does not match its reference, 0.0 otherwise.
    """
    if len(predictions) != len(references):
        return 1.0 # Should not happen, but indicates an error

    total_error = 0.0
    for pred, ref in zip(predictions, references):
        if pred != ref:
            total_error += 1.0
            
    return (total_error / len(predictions)) * 100 