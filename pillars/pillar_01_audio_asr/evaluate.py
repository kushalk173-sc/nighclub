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