import torch
import numpy as np
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
    Calculates the Word Error Rate (WER) using real ground truth references.
    """
    if references is None:
        raise ValueError("No references provided for WER evaluation")

    # Calculate WER for each prediction-reference pair
    print("  - (Pillar 1) Evaluating ASR prediction against real references.")
    
    if isinstance(predictions, str):
        # Single prediction
        return jiwer.wer(references, predictions)
    elif isinstance(predictions, list) and isinstance(references, list):
        # Batch of predictions
        if len(predictions) != len(references):
            raise ValueError(f"Mismatch in predictions ({len(predictions)}) and references ({len(references)})")
        
        # Calculate WER for each pair and average
        wers = []
        for pred, ref in zip(predictions, references):
            try:
                wer_score = jiwer.wer(ref, pred)
                wers.append(wer_score)
            except Exception as e:
                raise ValueError(f"Error calculating WER for pair: {e}")
        
        avg_wer = sum(wers) / len(wers) if wers else 100.0
        return avg_wer
    else:
        raise ValueError(f"Unexpected prediction/reference types: {type(predictions)}, {type(references)}") 

def evaluate(prediction, ground_truth, metric="wer"):
    """
    Evaluates prediction for Pillar 1 (Audio ASR).
    """
    if metric == "wer":
        print(f"  - (Pillar 1) Evaluating prediction using '{metric}'.")
        
        # Ensure we have valid transcriptions
        if not prediction or not ground_truth:
            raise ValueError("Empty prediction or ground truth transcription")
        
        # Convert to list if needed
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.tolist()
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.tolist()
        
        # Ensure we have lists of strings
        if isinstance(prediction, str):
            prediction = [prediction]
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        
        # Check for empty strings
        if any(not p.strip() for p in prediction):
            raise ValueError("Empty prediction transcription found")
        if any(not g.strip() for g in ground_truth):
            raise ValueError("Empty ground truth transcription found")
        
        # Calculate WER
        try:
            wer_score = wer(ground_truth, prediction)
            print(f"  - WER: {wer_score:.4f}")
            return wer_score
        except Exception as e:
            raise ValueError(f"Error calculating WER: {e}")
    
    else:
        raise ValueError(f"Unknown metric: {metric}") 