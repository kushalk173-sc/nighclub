import torch
import random

def evaluate(prediction):
    """
    Evaluates the prediction for a vision task.
    Since we are not passing ground truth labels, this returns a random score
    to simulate an evaluation process.
    """
    print("  - (Pillar 2) Evaluating mock prediction using 'accuracy'.")
    # Return a random score between 0 and 100
    return random.uniform(0, 100)
