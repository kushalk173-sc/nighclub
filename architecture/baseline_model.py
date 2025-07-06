import numpy as np
import time

class BaselineModel:
    """
    A simple, universal baseline model for the testbed.
    
    This model does not perform any real computation. It's a placeholder
    to demonstrate that the full testing pipeline is working, from data loading
    to model prediction to evaluation.
    """
    def __init__(self):
        print("Initialized a simple baseline model.")
        self.supported_pillars = list(range(1, 12))

    def transcribe(self, audio_data):
        """
        Mocks the transcription process for Pillar 1 (ASR).
        """
        print("  - [BaselineModel] 'Transcribing' audio data...")
        # Simulate some processing time
        time.sleep(0.5)
        # Return a fixed, slightly different transcript to ensure WER is non-zero
        return "this is a baseline transcription"

    def predict(self, data, pillar_id):
        """
        Mocks the prediction process for all other pillars.
        """
        if pillar_id not in self.supported_pillars:
            raise ValueError(f"Pillar {pillar_id} is not supported by this model.")

        print(f"  - [BaselineModel] 'Predicting' for Pillar {pillar_id}...")
        # Simulate some processing time
        time.sleep(0.5)
        
        # Return a random value to simulate a model's output score
        return np.random.rand()

    def __str__(self):
        return "BaselineModel" 