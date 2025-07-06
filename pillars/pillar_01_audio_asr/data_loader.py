import torch
import random

# A small set of mock transcripts to make the test more realistic
MOCK_TRANSCRIPTS = [
    "the quick brown fox jumps over the lazy dog",
    "a stitch in time saves nine",
    "curiosity killed the cat",
    "speak of the devil",
]

def load_audio_data(test_id):
    """
    Mocks the data loading process for a given ASR test.
    This now returns a realistic audio tensor.
    """
    print(f"  - (Pillar 1) Loading mock audio for test {test_id}.")
    
    # Simulate loading a 5-second audio clip at 16kHz, as a PyTorch tensor
    mock_audio_tensor = torch.randn(16000 * 5)
    
    # Provide a mock ground-truth transcript.
    ground_truth = random.choice(MOCK_TRANSCRIPTS)
    
    print(f"  - Loaded mock audio tensor. Shape: {mock_audio_tensor.shape}")
    
    return mock_audio_tensor, ground_truth 