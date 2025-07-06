import torch

def load_data(test_id, batch_size=4):
    """
    Generates a batch of dummy audio data.
    This is a robust placeholder to ensure the pipeline runs without real data.
    """
    print(f"  - (Pillar 1) Loading mock audio for test {test_id}.")
    
    # batch_size, num_samples (1 second of audio at 16kHz)
    wav = torch.randn(batch_size, 16000)
    
    # Mock ground truth for the batch
    target = ["the quick brown fox jumps over the lazy dog"] * batch_size
    
    print(f"  - Loaded mock audio batch. Shape: {wav.shape}")
    return wav, target 