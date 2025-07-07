import torch
import os
import random
from pathlib import Path

def load_data(test_id, batch_size=4):
    """
    Loads real LibriSpeech audio data and corresponding transcripts.
    Uses the processed data from data/pillar_1_processed/.
    """
    print(f"  - (Pillar 1) Loading real LibriSpeech data for test {test_id}.")
    
    # Paths to processed data
    audio_dir = Path("data/pillar_1_processed/audio")
    text_dir = Path("data/pillar_1_processed/text")
    
    # Get all available audio files
    audio_files = list(audio_dir.glob("*.pt"))
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")
    
    # Randomly sample batch_size files
    selected_files = random.sample(audio_files, min(batch_size, len(audio_files)))
    
    # Load audio tensors
    audio_batch = []
    text_batch = []
    
    for audio_file in selected_files:
        # Load audio tensor with weights_only=True to suppress warnings
        audio_tensor = torch.load(audio_file, weights_only=True)
        # Ensure tensor is on CPU
        audio_tensor = audio_tensor.cpu()
        audio_batch.append(audio_tensor)
        
        # Load corresponding text
        text_file = text_dir / f"{audio_file.stem}.txt"
        if text_file.exists():
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            text = "UNKNOWN"  # Fallback if text file missing
        text_batch.append(text)
    
    # Stack audio tensors into a batch
    # Handle variable lengths by padding or truncating to the shortest length
    min_length = min(tensor.shape[-1] for tensor in audio_batch)
    padded_audio = []
    for tensor in audio_batch:
        if tensor.shape[-1] > min_length:
            # Truncate to min_length
            padded_audio.append(tensor[..., :min_length])
        else:
            # Pad with zeros if needed (shouldn't happen with our min_length logic)
            padded = torch.zeros(tensor.shape[:-1] + (min_length,))
            padded[..., :tensor.shape[-1]] = tensor
            padded_audio.append(padded)
    
    wav = torch.stack(padded_audio)
    
    print(f"  - Loaded real audio batch. Shape: {wav.shape}")
    print(f"  - Sample transcript: '{text_batch[0][:50]}...'")
    
    return wav, text_batch 