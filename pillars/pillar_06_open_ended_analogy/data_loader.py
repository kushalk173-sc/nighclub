import torch
import os
import json
import random
import numpy as np
from pathlib import Path
from utils.dev import get_device, to_device

def load_data(test_id, batch_size=4):
    """
    Loads real analogy data from the synthetic_analogy.jsonl file.
    Uses the processed data from data/pillar_6_processed/.
    """
    print(f"  - (Pillar 6) Loading real analogy data for test {test_id}.")
    
    # Path to processed data
    analogy_file = Path("data/pillar_6_processed/synthetic_analogy.jsonl")
    
    if not analogy_file.exists():
        raise FileNotFoundError(f"Analogy data file not found: {analogy_file}")
    
    # Load all analogy examples
    analogies = []
    with open(analogy_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                analogies.append(json.loads(line))
    
    if not analogies:
        raise ValueError("No analogy data found in the file")
    
    # Randomly sample batch_size examples
    selected_analogies = random.sample(analogies, min(batch_size, len(analogies)))
    
    # Extract word analogy data and convert to numeric representation
    input_patterns = []
    output_patterns = []
    skipped = 0
    for analogy in selected_analogies:
        try:
            # Handle word analogies: convert words to numeric indices
            stem = analogy.get('stem', [])
            choices = analogy.get('choices', [])
            answer = analogy.get('answer', 0)
            if not stem or not choices:
                raise ValueError(f"Invalid analogy format: {analogy}")
            vocab = {}
            word_idx = 0
            for word in stem:
                if word not in vocab:
                    vocab[word] = word_idx
                    word_idx += 1
            for choice_pair in choices:
                for word in choice_pair:
                    if word not in vocab:
                        vocab[word] = word_idx
                        word_idx += 1
            stem_pattern = [vocab[word] for word in stem]
            correct_choice = choices[answer]
            choice_pattern = [vocab[word] for word in correct_choice]
            all_choices_flat = []
            for choice_pair in choices:
                all_choices_flat.extend(choice_pair)
            input_pattern = stem_pattern + all_choices_flat
            output_pattern = choice_pattern
            input_pattern = input_pattern[:10] + [0] * max(0, 10 - len(input_pattern))
            output_pattern = output_pattern[:2] + [0] * max(0, 2 - len(output_pattern))
            # Ensure all elements are numeric
            if not all(isinstance(x, (int, float, np.integer, np.floating)) for x in input_pattern + output_pattern):
                raise ValueError(f"Non-numeric value in analogy: {analogy}")
            input_patterns.append(input_pattern)
            output_patterns.append(output_pattern)
        except Exception as e:
            print(f"  - Skipping malformed analogy entry: {e}")
            skipped += 1
    if not input_patterns:
        raise ValueError(f"No valid analogy data found in the file (skipped {skipped} entries)")
    # Convert to tensors
    try:
        input_data = torch.tensor(input_patterns, dtype=torch.float32)
        output_data = torch.tensor(output_patterns, dtype=torch.float32)
    except Exception as e:
        raise ValueError(f"Error converting analogy data to tensors: {e}")
    # Move to the correct device
    input_data = to_device(input_data)
    output_data = to_device(output_data)
    print(f"  - Loaded real analogy batch. Input shape: {input_data.shape}")
    print(f"  - Output shape: {output_data.shape}")
    return input_data, output_data
