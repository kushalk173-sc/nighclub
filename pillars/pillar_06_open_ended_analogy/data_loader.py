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
    
    # Extract numeric patterns only
    input_patterns = []
    output_patterns = []
    
    for analogy in selected_analogies:
        # Look for numeric pattern data - ignore string metadata
        pattern_data = None
        
        # Check for grid/matrix data specifically
        for field in ['grid', 'matrix', 'pattern', 'data']:
            if field in analogy and isinstance(analogy[field], (list, np.ndarray)):
                # Verify it's numeric data
                if isinstance(analogy[field], list) and len(analogy[field]) > 0:
                    if isinstance(analogy[field][0], (int, float)) or (isinstance(analogy[field][0], list) and isinstance(analogy[field][0][0], (int, float))):
                        pattern_data = analogy[field]
                        break
                elif isinstance(analogy[field], np.ndarray):
                    pattern_data = analogy[field]
                    break
        
        if pattern_data is None:
            # If no explicit grid field, look for any numeric array data
            for key, value in analogy.items():
                if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                    # Skip string fields like 'king', 'queen', etc.
                    if key.lower() in ['king', 'queen', 'name', 'description', 'type']:
                        continue
                    if isinstance(value, list):
                        if isinstance(value[0], (int, float)) or (isinstance(value[0], list) and isinstance(value[0][0], (int, float))):
                            pattern_data = value
                            break
                    elif isinstance(value, np.ndarray):
                        pattern_data = value
                        break
        
        if pattern_data is None:
            raise ValueError(f"No valid numeric pattern found in analogy: {analogy}")
        
        # Convert to numpy array and ensure it's numeric
        try:
            pattern_array = np.array(pattern_data, dtype=np.float32)
            
            # Reshape to 2x16x16 if needed (2 patterns, each 16x16)
            if pattern_array.ndim == 1:
                if len(pattern_array) == 512:  # 2*16*16
                    pattern_array = pattern_array.reshape(2, 16, 16)
                elif len(pattern_array) == 256:  # 16*16
                    pattern_array = pattern_array.reshape(1, 16, 16)
                    pattern_array = np.tile(pattern_array, (2, 1, 1))  # Duplicate for input/output
            
            # Ensure we have 2 patterns
            if pattern_array.shape[0] == 1:
                pattern_array = np.tile(pattern_array, (2, 1, 1))
            elif pattern_array.shape[0] > 2:
                pattern_array = pattern_array[:2]
            
            input_patterns.append(pattern_array[0])  # First pattern
            output_patterns.append(pattern_array[1])  # Second pattern
            
        except Exception as e:
            raise ValueError(f"Error processing pattern data: {e}")
    
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
