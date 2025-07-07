import torch
import json
import random
from pathlib import Path

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
    
    # Extract input and output patterns
    input_patterns = []
    output_patterns = []
    
    for analogy in selected_analogies:
        # Assuming the analogy data has 'input' and 'output' fields
        # Adjust field names based on actual data structure
        if 'input' in analogy and 'output' in analogy:
            input_patterns.append(analogy['input'])
            output_patterns.append(analogy['output'])
        elif 'pattern' in analogy and 'solution' in analogy:
            input_patterns.append(analogy['pattern'])
            output_patterns.append(analogy['solution'])
        else:
            # Use the first two keys as input/output
            keys = list(analogy.keys())
            if len(keys) >= 2:
                input_patterns.append(analogy[keys[0]])
                output_patterns.append(analogy[keys[1]])
            else:
                raise ValueError(f"Invalid analogy data structure: {analogy}")
    
    # Convert to tensors if they aren't already
    # This is a simplified conversion - adjust based on actual data format
    try:
        if isinstance(input_patterns[0], list):
            input_data = torch.tensor(input_patterns, dtype=torch.float32)
        else:
            input_data = torch.stack(input_patterns) if isinstance(input_patterns[0], torch.Tensor) else torch.tensor(input_patterns)
        
        if isinstance(output_patterns[0], list):
            output_data = torch.tensor(output_patterns, dtype=torch.float32)
        else:
            output_data = torch.stack(output_patterns) if isinstance(output_patterns[0], torch.Tensor) else torch.tensor(output_patterns)
    except Exception as e:
        raise ValueError(f"Error converting analogy data to tensors: {e}")
    
    print(f"  - Loaded real analogy batch. Input shape: {input_data.shape}")
    print(f"  - Output shape: {output_data.shape}")
    
    return input_data, output_data
