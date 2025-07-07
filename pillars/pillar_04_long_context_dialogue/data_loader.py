import numpy as np
import random
import pickle
from pathlib import Path

def load_data(test_id, batch_size=4):
    """
    Loads real dialogue data from the validation_data.pkl file.
    Uses the processed data from data/pillar_4_processed/.
    """
    print(f"  - (Pillar 4) Loading real dialogue data for test {test_id}.")
    
    # Path to processed data
    dialogue_file = Path("data/pillar_4_processed/validation_data.pkl")
    
    if not dialogue_file.exists():
        raise FileNotFoundError(f"Dialogue data file not found: {dialogue_file}")
    
    # Load dialogue data
    try:
        with open(dialogue_file, 'rb') as f:
            dialogue_data = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Error loading dialogue data: {e}")
    
    # Handle different data formats
    if isinstance(dialogue_data, list):
        dialogues = dialogue_data
    elif isinstance(dialogue_data, dict):
        # Extract dialogues from dictionary
        if 'dialogues' in dialogue_data:
            dialogues = dialogue_data['dialogues']
        elif 'data' in dialogue_data:
            dialogues = dialogue_data['data']
        else:
            # Use the first list-like value
            dialogues = next((v for v in dialogue_data.values() if isinstance(v, list)), [])
    else:
        raise ValueError(f"Unexpected dialogue data format: {type(dialogue_data)}")
    
    if not dialogues:
        raise ValueError("No dialogue data found in the file")
    
    # Randomly sample batch_size examples
    selected_dialogues = random.sample(dialogues, min(batch_size, len(dialogues)))
    
    # Extract dialogue text and labels
    dialogue_texts = []
    labels = []
    
    for dialogue in selected_dialogues:
        if isinstance(dialogue, str):
            # Single dialogue string
            dialogue_texts.append(dialogue)
            labels.append(1)  # Assume coherent by default
        elif isinstance(dialogue, dict):
            # Dictionary with dialogue and label
            if 'text' in dialogue:
                dialogue_texts.append(dialogue['text'])
            elif 'dialogue' in dialogue:
                dialogue_texts.append(dialogue['dialogue'])
            else:
                # Use the first string value
                text_val = next((v for v in dialogue.values() if isinstance(v, str)), "Empty dialogue")
                dialogue_texts.append(text_val)
            
            # Extract label
            if 'label' in dialogue:
                labels.append(dialogue['label'])
            elif 'coherent' in dialogue:
                labels.append(dialogue['coherent'])
            else:
                labels.append(1)  # Default to coherent
        elif isinstance(dialogue, list):
            # List of dialogue turns
            dialogue_texts.append(" ".join(str(turn) for turn in dialogue))
            labels.append(1)  # Default to coherent
        else:
            # Fallback
            dialogue_texts.append(str(dialogue))
            labels.append(1)
    
    # Convert labels to numpy array
    ground_truth_labels = np.array(labels)
    
    print(f"  - Loaded real dialogue batch. Size: {len(dialogue_texts)}")
    print(f"  - Sample dialogue: '{dialogue_texts[0][:100]}...'")
    
    return dialogue_texts, ground_truth_labels
