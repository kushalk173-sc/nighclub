import numpy as np
import random
import pickle
from pathlib import Path

def load_data(test_id, batch_size=4):
    """
    Loads real multi-hop reasoning data from the validation_data.pkl file.
    Uses the processed data from data/pillar_5_processed/.
    """
    print(f"  - (Pillar 5) Loading real multi-hop reasoning data for test {test_id}.")
    
    # Path to processed data
    qa_file = Path("data/pillar_5_processed/validation_data.pkl")
    
    if not qa_file.exists():
        raise FileNotFoundError(f"QA data file not found: {qa_file}")
    
    # Load QA data
    try:
        with open(qa_file, 'rb') as f:
            qa_data = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Error loading QA data: {e}")
    
    # Handle different data formats
    if isinstance(qa_data, list):
        qa_examples = qa_data
    elif isinstance(qa_data, dict):
        # Extract QA examples from dictionary
        if 'questions' in qa_data:
            qa_examples = qa_data['questions']
        elif 'data' in qa_data:
            qa_examples = qa_data['data']
        elif 'qa_pairs' in qa_data:
            qa_examples = qa_data['qa_pairs']
        else:
            # Use the first list-like value
            qa_examples = next((v for v in qa_data.values() if isinstance(v, list)), [])
    else:
        raise ValueError(f"Unexpected QA data format: {type(qa_data)}")
    
    if not qa_examples:
        raise ValueError("No QA data found in the file")
    
    # Randomly sample batch_size examples
    selected_examples = random.sample(qa_examples, min(batch_size, len(qa_examples)))
    
    # Extract QA contexts and labels
    qa_contexts = []
    labels = []
    
    for example in selected_examples:
        if isinstance(example, str):
            # Single QA string
            qa_contexts.append(example)
            labels.append(1)  # Assume correct by default
        elif isinstance(example, dict):
            # Dictionary with question, context, and answer
            context_parts = []
            
            # Extract context
            if 'context' in example:
                context_parts.append(f"Context: {example['context']}")
            elif 'passage' in example:
                context_parts.append(f"Context: {example['passage']}")
            
            # Extract question
            if 'question' in example:
                context_parts.append(f"Question: {example['question']}")
            elif 'query' in example:
                context_parts.append(f"Question: {example['query']}")
            
            # Extract answer if available
            if 'answer' in example:
                context_parts.append(f"Answer: {example['answer']}")
            elif 'gold_answer' in example:
                context_parts.append(f"Answer: {example['gold_answer']}")
            
            qa_contexts.append(" ".join(context_parts))
            
            # Extract label
            if 'label' in example:
                labels.append(example['label'])
            elif 'correct' in example:
                labels.append(example['correct'])
            elif 'is_correct' in example:
                labels.append(example['is_correct'])
            else:
                labels.append(1)  # Default to correct
        elif isinstance(example, list):
            # List of QA components
            qa_contexts.append(" ".join(str(comp) for comp in example))
            labels.append(1)  # Default to correct
        else:
            # Fallback
            qa_contexts.append(str(example))
            labels.append(1)
    
    # Convert labels to numpy array
    ground_truth_labels = np.array(labels)
    
    print(f"  - Loaded real QA batch. Size: {len(qa_contexts)}")
    print(f"  - Sample QA: '{qa_contexts[0][:100]}...'")
    
    return qa_contexts, ground_truth_labels
