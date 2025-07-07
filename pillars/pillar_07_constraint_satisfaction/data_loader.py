import torch
import numpy as np
import random
from pathlib import Path

def load_data(test_id, batch_size=4):
    """
    Loads real Sudoku puzzles and solutions from processed data.
    Uses the processed data from data/pillar_7_processed/.
    """
    print(f"  - (Pillar 7) Loading real Sudoku data for test {test_id}.")
    
    # Paths to processed data
    puzzles_file = Path("data/pillar_7_processed/puzzles.npy")
    solutions_file = Path("data/pillar_7_processed/solutions.npy")
    
    if not puzzles_file.exists() or not solutions_file.exists():
        raise FileNotFoundError(f"Sudoku data files not found in data/pillar_7_processed/")
    
    # Load Sudoku puzzles and solutions
    puzzles = np.load(puzzles_file)
    solutions = np.load(solutions_file)
    
    # Randomly sample batch_size puzzles
    n_puzzles = len(puzzles)
    if batch_size > n_puzzles:
        batch_size = n_puzzles
    
    indices = random.sample(range(n_puzzles), batch_size)
    selected_puzzles = puzzles[indices]
    selected_solutions = solutions[indices]
    
    # Convert to tensors
    puzzle_tensors = torch.tensor(selected_puzzles, dtype=torch.float32)
    solution_tensors = torch.tensor(selected_solutions, dtype=torch.float32)
    
    print(f"  - Loaded real Sudoku batch. Shape: {puzzle_tensors.shape}")
    print(f"  - Solutions shape: {solution_tensors.shape}")
    
    return puzzle_tensors, solution_tensors
