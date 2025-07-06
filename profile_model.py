import argparse
import time
import torch
from main import get_model, run_pillar_tests
from utils.dev import get_device

def profile_model(version_str):
    """
    Profiles a model's performance by running the full test battery and
    measuring execution time and peak memory usage.
    """
    device = get_device()
    print(f"--- Starting profiling for model version: '{version_str}' ---")

    # 1. Initialize Model
    model = get_model(version_str)

    # 2. Reset memory stats and start timer
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()

    # 3. Run the full 11-pillar test battery
    run_pillar_tests(model, pillars_to_run=range(1, 12))

    # 4. Stop timer and get memory stats
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    peak_memory_bytes = 0
    if torch.cuda.is_available():
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)

    # 5. Print Summary
    print("\n" + "="*50)
    print("           PROFILING SUMMARY             ")
    print("="*50)
    print(f"Model Version:    {version_str}")
    print(f"Wall-Clock Time:  {total_time:.2f} seconds")
    print(f"Peak VRAM Usage:  {peak_memory_mb:.2f} MB")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile a model's performance on the testbed.")
    parser.add_argument(
        '--model_version',
        type=str,
        required=True,
        help="Specify the model version to profile (e.g., 'v1', 'static-tfm')."
    )
    args = parser.parse_args()
    profile_model(args.model_version) 