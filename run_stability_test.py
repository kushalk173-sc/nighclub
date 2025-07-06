import argparse
import subprocess
import json
import numpy as np
import sys

def run_stability_test(model_version, num_runs=3):
    """
    Runs the test suite for a given model multiple times with different seeds
    and reports the mean and standard deviation of the results.
    """
    print("="*60)
    print(f"  Running Stability Test for Model: '{model_version}' ({num_runs} runs)  ")
    print("="*60)

    all_pillar_scores = {} # {pillar_id: [run1_score, run2_score, ...]}

    for i in range(num_runs):
        seed = 42 + i
        print(f"\n--- Starting Run {i+1}/{num_runs} (Seed: {seed}) ---")
        
        command = [
            sys.executable,
            "nighclub_bootstrap.py",
            "main.py",
            "--model_version", model_version,
            "--seed", str(seed),
            "--json_output"
        ]
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            
            # Find the JSON output block
            json_output = None
            for line in result.stdout.splitlines():
                if line.strip().startswith('{'):
                    json_output = line
                    break
            
            if not json_output:
                print("ERROR: Could not find JSON output from main.py script.", file=sys.stderr)
                continue

            run_results = json.loads(json_output)
            
            # Aggregate the average scores from each pillar
            for pillar_id, pillar_data in run_results.items():
                pillar_id = int(pillar_id)
                # Find the key for the average score (e.g., 'average_wer', 'average_accuracy')
                avg_key = next((key for key in pillar_data if key.startswith('average_')), None)
                if avg_key:
                    if pillar_id not in all_pillar_scores:
                        all_pillar_scores[pillar_id] = []
                    all_pillar_scores[pillar_id].append(pillar_data[avg_key])

        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"!!! ERROR during run {i+1} for model '{model_version}' !!!", file=sys.stderr)
            print(f"Error details: {e}", file=sys.stderr)
            if hasattr(e, 'stderr'):
                print(f"Stderr: {e.stderr}", file=sys.stderr)
            continue
            
    print("\n" + "="*60)
    print("           STABILITY TEST SUMMARY            ")
    print("="*60)
    print(f"Model: {model_version}")
    print(f"Number of runs: {num_runs}\n")

    for pillar_id, scores in sorted(all_pillar_scores.items()):
        if not scores: continue
        mean = np.mean(scores)
        std = np.std(scores)
        # Higher is better for accuracy/f1, lower is better for wer/mae
        metric_type = "acc/f1" if pillar_id in [2, 4, 5] else "wer/mae"
        print(f"Pillar {pillar_id:02d}: Mean Score = {mean:.4f}, Std Dev = {std:.4f}  ({metric_type})")

    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stability tests for a model.")
    parser.add_argument(
        '--model_version',
        type=str,
        required=True,
        help="Specify the model version to test (e.g., 'v1', 'static-tfm')."
    )
    parser.add_argument(
        '--num_runs',
        type=int,
        default=3,
        help="Number of runs with different seeds."
    )
    args = parser.parse_args()
    run_stability_test(args.model_version, args.num_runs) 