import subprocess
import json
import sys
import pandas as pd

# List of all models to be benchmarked
ALL_MODELS = [
    'v1', 'v2', 'v3', 
    'static-tfm', 'static-mlpmix', 'static-conv', 'static-txl', 
    'static-node', 'perceiver-io', 'fnet-fourier', 'attnfree-rwkv'
]

# A mapping to remember which metric each pillar uses
# Lower is better for WER/MAE, Higher is better for Acc/F1
PILLAR_METRICS = {
    1: "WER", 2: "Acc", 3: "MAE", 4: "F1", 5: "Acc",
    6: "MAE", 7: "MAE", 8: "MAE", 9: "MAE", 10: "MAE", 11: "MAE"
}

def run_all_benchmarks():
    """
    Runs the full test suite for all supported models and compiles the results.
    """
    print("="*70)
    print("  Starting Full Benchmark Run for All 11 Models Across 110 Tests  ")
    print("="*70)

    # Use a fixed seed for a fair comparison between models
    seed = 42
    all_results = {}

    for model_version in ALL_MODELS:
        print(f"\\n--- Benchmarking Model: {model_version} ---")
        
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
                print(f"ERROR: Could not find JSON output for model '{model_version}'. Skipping.", file=sys.stderr)
                continue

            run_results = json.loads(json_output)
            model_scores = {}
            for pillar_id, pillar_data in run_results.items():
                pillar_id = int(pillar_id)
                avg_key = next((key for key in pillar_data if key.startswith('average_')), None)
                if avg_key:
                    model_scores[pillar_id] = pillar_data[avg_key]
            all_results[model_version] = model_scores
            print(f"--- Finished Benchmarking {model_version} ---")

        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"!!! ERROR during benchmark for model '{model_version}' !!!", file=sys.stderr)
            print(f"Error details: {e}", file=sys.stderr)
            if hasattr(e, 'stderr'):
                print(f"Stderr: {e.stderr}", file=sys.stderr)
            # Record failure
            all_results[model_version] = {p: "FAIL" for p in range(1, 12)}
            
    print_results_table(all_results)

def print_results_table(results):
    """
    Prints a formatted markdown table of the benchmark results.
    """
    print("\\n" + "="*70)
    print("                 FULL BENCHMARK RESULTS                 ")
    print("="*70)

    # Create a DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')
    df = df.reindex(sorted(df.columns), axis=1) # Sort columns by pillar number

    # Format header
    header = [f"Pillar {i} ({PILLAR_METRICS[i]})" for i in df.columns]
    df.columns = header
    
    # Format floating point numbers
    for col in df.columns:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

    # Add an index name
    df.index.name = "Model"

    # Print the table in markdown format
    print(df.to_markdown())
    
    print("\\n" + "="*70)
    print("Lower is better for WER/MAE. Higher is better for Acc/F1.")
    print("="*70)

if __name__ == "__main__":
    run_all_benchmarks() 