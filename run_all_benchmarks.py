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
    total_models = len(ALL_MODELS)

    for model_idx, model_version in enumerate(ALL_MODELS, 1):
        print(f"\n[{model_idx}/{total_models}] --- Benchmarking Model: {model_version} ---")
        print(f"  Running 11 pillars × 10 tests = 110 tests for {model_version}...")
        
        command = [
            sys.executable,
            "nighclub_bootstrap.py",
            "main.py",
            "--model_version", model_version,
            "--seed", str(seed),
            "--json_output"
        ]
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')
            if result.stdout is None:
                print(f"stdout is None for {model_version}")
                print(f"stderr: {result.stderr}")
                raise Exception("Subprocess stdout is None - likely encoding error")
            
            # Find the JSON output block
            json_output = None
            lines = result.stdout.splitlines()
            for i, line in enumerate(lines):
                if line.strip().startswith('{'):
                    # Found the start of JSON, collect all lines until we find a complete JSON
                    json_lines = []
                    brace_count = 0
                    for j in range(i, len(lines)):
                        json_lines.append(lines[j])
                        brace_count += lines[j].count('{') - lines[j].count('}')
                        if brace_count == 0:
                            # We've found a complete JSON object
                            json_output = '\n'.join(json_lines)
                            break
                    break
            
            if json_output:
                try:
                    parsed_results = json.loads(json_output)
                    all_results[model_version] = parsed_results
                    print(f"  ✓ Successfully parsed JSON for {model_version}")
                except json.JSONDecodeError as e:
                    print(f"!!! ERROR during benchmark for model '{model_version}' !!!")
                    print(f"JSON Decode Error: {e}")
                    print(f"Exception details: {e} ")
                    all_results[model_version] = {"error": f"JSON decode failed: {str(e)}"}
            else:
                print(f"!!! ERROR during benchmark for model '{model_version}' !!!")
                print("No JSON output found in stdout")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                all_results[model_version] = {"error": "No JSON output found"}
                
        except subprocess.CalledProcessError as e:
            print(f"!!! ERROR during benchmark for model '{model_version}' !!!")
            print(f"Subprocess failed with return code {e.returncode}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            all_results[model_version] = {"error": f"Subprocess failed: {str(e)}"}
        except Exception as e:
            print(f"!!! ERROR during benchmark for model '{model_version}' !!!")
            print(f"Unexpected error: {e}")
            all_results[model_version] = {"error": f"Unexpected error: {str(e)}"}

    print(f"\n{'='*70}")
    print(f"  COMPLETED: {len(all_results)}/{total_models} models benchmarked")
    print(f"{'='*70}")
    print_results_table(all_results)

def print_results_table(results):
    """
    Prints a formatted markdown table of the benchmark results.
    """
    print("\n" + "="*70)
    print("                 FULL BENCHMARK RESULTS                 ")
    print("="*70)

    # Create a DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')
    df = df.reindex(sorted(df.columns), axis=1) # Sort columns by pillar number

    # Format header - handle both string and integer keys
    header = []
    for col in df.columns:
        try:
            pillar_num = int(col)
            metric = PILLAR_METRICS.get(pillar_num, "Unknown")
            header.append(f"Pillar {pillar_num} ({metric})")
        except ValueError:
            # If it's not a number, use it as is
            header.append(col)
    df.columns = header
    
    # Format floating point numbers
    for col in df.columns:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

    # Add an index name
    df.index.name = "Model"

    # Print the table in markdown format
    print(df.to_markdown())
    
    print("\n" + "="*70)
    print("Lower is better for WER/MAE. Higher is better for Acc/F1.")
    print("="*70)

if __name__ == "__main__":
    run_all_benchmarks() 