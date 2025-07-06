# nighclub_bootstrap.py
import sys
import pathlib
import subprocess
import os

# --- Dependency Check ---
# Before doing anything else, verify that the core dependencies are available.
# This prevents a cascade of errors if the wrong environment is active.
try:
    import torch
    import transformers
    import timm
    import torchdiffeq
except ImportError as e:
    print("="*60, file=sys.stderr)
    print("           ! SETUP ERROR: MISSING DEPENDENCIES !            ", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"\nCould not import a required library: {e}", file=sys.stderr)
    print("\nThis almost always means the correct Conda environment is not active.", file=sys.stderr)
    print("Please set up and activate the 'nightclub-env' environment.", file=sys.stderr)
    print("\n--- SETUP INSTRUCTIONS ---")
    print("1. Create the environment (only needs to be done once):")
    print("   conda env create -f env.yml\n")
    print("2. Activate the environment (must be done in every new terminal):")
    print("   conda activate nightclub-env\n")
    sys.exit(1)


# Add the repository root to the Python path.
# This ensures that all modules can be imported correctly, regardless of where
# the script is called from.
repo_root = pathlib.Path(__file__).resolve().parent
sys.path.append(str(repo_root))

print(f"--- Bootstrap: Added {repo_root} to PYTHONPATH.")

# Get the target script to run (e.g., 'main.py') and its arguments.
if len(sys.argv) < 2:
    print("Bootstrap Error: Please specify a script to run after the bootstrap.", file=sys.stderr)
    print("Example: python nighclub_bootstrap.py main.py --pillar 1", file=sys.stderr)
    sys.exit(1)

target_script_path = sys.argv[1]
arguments = sys.argv[2:]

# Construct the full command to execute.
command = [sys.executable, target_script_path] + arguments

print(f"--- Bootstrap: Executing command: {' '.join(command)}")
print("-" * 50)

# Run the target script as a subprocess.
try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"\n--- Bootstrap: Target script '{target_script_path}' exited with error (Code: {e.returncode}).", file=sys.stderr)
    sys.exit(e.returncode)
except FileNotFoundError:
    print(f"\n--- Bootstrap: Target script '{target_script_path}' not found.", file=sys.stderr)
    sys.exit(1)

print("-" * 50)
print(f"--- Bootstrap: Target script '{target_script_path}' finished successfully.") 