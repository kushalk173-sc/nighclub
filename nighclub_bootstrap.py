# nighclub_bootstrap.py
import sys
import pathlib
import subprocess
import os

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