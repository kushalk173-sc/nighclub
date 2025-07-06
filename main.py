import argparse
import torch
import importlib
import os

# --- Model Imports ---
from architecture.fluid_network_v1 import FluidNetworkV1
from architecture.fluid_network_v2 import FluidNetwork as FluidNetworkV2 # Alias for clarity
from architecture.fluid_network_v3 import FluidNetworkV3

# A dictionary to map pillar numbers to their folder names.
# This makes the dynamic import more robust.
PILLAR_MAP = {
    1: "pillar_01_audio_asr",
    2: "pillar_02_vision",
    3: "pillar_03_regime_switching_timeseries",
    4: "pillar_04_long_context_dialogue",
    5: "pillar_05_multi_hop_reasoning",
    6: "pillar_06_open_ended_analogy",
    7: "pillar_07_constraint_satisfaction",
    8: "pillar_08_graceful_degradation",
    9: "pillar_09_rapid_sensor_drift",
    10: "pillar_10_memory_retention",
    11: "pillar_11_ood_geometry_fit",
}

def get_model(version_str):
    """Dynamically selects the model based on version."""
    if version_str.lower() == 'v1':
        print("--- Loading Model Architecture: v1 (Fixed Graph ODE) ---")
        return FluidNetworkV1()
    if version_str.lower() == 'v3':
        print("--- Loading Model Architecture: v3 (Affinity-Biased Attention ODE) ---")
        return FluidNetworkV3()
    # Default to v2
    print(f"--- Loading Model Architecture: v2 (Fluid Attention ODE) ---")
    return FluidNetworkV2()

def main():
    """
    Main entry point to run the fluid network testbed.
    """
    parser = argparse.ArgumentParser(description="Run the full testbed for the Fluid Network.")
    parser.add_argument(
        '--pillar', 
        nargs='+', 
        type=int, 
        help="Which pillar(s) to run (e.g., 1 2 5). Runs all if not specified."
    )
    parser.add_argument(
        '--model_version',
        type=str,
        default='v3', # Default to the latest model
        help="Specify the model version to run ('v1', 'v2', 'v3')."
    )
    args = parser.parse_args()

    print("--- Setting up the model ---")
    model = get_model(args.model_version)
    print("--- Model setup complete ---")

    pillars_to_run = args.pillar if args.pillar else range(1, 12)

    for pillar_id in pillars_to_run:
        if pillar_id not in PILLAR_MAP:
            print(f"Pillar {pillar_id} is not defined. Skipping.")
            continue
        
        try:
            pillar_folder = PILLAR_MAP[pillar_id]
            
            # Dynamically import the tests for the requested pillar
            pillar_module_name = f"pillars.{pillar_folder}.tests"
            pillar_module = importlib.import_module(pillar_module_name)
            
            # The convention is that each pillar's test module has a `run_pillar_X_tests` function
            run_func_name = f"run_pillar_{pillar_id}_tests"
            run_func = getattr(pillar_module, run_func_name)
            
            # Run the tests
            run_func(model)

        except (ModuleNotFoundError, AttributeError) as e:
            print(f"\nCould not run tests for pillar {pillar_id}. Please ensure the directory and files are set up correctly.")
            print(f"  - Directory expected: pillars/{PILLAR_MAP.get(pillar_id, 'N/A')}")
            print(f"  - Function expected: {run_func_name}")
            print(f"Error details: {e}")
        except Exception as e:
            print(f"!!!!! An error occurred while running Pillar {pillar_id}: {e} !!!!!")

if __name__ == "__main__":
    main() 