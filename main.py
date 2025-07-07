import argparse
import torch
import importlib
import os
import random
import numpy as np
import json
import sys

# --- Model Imports ---
from architecture.fluid_network_v1 import FluidNetworkV1
from architecture.fluid_network_v2 import FluidNetwork as FluidNetworkV2 # Alias for clarity
from architecture.fluid_network_v3 import FluidNetworkV3
from architecture.baselines import BaselineHostNetwork

# --- Static Core Imports ---
from architecture.static_cores.static_transformer import StaticTransformerCore
from architecture.static_cores.static_mlpmixer import StaticMlpMixerCore
from architecture.static_cores.static_conv1d import StaticConv1DCore
from architecture.static_cores.static_transformer_xl import StaticTransformerXLCore
from architecture.static_cores.static_node import StaticNodeCore
from architecture.static_cores.perceiver_io import PerceiverIOCore
from architecture.static_cores.fnet_fourier import StaticFNetCore
from architecture.static_cores.attnfree_rwkv import StaticRwkvCore

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

SUPPORTED_MODELS = {
    'v1', 'v2', 'v3', 
    'static-tfm', 'static-mlpmix', 'static-conv', 'static-txl', 
    'static-node', 'perceiver-io', 'fnet-fourier', 'attnfree-rwkv'
}

def set_seed(seed):
    """Sets the random seed for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # Not strictly necessary, but good for full reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"--- Seed set to {seed} for reproducibility. ---")

def get_model(version_str):
    """Dynamically selects the model based on version."""
    version_key = version_str.lower()
    if version_key not in SUPPORTED_MODELS:
        raise ValueError(f"Model version '{version_str}' is not supported. Please choose from: {list(SUPPORTED_MODELS)}")

    # Fluid Models
    if version_key == 'v1':
        print("--- Loading Model Architecture: v1 (Fixed Graph ODE) ---")
        return FluidNetworkV1()
    if version_key == 'v2':
        print(f"--- Loading Model Architecture: v2 (Fluid Attention ODE) ---")
        return FluidNetworkV2()
    if version_key == 'v3':
        print("--- Loading Model Architecture: v3 (Affinity-Biased Attention ODE) ---")
        return FluidNetworkV3()

    # Static Baseline Models
    if version_key == 'static-tfm':
        core = StaticTransformerCore()
        return BaselineHostNetwork(core, model_name="Static Transformer")
    if version_key == 'static-mlpmix':
        core = StaticMlpMixerCore()
        return BaselineHostNetwork(core, model_name="Static MLP-Mixer")
    if version_key == 'static-conv':
        core = StaticConv1DCore()
        return BaselineHostNetwork(core, model_name="Static Conv1D")
    if version_key == 'static-txl':
        core = StaticTransformerXLCore()
        return BaselineHostNetwork(core, model_name="Static Transformer-XL")
    if version_key == 'static-node':
        core = StaticNodeCore()
        return BaselineHostNetwork(core, model_name="Static NODE (Frozen v1)")
    if version_key == 'perceiver-io':
        core = PerceiverIOCore()
        return BaselineHostNetwork(core, model_name="Perceiver-IO")
    if version_key == 'fnet-fourier':
        core = StaticFNetCore()
        return BaselineHostNetwork(core, model_name="FNet Fourier")
    if version_key == 'attnfree-rwkv':
        core = StaticRwkvCore()
        return BaselineHostNetwork(core, model_name="Attention-Free RWKV")
        
    # Fallback, should not be reached due to the check above
    raise NotImplementedError(f"Model version '{version_key}' is defined but not implemented in the factory.")

def run_pillar_tests(model, pillars_to_run):
    """
    Runs all tests for a given list of pillars and returns the results.
    """
    all_results = {}
    total_pillars = len(pillars_to_run)
    
    for pillar_idx, pillar_id in enumerate(pillars_to_run, 1):
        pillar_name = f"pillar_{pillar_id:02d}"
        print(f"\n[{pillar_idx}/{total_pillars}] Running Pillar {pillar_id}...")
        
        if pillar_id not in PILLAR_MAP:
            print(f"Pillar {pillar_id} is not defined. Skipping.")
            continue
            
        try:
            pillar_module_name = PILLAR_MAP[pillar_id]
            pillar_test_module = importlib.import_module(f"pillars.{pillar_module_name}.tests")
            
            # This runs all 10 tests within the pillar's `tests.py` file
            print(f"  - Executing 10 tests for Pillar {pillar_id}...")
            pillar_results = pillar_test_module.run_all_tests(model)
            all_results[pillar_id] = pillar_results
            print(f"  ✓ Completed Pillar {pillar_id}")

        except ImportError as e:
            print(f"Could not import or run tests for Pillar {pillar_id}. Error: {e}", file=sys.stderr)
            # Return a default result structure for failed pillars
            all_results[pillar_id] = {"error": f"Import failed: {str(e)}"}
        except Exception as e:
            print(f"!!!!! An error occurred while running Pillar {pillar_id}: {e} !!!!!", file=sys.stderr)
            # Return a default result structure for failed pillars
            all_results[pillar_id] = {"error": f"Runtime error: {str(e)}"}
            
    return all_results

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
        default='v3',
        help=f"Specify the model version to run {list(SUPPORTED_MODELS)}."
    )
    parser.add_argument(
        '--smoke',
        action='store_true',
        help="Run a quick smoke test to check for basic errors."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        '--json_output',
        action='store_true',
        help="Output final results as a JSON string."
    )
    args = parser.parse_args()
    print(f"[DEBUG] Parsed args: {args}")
    set_seed(args.seed)

    if args.smoke:
        print("--- Smoke test mode: Will run minimal checks and exit. ---")
        pillars_to_run = [1, 2, 3]  # Run first 3 pillars for quick test
    else:
        pillars_to_run = args.pillar if args.pillar else range(1, 12)
    print(f"[DEBUG] Pillars to run: {pillars_to_run}")
    print(f"[DEBUG] Model version: {args.model_version}")

    print("--- Setting up the model ---")
    try:
        model = get_model(args.model_version)
        print("--- Model setup complete ---")
    except Exception as e:
        print(f"Error setting up model: {e}", file=sys.stderr)
        if args.json_output:
            error_result = {pillar_id: {"error": f"Model setup failed: {str(e)}"} for pillar_id in pillars_to_run}
            print(f"[DEBUG] Error result JSON: {error_result}")
            print(json.dumps(error_result, indent=2))
            sys.stdout.flush()
            os._exit(0)  # Exit immediately without cleanup
        return

    final_results = run_pillar_tests(model, pillars_to_run)
    print(f"[DEBUG] Final results: {final_results}")
    if args.json_output:
        # Ensure the output is valid JSON by handling any non-serializable objects
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {str(k): clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        cleaned = clean_for_json(final_results)
        print(json.dumps(cleaned, indent=2))
        os._exit(0)  # Exit immediately without any cleanup
    else:
        # Existing pretty-printing logic would go here, for now just print dict
        print("\n--- Test Run Complete ---")
        print(final_results)

if __name__ == "__main__":
    main() 