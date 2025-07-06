#!/usr/bin/env bash

# This script automates the process of running the performance profiler
# across all 11 implemented model architectures.

# Define all model versions to be profiled
MODELS=(
    "v1"
    "v2"
    "v3"
    "static-tfm"
    "static-mlpmix"
    "static-conv"
    "static-txl"
    "static-node"
    "perceiver-io"
    "fnet-fourier"
    "attnfree-rwkv"
)

echo "--- Starting full profiling run for all 11 models ---"
echo "====================================================="

for model_version in "${MODELS[@]}"; do
    echo ""
    echo ">>> Profiling model: ${model_version}"
    
    # Use the bootstrap script to launch the profiler
    # This ensures the environment is checked first.
    python nighclub_bootstrap.py profile_model.py --model_version "${model_version}"
    
    if [ $? -ne 0 ]; then
        echo "!!! Profiling FAILED for model: ${model_version} !!!"
    fi
    
    echo "<<< Finished profiling ${model_version}"
    echo "-----------------------------------------------------"
done

echo ""
echo "====================================================="
echo "--- Full profiling run complete. ---" 