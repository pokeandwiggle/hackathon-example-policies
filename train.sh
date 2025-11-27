#!/bin/bash
# Wrapper script to run training with correct library paths for torchcodec

# Set library path to include PyTorch libraries
export LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"

# Run training with all arguments passed through
uv run python src/example_policies/train.py "$@"
