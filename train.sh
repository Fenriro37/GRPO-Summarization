#!/bin/bash

echo "--- train.sh: Starting execution inside the container ---"

#MODEL_NAME="meta-llama/meta-Llama-3.1-8B-Instruct"
MODEL_NAME="google/gemma-3-4b-it"
ACCELERATE_CONFIG_FILE="/workspace/deepspeed_zero3.yaml"

echo "--> Launching training script on 2 GPUs with accelerate config: $ACCELERATE_CONFIG_FILE"
accelerate launch  --config_file "$ACCELERATE_CONFIG_FILE" /workspace/train.py "$@"

echo "--- train.sh: Training script finished. ---"
