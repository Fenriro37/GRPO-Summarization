#!/bin/bash

echo "--- Minimal Test: Starting vLLM server and trainer ---"

# --- 1. Launch vLLM server on GPU 0 in the background ---
# Model name is hardcoded for this test.
echo "--> Launching vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model "meta-llama/meta-Llama-3.1-8B-Instruct" &

# Capture the server's Process ID (PID) to kill it later
VLLM_PID=$!

trap 'echo "--> Shutting down vLLM server..."; kill $VLLM_PID' EXIT

# --- 2. Wait for the server to be ready ---
# This loop is essential. The script will not proceed until the server is online.
echo "--> Waiting for vLLM server to respond..."
while ! curl -s --fail "http://localhost:8000/health"; do
    sleep 2 # Wait 2 seconds before checking again
done
echo "--> Server is ready!"


echo "--> Launching training script on GPU 1..."
CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 /workspace/train.py "$@"

echo "--- Minimal Test: Training finished. ---"