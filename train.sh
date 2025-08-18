#!/bin/bash

echo "--- train.sh: Starting execution inside the container ---"
echo "--- train.sh: Orchestrating vLLM server and training script ---"

MODEL_NAME="meta-llama/meta-Llama-3.1-8B-Instruct"
VLLM_API_URL="http://localhost:8000"

# --- 1. Launch vLLM Server in the background on GPU 0 ---
echo "--> Launching vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model "$MODEL_NAME" --max-model-len 4096 &
VLLM_PID=$!
echo "--> vLLM server started in background with PID: $VLLM_PID"

# --- FIX: Use `trap` for robust cleanup ---
# This ensures the server is killed when the script exits for any reason.
trap 'echo "--> Shutting down vLLM server (PID: $VLLM_PID)..."; kill $VLLM_PID' EXIT

# --- FIX: Use a robust `curl` loop instead of `sleep` ---
echo "--> Waiting for vLLM server to be ready at ${VLLM_API_URL}/health ..."
while ! curl -s --fail "${VLLM_API_URL}/health"; do
    echo -n "."
    sleep 5
done
echo "" # Newline for clean logging
echo "--> Server is ready!"

# --- 3. Launch the training script on GPU 1 ---
echo "--> Launching training script on GPU 1 with args: $@"
CUDA_VISIBLE_DEVICES=1 accelerate launch \
    --num_processes=1 \
    /workspace/train.py "$@"

echo "--- train.sh: Training script finished. Trap will now clean up the vLLM server. ---"