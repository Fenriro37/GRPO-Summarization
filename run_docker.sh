#!/bin/bash

PHYS_DIR="/home/maranzana/project"        
LLM_CACHE_DIR="/llms"                     
DOCKER_INTERNAL_CACHE_DIR="/llms"         

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_TOKEN="$HF_TOKEN" \
    -e NCCL_DEBUG=INFO \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_SHM_DISABLE=0 \
    -e NCCL_IGNORE_DISABLED_P2P=1 \
    --rm \
    --memory="30g" \
    --gpus '"device=0,1"' \
    --shm-size=8g \
    maranzana/project \
    "/workspace/train.sh" \
    "$@" 