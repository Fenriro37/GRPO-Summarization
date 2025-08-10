#!/bin/bash

PHYS_DIR="/home/maranzana/project"        
LLM_CACHE_DIR="/llms"                     
DOCKER_INTERNAL_CACHE_DIR="/llms"         

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_TOKEN="$HF_TOKEN" \
    --rm \
    --memory="30g" \
    --gpus all \
    maranzana/project \
    "/workspace/train.sh" \
    "$@" 