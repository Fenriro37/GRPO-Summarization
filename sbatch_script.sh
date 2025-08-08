#!/bin/bash
 
# PARAMS:
# 1: model_checkpoint
# 2: dataset
MAX_STEPS=5
DATASET_PATH="/workspace/small_dataset.json"

sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 run_docker.sh --max_steps $MAX_STEPS --dataset_path $DATASET_PATH
