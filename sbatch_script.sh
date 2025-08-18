#!/bin/bash
export HF_TOKEN="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"

# PARAMS:
# 1: model_checkpoint
# 2: dataset
MAX_STEPS=5
DATASET_PATH="/workspace/dataset_with_len_filtered.json"
MAX_LEN=4096
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:2 run_docker.sh --max_steps $MAX_STEPS --dataset_path $DATASET_PATH --max_seq_length $MAX_LEN
