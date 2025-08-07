#!/bin/bash
 
# PARAMS:
# 1: model_checkpoint
# 2: dataset
max_steps="--max_steps 5"

# run test
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 run_docker.sh $max_steps