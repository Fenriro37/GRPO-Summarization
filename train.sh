#!/bin/bash

echo "--- train.sh: Starting execution inside the container ---"
echo "--- train.sh: Current directory is $(pwd) ---"
echo "--- train.sh: Contents of /workspace: ---"
ls -l /workspace
echo "-----------------------------------------------------"
echo "--- train.sh: Executing python script with args: $@ ---"

python3 /workspace/train.py "$@"