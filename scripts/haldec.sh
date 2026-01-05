#!/bin/bash

# =====================================================================================
# Configurations
# =====================================================================================
# - Set the GPU devices to use. This value can be overridden by setting the environment
#   variable when running the script (e.g., `CUDA_VISIBLE_DEVICES=1 ./caption.sh`).
# - Multiple GPUs can be specified by separating them with commas (e.g., "0,1").
# =====================================================================================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4} # need 5 at least

echo "Running captioning with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# input-jsonl-path: output captions for haldec evaluation
# output-jsonl-path: haldec evaluation result
python haldec.py \
   --model-name "OpenGVLab/InternVL3_5-30B-A3B" \
   --input-jsonl-path input_video_paths.json \
   --output-jsonl-path haldec_output.json \
   --use-sys-prompt False \
   --sys-prompt None