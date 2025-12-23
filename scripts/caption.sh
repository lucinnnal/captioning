#!/bin/bash

# =====================================================================================
# Configurations
# =====================================================================================
# - Set the GPU devices to use. This value can be overridden by setting the environment
#   variable when running the script (e.g., `CUDA_VISIBLE_DEVICES=1 ./caption.sh`).
# - Multiple GPUs can be specified by separating them with commas (e.g., "0,1").
# =====================================================================================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,5,6,7}

echo "Running captioning with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Run the python script, passing all command-line arguments to it.
#
# Example usage:
# ./caption.sh \
#   --input-json-path "example_video_paths.json" \
#   --output-json-path "results.json" \
#   --model-name "OpenGVLab/InternVL3_5-8B"
#
python captioning.py \
   --model-name "OpenGVLab/InternVL3_5-30B-A3B" \
   --input-json-path input_video_paths.json \
   --output-json-path output_captions.json \
   --use-sys-prompt False \
   --sys-prompt "You are an AI assistant that rigorously follows this response protocol: 1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags. 2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline. Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section." \
   --question-suffix "Please provide detailed and comprehensive captions for the video."