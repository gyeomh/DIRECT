#!/usr/bin/env bash
set -uo pipefail
source /home/gyeom/miniconda3/bin/activate vllm_env
export HF_HOME=/data/gyeom/hf_cache
export CUDA_VISIBLE_DEVICES=4
exec vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt.video 0 \
  --port 8020
