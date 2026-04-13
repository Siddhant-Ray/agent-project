#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct \
  --tensor-parallel-size 4 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.95 \
  --generation-config vllm