#!/bin/bash

export HF_TOKEN=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

python generate_dataset.py \
    --dataset_name "gaia-benchmark/GAIA" \
    --save_folder "gaia"