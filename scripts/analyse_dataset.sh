#!/bin/bash

export HF_TOKEN=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

python analyse_dataset.py \
    --dataset_name "gaia-benchmark/GAIA" \
    --save_folder "gaia"