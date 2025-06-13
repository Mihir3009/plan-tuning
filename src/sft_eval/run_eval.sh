#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

export HF_HOME="/home/mihirparmar/.cache"

python vllm_hf_eval.py \
    --model_path /home/mihirparmar/plangen_training/sft_models/deepseek-7b/grpo_model_gsm8k \
    --test_path /home/mihirparmar/plangen_training/data/gsm8k/processed_test_sft.json \
    --output_path /home/mihirparmar/plangen_training/sft_results/deepseek-7b/gsm8k/vanilla_grpo