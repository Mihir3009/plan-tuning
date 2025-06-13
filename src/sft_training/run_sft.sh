#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,3

export HF_HOME="/home/mihirparmar/.cache"

# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ./accelerate_configs/deepspeed_zero3.yaml --main_process_port 29501 run_sft.py
python run_sft.py \
    --data_path /home/mihirparmar/plangen_training/data/gsm8k/processed_train_1.json \
    --model_path "Qwen/Qwen3-8B" \
    --output_path /home/mihirparmar/plangen_training/sft_models \
    --name plan_tuned_gsm8k_1 \
    --model_type qwen-8b \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_epochs 1 \
    --max_length 3500 \
    --wandb_name plan_tuned_gsm8k_1