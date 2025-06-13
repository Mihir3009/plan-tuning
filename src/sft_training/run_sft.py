# from transformers import GPT2TokenizerFast, GPT2LMHeadModel, AutoModelForCausalLM
from transformers import AutoTokenizer
import logging
import os
import sys
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import torch
import json
# import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from evaluate import load as load_metric
import ipdb
# CUDA_LAUNCH_BLOCKING=1
import os
# os.environ["WANDB_DISABLED"] = "true"
# from sklearn.metrics import classification_report
# from peft import PeftConfig
import transformers
from filelock import FileLock
from transformers import pipeline as pi
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments, 
    Trainer,
    set_seed,
)
from typing import List, Literal, Optional
# from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import gc
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
# from trl import DPOTrainer, DPOConfig
# from tpo_trainer import TPOTrainer
# from tpo_config import TPOConfig

from trl import SFTConfig #SFTTrainer
from sft_trainer import SFTTrainer
from data_collator import DataCollatorForCompletionOnlyLM

# from peft import LoraConfig
# from trl import DPOTrainer
# from utils.utils import FDivergenceType
# from utils.tpo_trainer import TPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from datasets import load_from_disk

import wandb


# DEFAULT_CHAT_TEMPLATE = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
# DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
# MISTRAL_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
# LLAMA_CHAT_TEMPLATE = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"],
):

    if task in ["sft", "generation"]:
        # prefix = [{"content": example['question'], "role": "user"},{'content':example['answer'], 'role': 'assistant'}]
        prefix = [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]}
            ]
        prefix = tokenizer.apply_chat_template(prefix, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        text = prefix+tokenizer.eos_token
        example["text"] = text

    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )
    return example


def main():

    parser = HfArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the training data JSON file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path or name of the pretrained model')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Directory to save the trained model and results')
    parser.add_argument('--name', type=str, required=True,
                        help='Name of the training run')
    parser.add_argument('--model_type', type=str, required=True,
                        help='Type of the model to be trained')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Maximum sequence length')
    parser.add_argument('--wandb_name', type=str, default='sft-training',
                        help='Name for the Weights & Biases run')
    
    args = parser.parse_args()
    wandb.init(project="plangen-training", name=args.wandb_name)

    # raw_datasets = load_dataset("openai/gsm8k", "main") #DATASET NAME
    # raw_datasets = raw_datasets["train"]

    # # Load JSON data
    # with open("/home/mihirparmar/multi_agent/data/sft/train/train.json", "r") as f:
    #     json_data = json.load(f)

    # # Convert JSON to Hugging Face Dataset format
    # raw_datasets = Dataset.from_dict({key: [d[key] for d in json_data["data"]] for key in json_data["data"][0].keys()})

    raw_datasets = load_dataset('json', data_files=args.data_path)
    raw_datasets = raw_datasets["train"]

    column_names = list(raw_datasets.features)

    print(raw_datasets)

    path =  args.model_path

    model = AutoModelForCausalLM.from_pretrained(path, attn_implementation="eager", token="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", torch_dtype=torch.bfloat16, device_map="auto") #attn_implementation="flash_attention_2"
    tokenizer = AutoTokenizer.from_pretrained(path, token="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", trust_code=True, device_map="auto") #device_map="auto"
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = '<eos>'
    # if tokenizer.model_max_length > 100_000:
    #     tokenizer.model_max_length = 2048
    # if tokenizer.chat_template is None and tokenizer.default_chat_template is None:
        # print("MISTRAL_CHAT_TEMPLATE activate")
    # tokenizer.chat_template = LLAMA_CHAT_TEMPLATE
    print("The Pad token is", tokenizer.decode(tokenizer.pad_token_id))
    print("The eos token is", tokenizer.decode(tokenizer.eos_token_id))
    tokenizer.truncation_side = 'left'


    #####################
    # Apply chat template
    #####################

    # ipdb.set_trace()
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "sft"},
        num_proc=12,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )
    # ipdb.set_trace()
    
    raw_datasets = raw_datasets.train_test_split(test_size=0.05, seed=42)

    name = args.name  # b: batch | l: learning rate | a = alpha | t = beta | e = epoch if there is no e in the name means e = 1
    model_type = args.model_type


    output_dir = f'{args.output_path}/{model_type}/{name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.config.use_cache = False

    # 4. initialize training arguments:
    training_args = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=5,
        save_steps=None,  # Prevents intermediate checkpoint saving
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # should be 16
        gradient_checkpointing=True,
        learning_rate=5.0e-6,
        do_eval=True,
        eval_steps=200,  # Continue evaluating every 200 steps
        output_dir=output_dir,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        run_name=name,
        report_to="wandb",
        bf16=True,
        log_level='info',
        num_train_epochs=args.num_epochs,
        save_total_limit=1,  # Limits saved checkpoints to 1
        save_strategy="no",  # Disables automatic saving
        seed=42,
        warmup_ratio=0.1,
        dataset_text_field="text",
        max_seq_length=args.max_length,
    )


    # response_template = "<start_of_turn>model\n"
    response_template = "<|im_start|>assistant\n"
    # response_template = "Assistant:"

    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=False)

    sft_trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=raw_datasets['train'],
        eval_dataset=raw_datasets['test'],
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # 6. train
    torch.cuda.empty_cache()
    sft_trainer.train()
    sft_trainer.save_model(output_dir)

    # 7. save
    # output_dir = os.path.join(output_dir, "final_checkpoint")
    sft_trainer.model.save_pretrained(output_dir)



if __name__ == "__main__":
    main()