# PLAN-TUNING: Post-Training Language Models to Learn Step-by-Step Planning for Complex Problem Solving

A framework for fine-tuning large language models to generate step-by-step solution plans and execute them on mathematical benchmarks (GSM8K, MATH), including out-of-distribution evaluations. Plan-Tuning improves LLM reasoning by first fine-tuning on "plans"—structured, step-by-step outlines of problem-solving strategies—before generating final answers.

## 🗂 Repository Structure

```
.
├── README.md                ← this file
├── requirements.txt         ← dependencies
├── data/
│   ├── math/
│   │   ├── processed_train_method_1.json              ← training file for method 1 in the paper
│   │   ├── processed_train_method_2.json              ← training file for method 2 in the paper
│   │   ├── processed_train_sft.json                   ← training file for baseline SFT in the paper
│   │   └── processed_test_sft.json                    ← test file for baseline SFT and proposed methods in the paper
│   │   └── processed_test_method_2.json               ← test file for method 2 in the paper
│   ├── gsm8k/
│   │   ├── processed_train_method_1.json
│   │   ├── processed_train_method_2.json
│   │   ├── processed_train_sft.json
│   │   ├── processed_test_sft.json                    ← test file for baseline SFT in the paper
│   │   └── processed_test_methods.json                ← test file for proposed methods in the paper
│   └── test_ood/
│       ├── aime/
│       │   ├── processed_test_zero_cot.json           ← test file for baseline zero-shot CoT in the paper
│       │   ├── processed_test_method_2_gsm8k.json     ← test file for method 2 in the paper for GSM8k trained models
│       │   ├── processed_test_method_2_math.json      ← test file for method 2 in the paper for MATH trained models
│       │   └── processed_test_sft.json                ← test file for baseline SFT and proposed methods in the paper
│       └── olympiad/
│           ├── processed_test_zero_cot.json
│           ├── processed_test_method_2_gsm8k.json
│           ├── processed_test_method_2_math.json
│           └── processed_test_sft.json
└── src/
    ├── sft_training/
    │   ├── run_sft.sh           ← launch script for SFT
    │   ├── run_sft.py           ← entry-point for training
    │   ├── sft_trainer.py       ← custom trainer implementation
    │   ├── data_collator.py     ← batch collation logic
    │   ├── trainer.py           ← wrapper around HF Trainer
    │   └── utils.py             ← data & helper functions
    ├── best_of_n/
    │   ├── llm.py               ← LLM interface & helpers
    │   ├── solve_gsm8k.py       ← best-of-n for GSM8K
    │   ├── solve_math.py        ← best-of-n for MATH
    │   ├── solve_test.py        ← inference on test splits
    │   └── self_consistency.py  ← self-consistency decoding
    └── sft_eval/
        ├── vllm_hf_eval.py      ← evaluation via vLLM/HF
        └── run_eval.sh          ← evaluation launcher
```

## ⚙️ Installation

### Clone the repo

```bash
git clone https://github.com/Mihir3009/plan-tuning.git
cd plan-tuning
```

### Create a Python 3.9+ environment & install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 📦 Data Preparation

### GSM8K & MATH

Process raw datasets into `data/gsm8k/` and `data/math/` folders. We already provided processed files for our paper in these folders.

### Out-of-Distribution (OOD) Splits

Evaluate on AIME & OlympiadBench: `data/test_ood/{aime,olympiad}/`.

## 🎓 Supervised Fine-Tuning (Plan-Tuning)

Train an LLM on plan-annotated examples:

```bash
cd src/sft_training
bash run_sft.sh \
  --data_path ../../data/both_math_and_gsm8k/processed_train_sft.json \
  --model_path "Qwen/Qwen3-8B" [huggingface path or trained model path] \
  --output_path   [output directory] \
  --name          [name of folder for directory] \
  --batch_size    2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs    3 \
  --max_seq_len    4096 \
  --wandb_project  [project name]
```

## 📝 Plan Generation

Generate plans & answers:

### Best-of-n:

```bash
python src/best_of_n/solve_gsm8k.py \ [change paths in the file]
```

- Do the same for the MATH dataset.

## 🧮 Evaluation

Evaluate generated outputs:

```bash
python vllm_hf_eval.py \
    --model_path [huggingface path or trained model path] \
    --test_path ../../data/both_math_and_gsm8k/processed_test_sft.json \
    --output_path [output directory]
```

## 📄 License

This project is licensed under the Apache 2.0 License. See LICENSE for details.

Happy plan-tuning! 🚀
