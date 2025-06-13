# PLAN-TUNING: Post-Training Language Models to Learn Step-by-Step Planning for Complex Problem Solving

A framework for fine-tuning large language models to generate step-by-step solution plans and execute them on mathematical benchmarks (GSM8K, MATH), including out-of-distribution evaluations. Plan-Tuning improves LLM reasoning by first fine-tuning on "plans"—structured, step-by-step outlines of problem-solving strategies—before generating final answers.

## 🗂 Project Structure

```
.
├── README.md                ← this file
├── requirements.txt         ← dependencies
├── data/
│   ├── math/
│   │   ├── processed_train_method_1.json              ← training file for method 1 in the paper
│   │   ├── processed_train_method_2.json              ← training file for method 2 in the paper
│   │   ├── processed_train_sft.json                   ← training file for baseline SFT in the paper
│   │   └── processed_test_sft_and_methods.json        ← test file for baseline SFT and proposed methods in the paper
│   ├── gsm8k/
│   │   ├── processed_train_method_1.json
│   │   ├── processed_train_method_2.json
│   │   ├── processed_train_sft.json
│   │   ├── processed_test_sft.json                    ← test file for baseline SFT in the paper
│   │   └── processed_test_methods.json                ← test file for proposed methods in the paper
│   └── test_ood/
│       ├── aime/
│       │   ├── processed_test_zero_cot.json           ← test file for baseline zero-shot CoT in the paper
│       │   ├── processed_test_method_2_gsm8k.json     ← test file for method 2 in the paper
│       │   ├── processed_test_method_2_math.json      ← test file for method 2 in the paper
│       │   └── processed_test_sft_and_methods.json    ← test file for baseline SFT and proposed methods in the paper
│       └── olympiad/
│           ├── processed_test_zero_cot.json
│           ├── processed_test_method_2_gsm8k.json
│           ├── processed_test_method_2_math.json
│           └── processed_test_sft_and_methods.json
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
git clone https://github.com/your-org/plan-tuning.git
cd plan-tuning
```

### Create a Python 3.8+ environment & install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**requirements.txt example:**

```
torch
transformers
datasets
trl
accelerate
wandb
pandas
tqdm
```

## 📦 Data Preparation

### Combined GSM8K + MATH

Use `data/both_math_and_gsm8k/processed_train_*.json` for multi-domain training.

### GSM8K & MATH

Process raw datasets into `data/gsm8k/` and `data/math/` folders.

### Out-of-Distribution (OOD) Splits

Evaluate on AIME & Olympiad: `data/test_ood/{aime,olympiad}/`.

*(Add or update scripts in `src/data_processing.py` as needed.)*

## 🎓 Supervised Fine-Tuning (Plan-Tuning)

Train an LLM on plan-annotated examples:

```bash
cd src/sft_training
bash run_sft.sh \
  --data_path ../../data/both_math_and_gsm8k/processed_train_sft.json \
  --model_name_or_path "Qwen/Qwen3-8B" \
  --output_dir   ../models/plan_tuned_combo \
  --name          plan_tuned_combo \
  --batch_size    2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs    3 \
  --max_seq_len    4096 \
  --wandb_project  plan_tuning
```

Use `run_sft.py --help` for full flag list.

## 📝 Plan Generation

Generate plans & answers:

### Best-of-n:

```bash
python src/best_of_n/solve_gsm8k.py \
  --model_path ../models/plan_tuned_combo \
  --output_dir  ../outputs/gsm8k_bestof5 \
  --num_samples 5
```

### Self-Consistency:

```bash
python src/best_of_n/self_consistency.py \
  --input  ../outputs/gsm8k_bestof5/plans.json \
  --output ../outputs/gsm8k_sc.json
```

## 🧮 Evaluation

Evaluate generated outputs:

```bash
cd src/sft_eval
bash run_eval.sh \
  --predictions ../../outputs/gsm8k_sc.json \
  --references ../../data/gsm8k/processed_test_methods.json \
  --output    ../eval_results/gsm8k_metrics.json
```

Or directly:

```bash
python vllm_hf_eval.py \
  --pred_file ../../outputs/gsm8k_sc.json \
  --ref_file  ../../data/gsm8k/processed_test_methods.json
```

## 📊 Results & Logging

- Logs & metrics tracked with Weights & Biases (set `--wandb_project`).
- Local logs: `src/sft_training/wandb/`, `src/sft_eval/logs/`.

## 🤝 Contributing

1. Fork & clone
2. Create a branch: `git checkout -b feature/your_feature`
3. Commit & push
4. Open a PR

Please adhere to existing code style and add tests/examples for new features.

## 📄 License

This project is licensed under the Apache 2.0 License. See LICENSE for details.

## 📖 Citation

If you use this code, please cite:

```bibtex
@inproceedings{parmar2025plan,
  title     = {Plan-Tuning: Fine-Tuning Language Models via Step-by-Step Planning},
  author    = {Parmar, Mihir and …},
  booktitle = {NeurIPS 2025},
  year      = {2025}
}
```

Happy plan-tuning! 🚀
