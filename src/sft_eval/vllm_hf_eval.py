import gc
import os
import json
import torch
import vllm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
from huggingface_hub import login
huggingface_token= "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"
login(token=huggingface_token)

def apply_chat_template_inference(example, tokenizer):
    """
    Prepare the input text for inference.
    
    Assumes that each test example contains a "question" field.
    It uses the tokenizer's chat template functionality to format the prompt
    and adds a generation prompt (so the model knows to generate an answer).
    """
    # Create a prompt with only the user message.
    # prompt = [{"content": example["question"], "role": "user"}]
    prompt = [{"role": "user", "content": example["question"]}]
    # Note: Using add_generation_prompt=True so that the tokenizer adds a placeholder for the answer.
    # text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # Append the eos token for proper termination.
    example["text"] = text + tokenizer.eos_token
    return example

def main_inference():
    # Set paths to the final checkpoint and test dataset.
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run inference with fine-tuned model')
    parser.add_argument('--model_path', type=str, 
                        default="/home/mihirparmar/multi_agent/sft_results/plan_tuned/gemma-2-2b-it",
                        help='Path to the model')
    parser.add_argument('--test_path', type=str,
                        default="/home/mihirparmar/multi_agent/data/sft/eval",
                        help='Path to test data')
    parser.add_argument('--output_path', type=str,
                        default="/home/mihirparmar/multi_agent/sft_results/plan_tuned/gemma-2-2b-it",
                        help='Path for output predictions')
    
    args = parser.parse_args()
    model_path = args.model_path
    test_path = args.test_path 
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    torch.cuda.empty_cache()

    # Load the fine-tuned model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_path, token="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", trust_code=True, device_map="auto")

    # Set pad token if it isn't defined.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = '<eos>'
    tokenizer.truncation_side = 'left'
    
    # Load the test dataset.
    raw_datasets = load_dataset("json", data_files=test_path)
    test_dataset = raw_datasets["train"]

    # Apply the chat template to format each example's prompt.
    test_dataset = test_dataset.map(lambda x: apply_chat_template_inference(x, tokenizer))
    formatted_prompts = test_dataset["text"]

    # Initialize LLM for inference
    llm_call = vllm.LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.85
    )

    sampling_params = vllm.SamplingParams(
        n=1,  # Number of responses
        temperature=0.0,
        max_tokens=4096,
        skip_special_tokens=True,
        seed=42,  # Add the random seed here
        )

    outputs = llm_call.generate(
        prompts=formatted_prompts,
        sampling_params=sampling_params
    )

    predictions = []
    for i, output in enumerate(outputs):
        prediction = {
            "question": test_dataset[i]["question"],
            "answer": test_dataset[i]["answer"],
            "prediction": output.outputs[0].text
        }
        predictions.append(prediction)
    
    # Save predictions to a JSON file.
    with open(os.path.join(output_path, 'predictions.json'), "w") as f:
        json.dump(predictions, f, indent=4)
    
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main_inference()