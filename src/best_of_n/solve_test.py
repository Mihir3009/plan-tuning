import os
from os.path import join
from tqdm import tqdm
import time
from llm import call_gemini
import json
from self_consistency import SelfConsistencyEvaluator
model_name = "gemini-2.5-pro-preview-03-25"

def save_predictions(predictions, save_path):
    with open(join(save_path, 'predictions.json'), 'w') as json_file:
        json.dump(predictions, json_file, indent=4, ensure_ascii=False)

def main():
    # Load JSON file from the given path
    json_file_path = "/home/mihirparmar/plangen_training/data/gsm8k/train.json"
    with open(json_file_path, "r") as f:
        data = json.load(f)
    
    results_dir = "/home/mihirparmar/plangen_training/planning_results/test"
    os.makedirs(results_dir, exist_ok=True)

    final_predictions = []
    # Iterate over the dataset with a progress bar
    for idx, data_instance in enumerate(tqdm(data, desc="Plan Generation Started", total=len(data))):

        question_str = "A farmer with a goat, and a cabbage must cross a river by boat. The boat can carry only the farmer and two items. If left unattended together, the wolf would eat the goat, or the goat would eat the cabbage. How can they cross the river without anything being eaten?"

        # First stage: Generate a plan
        plan_prompt = """Analyze the given question; and create a plan to solve it:

<question>
{question}
</question>

Feel free to break down the problem in whatever way you think is most effective. Consider key concepts, formulas, relevant facts, or any logical approach that would help solve this. Your task is to only provide a plan and not solving it during this process."""

        problem_statement = plan_prompt.format(question=question_str)

        verification_prompt = """You are an expert in identifying explicit and implicit constraints for verifying plans generated to solve complex problems. Your job is to generate those constraints for the following question which can be helpful in verifying and evaluating the given plan. 
			
<question>
{question}
</question>

Make sure to identify all constraints in the question. Please output the constraints as a list. DO NOT include any other text in your response."""
        
        verification_prompt = verification_prompt.format(question=question_str)

        generated_constraints = call_gemini(verification_prompt, model_name=model_name)

        evaluator = SelfConsistencyEvaluator(verification_prompt= generated_constraints, model_name=model_name)
        
        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                responses, best_answer = evaluator.get_best_response(problem_statement, no_of_times=5)
                break
            except Exception as e:
                print(f"Error during Best of N run (attempt {retry_count + 1}): {e}")
                retry_count += 1
                if retry_count == max_retries:
                    print("All retry attempts failed")
                    responses = []
                    best_answer = ""
                else:
                    print(f"Retrying... ({retry_count + 1}/{max_retries})")
                    time.sleep(2)  # Add small delay between retries
        
        # Second stage: Execute the plan
        predictions= []
        for response in responses:
            # Second stage: Execute the plan
            solve_prompt = """Using the generated plan:

    <plan>        
    {plan}
    </plan>

    Now solve the below question by following the plan step by step:
    <question>
    {question}
    </question>

    Format your response as follows:
    Plan Execution: [Add your step-by-step plan execution to solve the question]
    Answer: [provide your final answer]"""

            solve_prompt = solve_prompt.format(plan=response[0], question=question_str)
            prediction = call_gemini(solve_prompt, model_name=model_name)
            predictions.append({"plan": response[0], 
                                "prediction": prediction, 
                                "score": response[2], 
                                "reason": response[1]})

        data_instance["model_plan"] = best_answer
        data_instance["model_prediction"] = predictions
        data_instance["sc_outputs"] = responses
        data_instance["constraints"] = generated_constraints

        final_predictions.append(data_instance)

        save_predictions(final_predictions, results_dir)
        exit()

    save_predictions(final_predictions, results_dir)
    
if __name__ == "__main__":
    main()