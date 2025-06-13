from llm import call_gemini
import re

class SelfConsistencyEvaluator:
    def __init__(self, verification_prompt=None, model_name=None):
        self.verification_prompt = verification_prompt
        self.model_name = model_name
        self.evaluate_system_prompt = """Provide a reward score between -100 and 100 for the plan quality, using very strict standards. Do not give a full score above 95. Make sure the reward score is an integer. 

Input: 
{input}
Generated plan to evaluate: 
{response}

Consider below constraints while evaluating:
{verification_prompt}

Make sure to check all the constraints before giving the reward score.

Please provide a reward in the below format:
[step-by-step reasoning for the reward score]
Score: [Strictly provide the reward score as an integer between -100 and 100]"""

    def extract_score(self, reward_output: str) -> int:
        try:
            # Try to find the score in the expected format
            match = re.search(
                r'(?i)^\s*(?:\*\*)?Score:[:\s]*(\d{1,3})(?:\*\*)?\s*$',
                reward_output,
                flags=re.MULTILINE
            )
            if match:
                reward = int(match.group(1))
            else:
                # Try fallback: find the last valid-looking number after 'Score:' anywhere
                match = re.search(r'(?i)Score:\s*(\d{1,3})(?!\.)', reward_output)
                reward = int(match.group(1)) if match else -100

            return max(-100, min(100, reward))
        except Exception:
            return -100

    def calculate_reward(self, prompt, output):
        # Get reward score from Gemini
        reward_output = call_gemini(
            self.evaluate_system_prompt.format(
                input=prompt,
                response=output,
                verification_prompt=self.verification_prompt
            ),
            model_name=self.model_name
        )
        
        # Parse reward and ensure it's an integer between -100 and 100
        reward = self.extract_score(reward_output)
        
        return (output, reward_output, reward)

    def get_best_response(self, prompt, no_of_times=5):
        lst_of_resp = []

        for i in range(no_of_times):
            print(f"Running iteration {i+1} of {no_of_times}")
            for attempt in range(3):
                try:
                    output = call_gemini(prompt, temperature=0.7, model_name=self.model_name)
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"Failed after 3 attempts: {e}")
                        output = ""
                    else:
                        print(f"Attempt {attempt + 1} failed: {e}")
            out_with_reward = self.calculate_reward(prompt, output)
            lst_of_resp.append(out_with_reward)

        # Sort responses by reward score and get the best one
        sorted_responses = sorted(lst_of_resp, key=lambda x: x[2], reverse=True)
        # sorted_responses = sorted(lst_of_resp, key=lambda x: x[2], reverse=False)
        best_response = sorted_responses[0][0]
        
        return sorted_responses, best_response