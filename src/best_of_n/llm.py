import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, SafetySetting
from retrying import retry

PROJECT_ID = "research-01-268019"
REGIONS = ["us-central1", "asia-east1"]
vertexai.init(project=PROJECT_ID, location=REGIONS[0])

# Define safety settings as a list of SafetySetting objects
safety_settings = [
    SafetySetting(category=HarmCategory.HARM_CATEGORY_UNSPECIFIED,threshold="BLOCK_NONE"),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold="BLOCK_NONE"),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold="BLOCK_NONE"),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold="BLOCK_NONE"),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold="BLOCK_NONE")
]

@retry(wait_exponential_multiplier=16000, wait_exponential_max=256000, stop_max_attempt_number=5)
def call_gemini(prompt, model_name, temperature=0, candidate_count=None, stop_sequences=None, response_type=None, response_schema=None):
    try:
        attempts = getattr(call_gemini, 'retry_count', 0) + 1
        
        model = GenerativeModel(model_name)
        generation_config = GenerationConfig(temperature=temperature, candidate_count= candidate_count, stop_sequences=stop_sequences, response_mime_type=response_type, response_schema=response_schema)
        
        response = model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)

        if hasattr(response, 'text'):
            return response.text
        else:
            return " "

    except Exception as e:
        print(f"Error occurred on attempt {attempts}: {e}")
        raise