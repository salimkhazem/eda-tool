import os
from dotenv import load_dotenv
from pathlib import Path

class Config:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Azure OpenAI settings
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.azure_api_version = os.getenv("AZURE_API_VERSION")
        
        # EDA settings
        self.output_dir = os.getenv("DEFAULT_OUTPUT_DIRECTORY", "eda_output")
        self.eda_depth = os.getenv("EDA_DEPTH", "basic")
        self.max_rows_sample = int(os.getenv("MAX_ROWS_SAMPLE", "5"))
        self.max_prompt_tokens = int(os.getenv("MAX_PROMPT_TOKENS", "4000"))
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
