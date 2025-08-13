import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4-turbo-preview"

# Database Configuration
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")