import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MOCK_OPENAI = os.getenv("MOCK_OPENAI", "0") == "1"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
