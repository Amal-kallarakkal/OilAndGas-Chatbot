import os
from dotenv import load_dotenv

load_dotenv()  # Reads .env file into os.environ

class Config:
    NVIDIA_API_KEY:  str = os.getenv('NVIDIA_API_KEY', '')
    NVIDIA_BASE_URL: str = os.getenv('NVIDIA_BASE_URL', '')
    LLM_MODEL:       str = os.getenv('LLM_MODEL', '')
    LLM_MAX_TOKENS:  int = int(os.getenv('LLM_MAX_TOKENS', '1024'))
    LLM_TEMPERATURE: float = float(os.getenv('LLM_TEMPERATURE', '0.0'))
    DUCKDB_PATH:     str = os.getenv('DUCKDB_PATH', './db/oil_gas.duckdb')

    @classmethod
    def validate(cls):
        missing = [k for k,v in vars(cls).items()
                   if not k.startswith('_') and v == '']
        if missing:
            raise EnvironmentError(f'Missing config keys: {missing}')

cfg = Config()

