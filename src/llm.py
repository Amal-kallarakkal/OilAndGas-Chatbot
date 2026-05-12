"""
LLM client factory.
All agents import get_llm() from here.
Never instantiate ChatNVIDIA directly in agent files.
"""
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from src.config import Config

def get_llm(
        temperature: float = None,
        max_tokens: int    = None) -> ChatNVIDIA:
    """
    Return a Configured ChatNVIDIA instance.

    Args:
        temperature: Override the default from Config.
                     Use 0.0 for structured outputs (intent, plans).
                     Use 0.3-0.5 for narrative explanation.
        max_tokens:  Override the default from Config.
    """

    return ChatNVIDIA(
        model=Config.LLM_MODEL,
        api_key=Config.NVIDIA_API_KEY,
        base_url=Config.NVIDIA_BASE_URL,
        temperature=temperature if temperature is not None else Config.LLM_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else Config.LLM_MAX_TOKENS,        
    )