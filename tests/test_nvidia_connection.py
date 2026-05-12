from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage
from src.config import cfg

def test_nvidia_connection():
    llm = ChatNVIDIA(
        api_key = cfg.NVIDIA_API_KEY,
        base_url = cfg.NVIDIA_BASE_URL,
        model = cfg.LLM_MODEL,
        max_tokens = cfg.LLM_MAX_TOKENS,
        temperature = cfg.LLM_TEMPERATURE
    )

    response = llm.invoke([HumanMessage(content = "response: ok")])
    print(f'Model : {cfg.LLM_MODEL}')
    print(f'Response: {response.content}')
    assert len(response.content) > 0, 'Empty response from NVIDIA API'
    print('PASS: NVIDIA API connection successful and response received.')

if __name__ == "__main__":
    test_nvidia_connection()