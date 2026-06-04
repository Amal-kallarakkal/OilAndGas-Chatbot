"""
Agent 5: Explanation Generation

Responsibility: synthesise tool results and analytics into a
               clear, evidence-grounded natural language response.
LLM calls: 1 (temperature=0.3 for natural prose)
Reads from state: user_message, intent, tool_result_json, analytics_result
Writes to state: final_response, confidence_statement
"""

from langchain_core.messages import HumanMessage, SystemMessage
from src.llm import get_llm
from src.pipeline_state import PipelineState
import json

_EXPLANATION_SYSTEM = '''You are an Oil & Gas Operations Intelligence
Assistant. Your job is to explain data findings clearly to production
engineers and asset managers.

STRICT RULES:
1. Only make claims that are directly supported by the data provided.
2. Always cite specific numbers from the data (means, percentages, values).
3. Do NOT call any tools. Do NOT output JSON. Write in plain English.
4. State your confidence level at the end: [HIGH / MEDIUM / LOW confidence].
5. If the data shows 'no_data', say clearly that no data was found and
   suggest checking the well ID or date range.
6. Keep the response concise: 3-6 sentences for simple queries,
   up to 10 sentences for complex analyses.'''

def run_explanation_agent(state: PipelineState) -> dict:
    """
    Generate the final natural language response.
    Builds a clean, tool-free message context.
    """
    llm = get_llm(temperature=0.3, max_tokens=600)

    # ── Build a clean context from first principles ───────────────
    # We do NOT pass pipeline message history here.
    # The LLM only sees: system rules, the user's question,
    # the retrieved data, and the computed analytics.

    context_parts = [
        f'USER_QUESTION: {state["user_message"]}',
    ]

    if state.get('tool_result_json'):
        context_parts.append(
            f'RETRIEVED DATA (from database): \n{state["tool_result_json"]}'
        )
    
    analystics = state.get('analytics_result')

    if analystics and analystics.get('status') == 'ok':
        context_parts.append(
            f'COMPUTED ANALYTICS (python result, guaranteed accurate):\n'
            f'{json.dumps(analystics, default=str)}'
        )
    
    context_parts.append(
        'please answer the user question based on the data above'
    )

    messages = [
        SystemMessage(content=_EXPLANATION_SYSTEM),
        HumanMessage(content='\n\n'.join(context_parts)),
    ]

    try:
        response = llm.invoke(messages)
        confidence_statement = ''
        # Extract confidence from response if stated.
        content_lower = response.content.lower()
        if '[high confidence]' in content_lower:
            confidence_statement = 'HIGH'
        elif '[medium confidence]' in content_lower:
            confidence_statement = 'MEDIUM'
        elif '[low confidence]' in content_lower:
            confidence_statement = 'LOW'
        else:
            confidence_statement = 'UNSPECIFIED'

    except Exception as e:
        return {
            'final_response':   'I encountered an error generating the explanation. '
                                f'The retrieved data is available in the debug state. Error: {e}',
            'error_log'     :   [f'ExplanationAgent error: {e}']
        }
        

    return {
        'llm_call_count':       state['llm_call_count'] + 1,
        'final_response':       response.content,
        'confidence_statement': confidence_statement,
        'error_log':            []
    }