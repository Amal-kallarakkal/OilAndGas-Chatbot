"""
Agent 1: Intent Classification

Responsibility: classify user message into a structured IntentResult.
LLM calls: 1 (temperature=0.0 for deterministic classification)
Reads from state: user_message, conversation_history
Writes to state: intent
"""

from langchain_core.messages import HumanMessage, SystemMessage
from src.llm import get_llm
from src.pipeline_state import PipelineState, IntentResult
import json, re
from src.schema_registry import registry

_INTENT_LABELS = [
    'PRODUCTION_QUERY',   # user wants raw production numbers
    'EQUIPMENT_QUERY',    # user wants equipment/sensor readings
    'DECLINE_ANALYSIS',   # user asks why production dropped
    'RISK_ASSESSMENT',    # user asks about failure risk
    'COMPARISON',         # user compares two or more wells
    'SUMMARY',            # user asks for an overview
    'UNKNOWN',            # cannot determine intent
]

_DEFAULT_DATE = 'last_30_days'
_DEFAULT_METRIC = registry.oil_column   # 'oil_produced_bbl'

def _build_system_prompt() -> str:
    well_sample = ', '.join(registry.well_ids[:10])
    asset_sample = ', '.join(registry.asset_ids)
    prod_cols = ', '.join(sorted(registry.production.numeric_columns))
    equip_cols = ', '.join(sorted(registry.equipment.numeric_columns))
    date_min = registry.date_range['min_date']
    date_max = registry.date_range['max_date']
    return f'''You are an intent classification engine for an Oil & Gas
            Operations Intelligence Assistant. Your ONLY job is to analyse the user
            message and return a single JSON object. Do not write any other text.

            DATABASE CONTEXT (use this to resolve entities):
            Available wells (sample): {well_sample} ...
            Available assets: {asset_sample}
            Production metrics: {prod_cols}
            Equipment metrics: {equip_cols}
            Data range: {date_min} to {date_max}

            INTENT LABELS:
            PRODUCTION_QUERY   - user wants production volumes, oil/gas output, water cut, downtime
            EQUIPMENT_QUERY    - user wants vibration, temperature, pressure, failure risk
            DECLINE_ANALYSIS   - user asks why production dropped or declined
            RISK_ASSESSMENT    - user asks which wells are at risk or about failure risk
            COMPARISON         - user compares 2+ wells or assets
            SUMMARY            - user asks for overview or general status
            UNKNOWN            - cannot determine from message

            RESPOND WITH EXACTLY THIS JSON SCHEMA:
            {{
            "label":           "<one of the INTENT LABELS above>",
            "well_ids":        ["<list of exact well IDs from the database, or []>"],
            "asset_ids":       ["<list of asset IDs like A001, or []>"],
            "date_expression": "<one of: last_30_days | last_60_days | last_90_days |
                                last_month | last_quarter | last_6_months | ytd>",
            "metric":          "<the most relevant column name from the lists above>",
            "confidence":      <float 0.0 to 1.0>,
            "needs_clarification": <true if well_ids is empty and label is not SUMMARY>
            }}

            RULES:
            - well_ids must contain EXACT IDs from the database. 'Well A' is NOT valid.
            - If user says a well name you cannot match exactly, set well_ids to []
            and needs_clarification to true.
            - If user says 'all wells' or no specific well, set well_ids to [].
            - If no time expression is mentioned, use last_30_days.
            - Respond ONLY with the JSON object. No preamble, no explanation.'''

def _extract_json(text: str) -> dict:
    """
    Extract JSON from LLM response.
    Handles cases where the model wraps JSON in markdown code fences.
    """
    # Strip markdown code fences if present
    text = re.sub(r'```(?:json)?', '', text).strip()
    # Find the first { ... } block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f'No JSON object found in LLM response: {text[:200]}')
    return json.loads(match.group())

def run_intent_agent(state: PipelineState) -> dict:
    """
    Classify the user's intent and extract entities.
    Writes state['intent'] and increments state['llm_call_count'].
    """
    llm = get_llm(temperature = 0.0, max_tokens=400)
    print('DEBUG: inside intent agent')
    messages = [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=state['user_message'] )
    ]
    print('messages: ', messages)
    try:
        response = llm.invoke(messages)
        parsed = _extract_json(response.content)
        print('\n\n parsed result:', parsed)
        # Validate label
        label = parsed.get('label','UNKNOWN')
        if label not in _INTENT_LABELS:
            label = 'UNKNOWN'
        return {
            'intent':           IntentResult(
                                    label             = label,
                                    well_ids          = parsed.get('well_ids', []),
                                    asset_ids         = parsed.get('asset_ids', []),
                                    date_expression   = parsed.get('date_expression', _DEFAULT_DATE),
                                    metric            = parsed.get('metric', _DEFAULT_METRIC),
                                    confidence        = parsed.get('confidence', 0.5),
                                    needs_clarification = parsed.get('needs_clarification', False)
                                ),
            'llm_call_count':   state['llm_call_count'] + 1,
            'error_log':        []
        }
        
    except Exception as e:
        print(f'Intent_agent_error: {e}')
        fallback_intent = IntentResult(
            label = 'UNKNOWN', well_ids=[], asset_ids=[],
            date_expression=_DEFAULT_DATE, metric= _DEFAULT_METRIC,
            confidence=0.0, needs_clarification=True
        )

    return {
        'intent': fallback_intent,
        'llm_call_count':   state['llm_call_count'] + 1,
        'error_log': [f'Intent agent error: {e}']
    }




