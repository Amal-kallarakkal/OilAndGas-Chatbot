"""
Agent 3: Data Retrieval (tool execution, zero LLM calls)

Responsibility: execute the tool specified in the execution plan
               and store both the DataFrame and JSON result.
LLM calls: 0
Reads from state: execution_plan
Writes to state: production_df, equipment_df, tool_result_json,
                 retrieval_status
"""
import json
import pandas as pd
from src.pipeline_state import PipelineState
from src.tools.production_tools import (
    fetch_production_data, fetch_equipment_health,
    inspect_database_schema, fetch_production_summary
)
from src.database import get_production_data, get_equipment_health
from src.utils import resolve_date_range, parse_list_param

# Tool registry: name -> LangChain @tool object
TOOL_REGISTRY = {
    'fetch_production_data':   fetch_production_data,
    'fetch_equipment_health':  fetch_equipment_health,
    'inspect_database_schema': inspect_database_schema,
    'fetch_production_summary': fetch_production_summary,
}

def run_retrieval_agent(state: PipelineState) -> PipelineState:
    """
    Execute the tool call from the execution plan.
    Stores both the JSON summary (for LLM) and raw DataFrame (for analytics).
    """
    plan = state['execution_plan']
    tool_name = plan['tool_name']
    tool_args = plan['tool_args']

    if tool_name not in TOOL_REGISTRY:
        state['error_log'].append(f'Unknown tool: {tool_name}')
        state['retrieval_status'] = 'error'
        return state
    
    try:
        # ── Execute the @tool function ────────────────────────────
        tool_fn = TOOL_REGISTRY[tool_name]
        json_out = tool_fn.invoke(tool_args)
        parsed = json.loads(json_out)

        state['tool_result_json'] = json_out
        state['retrieval_status'] = parsed.get('status')

        # ── Also fetch raw DataFrames for analytics agent ─────────
        # The @tool returns a JSON summary for the LLM.
        # The analytics agent needs the raw DataFrame.
        # We fetch it separately here to avoid duplicating logic in
        # the @tool itself.

        if tool_name == 'fetch_production_data' and parsed.get('status') == 'ok':
            well_ids = parse_list_param(tool_args.get('well_ids'), [])
            start, end = resolve_date_range(tool_args.get('date_expression', 'last_30_days'))
            state['production_df'] = get_production_data(well_ids, start, end)
            
        elif tool_name == 'fetch_equipment_health' and parsed.get('status') == 'ok':
            well_ids = parse_list_param(tool_args.get('well_ids'), [])
            start, end = resolve_date_range(tool_args.get('date_expression', 'last_30_days'))
            state['equipment_df'] = get_equipment_health(well_ids, start, end)
        elif tool_name == 'fetch_production_summary' and parsed.get('status') == 'ok':
            well_ids = parse_list_param(tool_args.get('well_ids'), [])
            start, end = resolve_date_range(tool_args.get('date_expression', 'last_30_days'))
            state['production_df'] = get_production_data(well_ids, start, end)

    except Exception as e:
        state['error_log'].append(f'Retrieval_agent error: {e}')
        state['retrieval_status'] = 'error'

    return state






