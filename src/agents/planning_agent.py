"""
Agent 2: Query Planning (fully deterministic, zero LLM calls)

Responsibility: translate IntentResult into a concrete ExecutionPlan.
LLM calls: 0
Reads from state: intent
Writes to state: execution_plan
"""

from src.pipeline_state import PipelineState, ExecutionPlan, IntentResult
from src.schema_registry import registry
from typing import List

def _resolve_well_ids(intent: IntentResult) -> List[str]:
    """
    Return the list of well IDs to query.
    If intent.well_ids is empty and asset_ids are provided,
    expand asset IDs to all wells in those assets.
    If both are empty, return all wells in the registry.
    """
    
    if intent['well_ids']:
        return intent['well_ids']
    
    if intent['asset_ids']:
        # Expand asset to its wells
         return [w for w in registry.well_ids 
                if any(w.startswith(a) for a in intent['asset_ids'])]
    
    # No well or asset specified: return all wells
    return registry.well_ids

def run_planning_agent(state: PipelineState) -> PipelineState:
    """
    Build the execution plan from the intent.
    All routing decisions are deterministic rule-based logic.
    """
    intent = state['intent']
    label = intent['label']
    well_ids = _resolve_well_ids(intent)
    date_exp = intent['date_expression']
    scope = ('single_well' if len(well_ids) == 1 else 
             'multi_well' if 1 < len(well_ids) <= 8 else 
             'all_wells')
    
# ── Decision table ───────────────────────────────────────────
    if label == 'PRODUCTION_QUERY':
        plan = ExecutionPlan(
            tool_name       = 'fetch_production_data',
            tool_args       = {'well_ids': well_ids, 'date_expression': date_exp},
            run_analytics   = (scope == 'single_well'),
            analytics_type  = 'trend' if scope == 'single_well' else 'summary',
            scope           = scope
        )

    elif label == 'EQUIPMENT_QUERY':
        plan = ExecutionPlan(
            tool_name       = 'fetch_equipment_health',
            tool_args       = {'well_ids': well_ids, 'date_expression': date_exp},
            run_analytics   = False,
            analytics_type  = 'none',
            scope           = scope
        )

    elif label == 'DECLINE_ANALYSIS':
        # Always use 60 days for decline analysis to have enough data
        plan = ExecutionPlan(
            tool_name       = 'fetch_production_data',
            tool_args       = {'well_ids': well_ids, 'date_expression': 'last_60_days'},
            run_analytics   = True,
            analytics_type  = 'trend',
            scope           =  scope
        )

    elif label == 'RISK_ASSESSMENT':
        plan = ExecutionPlan(
            tool_name       = 'fetch_equipment_health',
            tool_args       = {'well_ids': well_ids, 'date_expression': date_exp},
            run_analytics   = True,
            analytics_type  = 'risk',
            scope           =  scope
        )

    elif label == 'COMPARISON':
        plan = ExecutionPlan(
            tool_name       = 'fetch_production_summary',
            tool_args       = {'well_ids': well_ids,
                              'date_expression': date_exp,
                              'group_by': 'well_id'},

            run_analytics   = False,
            analytics_type  = 'summary',
            scope           =  scope
        )

    else:
        plan = ExecutionPlan(
            tool_name       = 'inspect_database_schema',
            tool_args       = {},
            run_analytics   = False,
            analytics_type  = 'none',
            scope           = 'all_wells'
        )

    state['execution_plan'] = plan
    return state