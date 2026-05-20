"""
Pipeline state definition for the Phase 4 multi-agent system.
Every agent reads from and writes to an instance of this dict.
No agent reads from or writes to anything outside this dict.
"""
from datetime import date
from typing import TypedDict, Optional, List, Any, Annotated
import operator

from streamlit import metric

class IntentResult(TypedDict):
    label:              str     # PRODUCTION_QUERY | EQUIPMENT_QUERY |
                                # DECLINE_ANALYSIS | RISK_ASSESSMENT |
                                # COMPARISON | SUMMARY | UNKNOWN
    well_ids:           list[str]
    asset_ids:          list[str]
    date_expression:    str     # last_30_days | last_month | etc.
    metric:             str     # oil_produced_bbl | failure_risk_score | etc.
    confidence:         float   #0.0 - 1.0
    needs_clarification:bool

class ExecutionPlan(TypedDict):
    tool_name:      str     # which @tool to call
    tool_args:      dict    # arguments for that tool
    run_analytics:  bool    # weather to run compute_production_trend
    analytics_type: str     # 'trend' | 'risk' | 'summary' | 'none'
    scope:          str     # 'single_well' | 'multi_well' | 'all_wells'


class PipelineState(TypedDict):
    # ── Input (set once at pipeline entry) ───────────────────────
    user_message:           str
    conversation_history:   list[dict]  #{role, content}
    session_id:             str

    # ── Agent 1 output ───────────────────────────────────────────
    intent:                 Optional[IntentResult]

    # ── Agent 2 output ───────────────────────────────────────────
    execution_plan:         Optional[ExecutionPlan]

    # ── Agent 3 output ───────────────────────────────────────────
    production_df:        Optional[Any]
    equipment_df:         Optional[Any]
    tool_result_json:     Optional[str]   # JSON string for LLM context
    retrieval_status:     Optional[str]   # 'ok' | 'no_data' | 'error'

    # ── Agent 4 output ───────────────────────────────────────────
    analytics_result:     Optional[dict]

    # ── Agent 5 output ───────────────────────────────────────────
    final_response:       Optional[str]
    confidence_statement: Optional[str]

    # ── Control ──────────────────────────────────────────────────
    # ── error_log uses operator.add reducer: lists are APPENDED ──
    # Each node returns {'error_log': ['new msg']}
    # LangGraph calls operator.add(existing, ['new msg'])
    # Result: errors accumulate across all nodes, never overwritten
    error_log:            Annotated[List[str], operator.add]
    llm_call_count:       int


