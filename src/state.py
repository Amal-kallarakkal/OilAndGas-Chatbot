# src/state.py
from typing import TypedDict, Optional, List, Any
import pandas as pd

class ConversationTurn(TypedDict):
    role:    str   # 'user' or 'assistant'
    content: str

class OpsAssistantState(TypedDict):
    # ── Input ─────────────────────────────────────────────────
    user_message:         str
    conversation_history: List[ConversationTurn]
    session_id:           str

    # ── Intent Stage (populated in Phase 3) ──────────────────
    intent_label:       Optional[str]   # e.g. 'PRODUCTION_DECLINE'
    intent_confidence:  Optional[float]
    extracted_entities: Optional[dict]  # {well_id, date_range, metric}

    # ── Planning Stage (Phase 4) ──────────────────────────────
    execution_plan: Optional[dict]

    # ── Data Stage (Phase 4) ──────────────────────────────────
    production_df:  Optional[Any]   # pd.DataFrame
    equipment_df:   Optional[Any]

    # ── Analysis Stage (Phase 7) ─────────────────────────────
    analysis_results: Optional[dict]
    correlation_results: Optional[dict]

    # ── Output Stage ─────────────────────────────────────────
    final_response: Optional[str]
    error_log:      List[str]
