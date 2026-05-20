"""
Phase 4 manual pipeline orchestrator.
Runs the five agents in sequence using a shared PipelineState dict.
In Phase 5 this file is replaced by a LangGraph StateGraph.
"""
from src.pipeline_state import PipelineState
from src.agents.intent_agent import run_intent_agent
from src.agents.planning_agent import run_planning_agent
from src.agents.retrieval_agent import run_retrieval_agent
from src.agents.analytics_agent import run_analytics_agent
from src.agents.explanation_agent import run_explanation_agent
import uuid

def _make_initial_state(user_message: str,
                        history: list = None,
                        session_id: str = None) -> PipelineState:
    return PipelineState(
        user_message          = user_message,
        conversation_history  = history or [],
        session_id            = session_id or (uuid.uuid4()),
        intent                = None,
        execution_plan        = None,
        production_df         = None,
        equipment_df          = None,
        tool_result_json      = None,
        retrieval_status      = None,
        analytics_result      = None,
        final_response        = None,
        confidence_statement  = None,
        error_log             = [],
        llm_call_count        = 0,
    )

def _clarification_response(state: PipelineState) -> str:
    well_sample = ', '.join(state['intent'].get('well_ids', [])[:5])
    from src.schema_registry import registry
    available   = ', '.join(registry.well_ids[:10])
    return (
        f'I need clarification on the well you are asking about. '
        f'Available wells include: {available} ... '
        f'Please use an exact well ID from this list.'
    )


def _error_response(state: PipelineState) -> str:
    errors = '; '.join(state['error_log'])
    return (
        f'I was unable to retrieve the data for your query. '
        f'Reason: {errors}. '
        f'Please check the well ID and try again.'
    )


def run_pipeline(user_message: str,history: list = None,session_id: str = None) -> PipelineState:
    """
    Run the full five-agent pipeline for a single user message.
    Returns the complete pipeline state including all intermediate results.
    """
    print('DEBUG: inside run pipeline------')
    state = _make_initial_state(user_message, history, session_id)			

    # ── Run agents in sequence ────────────────────────────────────
    # Each agent reads from state, does its work, and writes back.
    # Agents do not call each other. The orchestrator controls the order.

    state = run_intent_agent(state) 		#Agent 1: classify intent
    print('1. completed intent')
    # Early exit: if intent is UNKNOWN and needs clarification,
    # skip data retrieval and return a clarification request.
    if state['intent']['needs_clarification']:
        state['final_response'] = _clarification_response(state)
        return state

    state = run_planning_agent(state)		#Agent 2: build execution plan
    print('2. ran planning agent')
    state = run_retrieval_agent(state)		#Agent 3: execute tool call
    print('3. ran retrieval')
    # Early exit: if retrieval failed, skip analytics and explanation
    if state['retrieval_status'] == 'error':
        state['final_response'] = _error_response(state)
        return state
        
    state = run_analytics_agent(state)		#Agent 4: run python analytics
    print('4. ran analytics')
    state = run_explanation_agent(state)	#Agent 5: generate natural language response
    print('5. ran explaination')
    return state