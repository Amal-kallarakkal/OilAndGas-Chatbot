"""
Conditional edge functions for the LangGraph StateGraph.

Each function:
  - Takes the current PipelineState
  - Returns a string: the name of the next node to run
  - Has NO side effects
  - Is independently testable

Node name strings must exactly match the names used in
StateGraph.add_node() calls in src/graph.py.
"""
from src.pipeline import PipelineState

def route_after_intent(state: PipelineState) -> str:
    """
    After intent classification:
    - needs_clarification=True  -> ask_clarification (terminal)
    - confidence < 0.35         -> ask_clarification
    - otherwise                 -> plan_query
    """
    intent = state.get('intent')

    if not intent:
        return 'ask_clarification'
    if intent.get('needs_clarification') == True:
        return 'ask_clarification'
    if intent.get('confidence') < 0.35:
        return 'ask_clarification'
    
    return 'plan_query'

def route_after_retrieval(state: PipelineState) -> str:
    """
    After data retrieval:
    - retrieval_status == 'error'   -> handle_error (terminal)
    - retrieval_status == 'no_data' -> explain_results
      (LLM explains why no data was found; analytics skipped)
    - otherwise                     -> run_analytics
    """
    status = state.get('retrieval_status')

    if status == 'error':
        return 'handle_error'
    if status == 'no_data':
        return 'explain_results'
    
    return 'run_analytics'

def route_after_analytics(state: PipelineState) -> str:
    """
    After analytics:
    - analytics errored critically  -> handle_error
    - otherwise                    -> explain_results
    Analytics failure is non-fatal: we still explain what we retrieved.
    Only route to error if analytics itself raised an unhandled exception
    that left analytics_result as None.
    """
    ar = state.get('analytics_result')
    if ar is None or ar == 'error':
        # Analytics node crashed entirely (status not set)
        return 'handle_error'
    if ar.get('status') in ('skipped', 'insufficient_data', 'ok'):
        return 'explain_results'
    
    return 'explain_results'

