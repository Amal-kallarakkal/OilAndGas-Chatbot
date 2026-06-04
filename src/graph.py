from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from src.pipeline import PipelineState

# Agent nodes
from src.agents.intent_agent import run_intent_agent
from src.agents.planning_agent import run_planning_agent
from src.agents.retrieval_agent import run_retrieval_agent
from src.agents.analytics_agent import run_analytics_agent
from src.agents.explanation_agent import run_explanation_agent
from src.agents.support_nodes import ask_clarification, handle_error

# Conditional edge function
from src.graph_edges import (
    route_after_analytics, 
    route_after_intent, 
    route_after_retrieval
)

import os

def build_graph(checkpointer=None):
    """
    Build and compile the LangGraph StateGraph.

    Args:
        checkpointer: A LangGraph checkpointer instance for state
                      persistence. If None, state is not persisted
                      between calls (useful for testing).

    Returns:
        A compiled graph ready for .invoke() or .stream() calls.
    """
    # ── 1. Create the graph with the state schema ───────────────
    graph = StateGraph(PipelineState)

    # ── 2. Register nodes ─────────────────────────────────────────
        # Node name (string) -> Python function
        # The string names are used in edge definitions and routing functions
    graph.add_node('classify_intent', run_intent_agent)
    graph.add_node('plan_query', run_planning_agent)
    graph.add_node('retrieve_data', run_retrieval_agent)
    graph.add_node('run_analytics', run_analytics_agent)
    graph.add_node('explain_results', run_explanation_agent)
    graph.add_node('ask_clarification', ask_clarification)
    graph.add_node('handle_error', handle_error)

    # ── 3. Entry edge: graph always starts at classify_intent ─────
    graph.add_edge(START, 'classify_intent')

    # ── 4. Conditional edge after intent classification ───────────
        # route_after_intent() returns 'ask_clarification' or 'plan_query'
    graph.add_conditional_edges(
        'classify_intent',
        route_after_intent,
        {   
            'plan_query'  : 'plan_query',
            'handle_error': 'handle_error',
            'ask_clarification': 'ask_clarification',
        }
    )

    # ── 5. Fixed edge: planning always goes to retrieval ──────────
    graph.add_edge('plan_query', 'retrieve_data')

    # ── 6. Conditional edge after retrieval ──────────────────────
        # route_after_retrieval() returns 'run_analytics', 'explain_results',
        # or 'handle_error'
    graph.add_conditional_edges(
        'retrieve_data',
        route_after_retrieval, 
        {
            'run_analytics':    'run_analytics',
            'explain_results':  'explain_results',
            'handle_error':     'handle_error',
        }
    )


    # ── 7. Conditional edge after analytics ──────────────────────
    graph.add_conditional_edges(
        'run_analytics',
        route_after_analytics, 
        {
            'explain_results': 'explain_results',
            'handle_error':    'handle_error'
        }
    )

    # ── 8. Terminal edges: all paths end at END ───────────────────
    graph.add_edge('explain_results', END)
    graph.add_edge('ask_clarification', END)
    graph.add_edge('handle_error', END)

    # ── 9. Compile with optional checkpointer ────────────────────
    return graph.compile(checkpointer=checkpointer)


_CHECKPOINTER = None
_CHECKPOINTER_CM = None

def get_checkpointer(db_path: str = './db/checkpoints.sqlite'):
    """
    Create a SqliteSaver checkpointer for local development.
    The SQLite file is created automatically if it does not exist.
    """
    global _CHECKPOINTER
    global _CHECKPOINTER_CM
    if _CHECKPOINTER is not None:
        return _CHECKPOINTER
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    _CHECKPOINTER_CM = SqliteSaver.from_conn_string(db_path)
    _CHECKPOINTER = _CHECKPOINTER_CM.__enter__()
    return _CHECKPOINTER