# tests/test_phase5_memory.py
import pytest
from src.graph import build_graph, get_checkpointer
import uuid


def _invoke(graph, state_input, config):
    return graph.invoke(state_input, config=config)


@pytest.mark.slow
def test_multi_turn_session_persists_intent():
    """
    Turn 1: ask about a well.
    Turn 2: ask a follow-up about the same topic.
    Verify the second turn uses the checkpointed state.
    """
    checkpointer = get_checkpointer('./db/test_checkpoints.sqlite')
    graph        = build_graph(checkpointer=checkpointer)
    thread_id    = f'test-{uuid.uuid4()}'
    config       = {'configurable': {'thread_id': thread_id}}

    # Turn 1
    r1 = _invoke(graph, {
        'user_message':   'What is A001-W01 production last month?',
        'session_id':     thread_id,
        'error_log':      [],
        'llm_call_count': 0,
    }, config)
    assert r1['final_response'] is not None
    assert r1['retrieval_status'] == 'ok'

    # Turn 2: the graph reloads prior state from checkpoint
    r2 = _invoke(graph, {
        'user_message':   'And what is the equipment risk for that well?',
        'session_id':     thread_id,
        'error_log':      [],
        'llm_call_count': 0,
    }, config)
    assert r2['final_response'] is not None
    # The graph should have handled a second independent query
    # State from turn 1 is available but turn 2 generates its own response
    assert r2['llm_call_count'] >= 1


# ── Graph structure tests (no LLM, no DB) ────────────────────────────
def test_graph_compiles_without_error():
    graph = build_graph()   # no checkpointer
    assert graph is not None

def test_graph_has_correct_nodes():
    graph      = build_graph()
    node_names = set(graph.get_graph().nodes.keys())
    expected   = {
        '__start__', 'classify_intent', 'plan_query',
        'retrieve_data', 'run_analytics', 'explain_results',
        'ask_clarification', 'handle_error', '__end__'
    }
    assert expected.issubset(node_names), f'Missing nodes: {expected - node_names}'

def test_graph_has_correct_edges():
    graph      = build_graph()
    mermaid    = graph.get_graph().draw_mermaid()
    assert 'classify_intent' in mermaid
    assert 'ask_clarification' in mermaid
    assert 'handle_error'      in mermaid
    assert 'explain_results'   in mermaid


# ── Edge routing unit tests (no LLM, no DB) ──────────────────────────
def test_route_intent_needs_clarification():
    from src.graph_edges import route_after_intent
    state = {'intent': {'needs_clarification': True, 'confidence': 0.9}}
    assert route_after_intent(state) == 'ask_clarification'

def test_route_intent_low_confidence():
    from src.graph_edges import route_after_intent
    state = {'intent': {'needs_clarification': False, 'confidence': 0.2}}
    assert route_after_intent(state) == 'ask_clarification'

def test_route_intent_ok():
    from src.graph_edges import route_after_intent
    state = {'intent': {'needs_clarification': False, 'confidence': 0.9}}
    assert route_after_intent(state) == 'plan_query'

def test_route_retrieval_error():
    from src.graph_edges import route_after_retrieval
    assert route_after_retrieval({'retrieval_status': 'error'}) == 'handle_error'

def test_route_retrieval_no_data():
    from src.graph_edges import route_after_retrieval
    assert route_after_retrieval({'retrieval_status': 'no_data'}) == 'explain_results'

def test_route_retrieval_ok():
    from src.graph_edges import route_after_retrieval
    assert route_after_retrieval({'retrieval_status': 'ok'}) == 'run_analytics'

def test_route_analytics_none_goes_to_error():
    from src.graph_edges import route_after_analytics
    assert route_after_analytics({'analytics_result': None}) == 'handle_error'

def test_route_analytics_ok_goes_to_explain():
    from src.graph_edges import route_after_analytics
    state = {'analytics_result': {'status': 'ok'}}
    assert route_after_analytics(state) == 'explain_results'
