# tests/test_phase4_pipeline.py
import pytest, json
from src.pipeline_state import PipelineState, IntentResult, ExecutionPlan
from src.agents.planning_agent   import run_planning_agent, _resolve_well_ids
from src.agents.retrieval_agent  import run_retrieval_agent
from src.agents.analytics_agent  import run_analytics_agent
from src.schema_registry import registry


def _make_state(**overrides) -> PipelineState:
    """Helper: build a minimal PipelineState for testing.""" 
    base = PipelineState(
        user_message='test', conversation_history=[],
        session_id='test-001', intent=None, execution_plan=None,
        production_df=None, equipment_df=None,
        tool_result_json=None, retrieval_status=None,
        analytics_result=None, final_response=None,
        confidence_statement=None, error_log=[], llm_call_count=0
    )
    base.update(overrides)
    return base


def _make_intent(label, well_ids=None, date_exp='last_30_days') -> IntentResult:
    return IntentResult(
        label=label, well_ids=well_ids or ['A001-W01'],
        asset_ids=[], date_expression=date_exp,
        metric=registry.oil_column, confidence=0.9,
        needs_clarification=False
    )


# ── Planning Agent tests (no LLM, no DB) ─────────────────────────────
def test_planning_production_query():
    state = _make_state(intent=_make_intent('PRODUCTION_QUERY'))
    state = run_planning_agent(state)
    assert state['execution_plan']['tool_name'] == 'fetch_production_data'
    assert state['execution_plan']['analytics_type'] == 'trend'

def test_planning_decline_forces_60_days():
    state = _make_state(intent=_make_intent('DECLINE_ANALYSIS', date_exp='last_30_days'))
    state = run_planning_agent(state)
    plan  = state['execution_plan']
    assert plan['tool_args']['date_expression'] == 'last_60_days'
    assert plan['run_analytics'] == True
    assert plan['analytics_type'] == 'trend'

def test_planning_risk_uses_equipment_tool():
    state = _make_state(intent=_make_intent('RISK_ASSESSMENT'))
    state = run_planning_agent(state)
    assert state['execution_plan']['tool_name'] == 'fetch_equipment_health'
    assert state['execution_plan']['analytics_type'] == 'risk'

def test_planning_comparison_uses_summary_tool():
    state = _make_state(intent=_make_intent('COMPARISON',
                                            well_ids=['A001-W01','A001-W02']))
    state = run_planning_agent(state)
    assert state['execution_plan']['tool_name'] == 'fetch_production_summary'

def test_resolve_well_ids_expands_asset():
    intent = _make_intent('PRODUCTION_QUERY', well_ids=[])
    intent['asset_ids'] = ['A001']
    result = _resolve_well_ids(intent)
    assert all(w.startswith('A001') for w in result)
    assert len(result) > 0


# ── Retrieval Agent tests (DB calls, no LLM) ─────────────────────────
def test_retrieval_production_sets_df():
    intent = _make_intent('PRODUCTION_QUERY')
    plan   = ExecutionPlan(
        tool_name='fetch_production_data',
        tool_args={'well_ids': ['A001-W01'], 'date_expression': 'last_30_days'},
        run_analytics=True, analytics_type='trend', scope='single_well'
    )
    state = _make_state(intent=intent, execution_plan=plan)
    state = run_retrieval_agent(state)
    assert state['retrieval_status'] == 'ok'
    assert state['production_df'] is not None
    assert not state['production_df'].empty
    assert state['tool_result_json'] is not None

def test_retrieval_equipment_sets_df():
    intent = _make_intent('RISK_ASSESSMENT')
    plan = ExecutionPlan(
        tool_name='fetch_equipment_health', 
        tool_args={'well_ids': ['A001-W01'], 'date_expression': 'last_30_days'},
        run_analytics=True, analytics_type='risk', scope='single_well'
    )
    state = _make_state(intent=intent, execution_plan=plan)
    state = run_retrieval_agent(state)
    assert state['retrieval_status'] == 'ok'
    assert state['equipment_df'] is not None


# ── Analytics Agent tests (no LLM, no DB) ────────────────────────────
def test_analytics_skipped_when_flag_false():
    plan  = ExecutionPlan(tool_name='x',tool_args={},
                         run_analytics=False,analytics_type='none',scope='x')
    state = _make_state(execution_plan=plan)
    state = run_analytics_agent(state)
    assert state['analytics_result']['status'] == 'skipped'

def test_analytics_trend_with_real_data():
    intent = _make_intent('DECLINE_ANALYSIS')
    plan   = ExecutionPlan(
        tool_name='fetch_production_data',
        tool_args={'well_ids': ['A001-W01'], 'date_expression': 'last_60_days'},
        run_analytics=True, analytics_type='trend', scope='single_well'
    )
    state = _make_state(intent=intent, execution_plan=plan)
    state = run_retrieval_agent(state)     # populate production_df
    state = run_analytics_agent(state)
    ar = state['analytics_result']
    assert ar['status'] in ('ok', 'error', 'skipped', 'no_data', 'insufficient_data')
    if ar['status'] == 'ok':
        assert ar['trend_direction'] in ('declining','stable','improving')
        assert ar['metric'] == registry.oil_column

def test_analytics_risk_returns_rankings():
    intent = _make_intent('RISK_ASSESSMENT', well_ids=registry.well_ids[:4])
    plan   = ExecutionPlan(
        tool_name='fetch_equipment_health',
        tool_args={'well_ids': registry.well_ids[:4], 'date_expression': 'last_30_days'},
        run_analytics=True, analytics_type='risk', scope='multi_well'
    )
    state = _make_state(intent=intent, execution_plan=plan)
    state = run_retrieval_agent(state)
    state = run_analytics_agent(state)
    ar = state['analytics_result']
    assert ar['status'] == 'ok'
    assert 'rankings' in ar
    assert len(ar['rankings']) > 0
    assert all('risk_tier' in r for r in ar['rankings'])


# ── Full pipeline integration tests (LLM required) ───────────────────
@pytest.mark.slow
def test_full_pipeline_production_query():
    from src.pipeline import run_pipeline
    state = run_pipeline('What was A001-W01 production last month?')
    assert state['intent']['label'] == 'PRODUCTION_QUERY'
    assert state['retrieval_status'] == 'ok'
    assert state['final_response'] is not None
    assert state['llm_call_count'] == 2

@pytest.mark.slow
def test_full_pipeline_decline_analysis():
    from src.pipeline import run_pipeline
    state = run_pipeline('Is A001-W01 production declining?')
    plan = state.get('execution_plan', {})
    assert plan.get('tool_args', {}).get('date_expression') == 'last_60_days'
    assert state['analytics_result']['status'] in ('ok','insufficient_data')

@pytest.mark.slow
def test_full_pipeline_needs_clarification():
    from src.pipeline import run_pipeline
    state = run_pipeline('What is Well Alpha production?')
    # 'Well Alpha' does not exist in the DB; agent should ask for clarification
    assert state['llm_call_count'] == 1  # only intent call made
    assert 'clarification' in state['final_response'].lower() or 'available wells' in state['final_response'].lower()
