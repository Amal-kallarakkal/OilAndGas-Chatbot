"""
Agent 4: Analytics (pure Python, zero LLM calls)

Responsibility: compute trends, risk scores, or summaries from
               DataFrames stored in the pipeline state.
LLM calls: 0
Reads from state: execution_plan, production_df, equipment_df
Writes to state: analytics_result
"""
import pandas as pd
from src.pipeline_state import PipelineState
from src.analytics.production_analytics import _production_trend
from src.schema_registry import registry

def _compute_trend(state: PipelineState) -> dict:
    """Compute production trend. Used for PRODUCTION_QUERY and DECLINE_ANALYSIS."""
    df = pd.DataFrame(
        state.get('production_df', [])
    )
    plan = state['execution_plan']
    intent = state['intent']
    if df is None or (hasattr(df, 'empty') and df.empty):
        return {'status': 'no_data'}
    
    well_ids = intent['well_ids']
    date_exp = plan['tool_args'].get('date_expression', 'last_60_days')

    if len(well_ids) == 1:
        # Single well: full trend analysis
        return _production_trend(well_ids[0], date_exp)
    else:
        # Multiple wells: compute trend for each and summarise
        results = {}
        for wid in well_ids[:8]:     # cap at 8 to avoid long compute
            results[wid] = _production_trend(wid, date_exp)
        declining = [w for w,r in results.items()
                     if r.get('trend_direction') == 'declining']
        
        return {
            'status':     'ok',
            'type':       'multi_well_trend',
            'well_count': len(results),
            'declining':  declining,
            'stable':     [w for w,r in results.items()
                          if r.get('trend_direction') == 'stable'],
            'details':    results
        }

def _compute_risk(state: PipelineState) -> dict:
    """Compute risk assessment from equipment DataFrame."""
    df = pd.DataFrame(
        state.get('equipment_df', [])
    ) 
    if df is None or (hasattr(df, 'empty') and df.empty):
        return {'status': 'no_data'}
    
    vib_col = 'vibration_level'
    risk_col = 'failure_risk_score'

    # latest reading per well
    latest = (df.sort_values('date').groupby('well_id').last().reset_index())

    records = []

    for _, row in latest.iterrows():
        risk = float(row.get(risk_col, 0))
        vib  = float(row.get(vib_col, 0))
        tier = ('CRITICAL' if risk >= 0.80 else
                'HIGH' if risk >= 0.65 else
                'MEDIUM' if risk >= 0.40 else 'LOW')
        records.append({
            'well_id':              row['well_id'],
            'failure_risk_score':   round(risk, 3),
            f'{vib_col}':           round(vib, 3),  
            'temperature_c':        round(row.get('temperature_c', 0), 1),
            'risk_tier':            tier
        })
        
    records.sort(key=lambda r: r['failure_risk_score'], reverse=True)

    return {
        'status':         'ok',
        'type':           'risk_assessment',
        'well_count':     len(records),
        'critical_count': sum(1 for r in records if r['risk_tier'] == 'CRITICAL'),
        'high_count':     sum(1 for r in records if r['risk_tier'] == 'HIGH'),
        'rankings':       records
    }


def run_analytics_agent(state: PipelineState) -> dict:
    """Dispatch to the correct analytics function based on analytics_type."""

    plan = state.get('execution_plan')
    analytics_result = {}
    # print('inside run_analytics_agent------')
    if not plan or not plan.get('run_analytics'):
        analytics_result = {'status': 'skipped'}
    
    analytics_type = plan.get('analytics_type', 'none')
    # print(analytics_type)
    try:
        if analytics_type == 'trend':
            analytics_result = _compute_trend(state)
        elif analytics_type == 'risk':
            analytics_result = _compute_risk(state)
        else:
            analytics_result = {'status': 'skipped'}

                
    except Exception as e:
        analytics_result = {'status': 'error', 'message': str(e)}
        return {
            'analytics_result': analytics_result,
            'error_log':        [f'AnalyticsAgent error: {e}']
        }
            

    return {
        'analytics_result': analytics_result,
        'error_log':        []
    }

    


            
    