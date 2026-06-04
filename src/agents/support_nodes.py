"""
Support nodes for the LangGraph graph.
These handle terminal conditions: clarification requests and errors.
Both route to END after setting final_response.
"""
from src.pipeline import PipelineState
from src.schema_registry import registry

def ask_clarification(state: PipelineState) -> dict:
    """
    Terminal node: intent was ambiguous or well ID unresolvable.
    Returns a clarification prompt listing valid well IDs.
    """
    sample_wells = ','.join(registry.well_ids[:8])

    return {
        'final_response': (
            f'I could not identify a specific well from your question. '
            f'Please use an exact well ID. Available wells include: '
            f'{sample_wells} ... '
            f'You can ask "What wells do you have?" to see the full list.'
        ),
        'error_log': []
    }

def handle_error(state: PipelineState)-> dict:
    """
    Terminal node: data retrieval failed or an agent crashed.
    Surfaces the error log in a user-friendly message.
    """
    errors = ','.join(state.get('error_log', 'Unknown error'))
    return {
        'final_response': (
            f'I encountered a problem retrieving your data. '
            f'Details: {errors}. '
            f'Please verify the well ID exists and the date range is valid.'
        ),
        'error_log': []
    }