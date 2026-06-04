"""
Interactive runner for the LangGraph pipeline.
Supports multi-turn conversations via session persistence.
"""
from src.graph import build_graph, get_checkpointer
import uuid, json


# Build graph once at startup
checkpointer    = get_checkpointer()
graph           = build_graph(checkpointer=checkpointer)
session_id      = str(uuid.uuid4())
config          = {'configurable': {'thread_id': session_id}}

print('Oil & Gas AI — Phase 5 LangGraph Pipeline')
print(f'Session ID: {session_id}')
print('Commands: "exit" | "debug" (toggle) | "new" (new session) | any query\n')
debug_mode = False

while True:
    user_input = input('You: ').strip()
    if not user_input:          continue
    if user_input == 'exit':    break
    if user_input == 'debug':
        debug_mode = not debug_mode
        print(f'Debug: {debug_mode}'); continue
    if user_input == 'new':
        session_id = str(uuid.uuid4())
        config     = {'configurable': {'thread_id': session_id}}
        print(f'New session: {session_id}'); continue

    print('\nProcessing...')
    state_input = {
        'user_message':   user_input,
        'session_id':     session_id,
        'error_log':      [],
        'llm_call_count': 0,
    }

    result = graph.invoke(state_input, config=config)

    if debug_mode:
        intent = result.get('intent') or {}
        plan   = result.get('execution_plan') or {}
        ar     = result.get('analytics_result') or {}
        print(f'\n[DEBUG] Intent:    {intent.get("label")}  ({intent.get("confidence",0):.2f})')
        print(f'[DEBUG] Well IDs:  {intent.get("well_ids")}')
        print(f'[DEBUG] Tool:      {plan.get("tool_name","n/a")}')
        print(f'[DEBUG] Retrieval: {result.get("retrieval_status","n/a")}')
        print(f'[DEBUG] Analytics: {ar.get("status","n/a")} | {ar.get("type","")}')
        if ar.get('trend_direction'):
            print(f'[DEBUG]   trend={ar["trend_direction"]} pct={ar.get("pct_change")}% conf={ar.get("confidence")}')
        if ar.get('rankings'):
            top = ar['rankings'][0]
            print(f'[DEBUG]   top risk: {top["well_id"]} score={top["failure_risk_score"]} tier={top["risk_tier"]}')
        print(f'[DEBUG] LLM calls: {result.get("llm_call_count")}')
        if result.get('error_log'):
            print(f'[DEBUG] Errors:    {result["error_log"]}')

    print(f'\nAssistant: {result["final_response"]}')
    print('-' * 60)
