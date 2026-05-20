from src.pipeline import run_pipeline
import json

print('Oil & Gas Multi-Agent Pipeline — Phase 4')
print('Commands: "exit" | "debug" (toggle) | any query\n')
debug_mode = False

while True:
    user_input = input('You: ').strip()
    if not user_input:          continue
    if user_input == 'exit':    break
    if user_input == 'debug':
        debug_mode = not debug_mode
        print(f'Debug: {debug_mode}'); continue

    print('\nProcessing...')
    state = run_pipeline(user_input)

    if debug_mode:
        intent = state.get('intent', {})
        plan   = state.get('execution_plan', {})
        if plan is not None:            
            print(f'\n[DEBUG] Intent:     {intent.get("label")}  (conf: {intent.get("confidence"):.2f})')
            print(f'[DEBUG] Well IDs:   {intent.get("well_ids")}')
            print('plan type:', plan)
            print(f'[DEBUG] Tool:       {plan.get("tool_name")}')
            print(f'[DEBUG] Retrieval:  {state.get("retrieval_status")}')
            if state.get('analytics_result'):
                ar = state['analytics_result']
                print(f'[DEBUG] Analytics:  {ar.get("status")} | type={ar.get("type","n/a")}')
                if ar.get('trend_direction'):
                    print(f'[DEBUG]   trend={ar["trend_direction"]} | pct={ar.get("pct_change")}% | conf={ar.get("confidence")}')
            print(f'[DEBUG] LLM calls:  {state.get("llm_call_count")}')
            if state.get('error_log'):
                print(f'[DEBUG] Errors:     {state["error_log"]}')
    print(f'\nAssistant: {state["final_response"]}\n')
    print('-' * 60)
