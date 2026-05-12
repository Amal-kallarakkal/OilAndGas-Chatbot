"""
Interactive test runner for the single-agent prototype.
Run: python tests/run_agent.py
"""
from src.agents.ops_agents import OpsAgent
import json

agent = OpsAgent()

print('Oil & Gas Operations Agent — Phase 3 Prototype')
print('Type "exit" to quit, "debug" to see full state.\n')

debug_mode = False

while True:
    user_input = input('you:').strip()
    if not user_input:
        continue
    if user_input.lower() == 'exit':
        print('Goodbye!')
        break
    if user_input.lower() == 'debug':
        debug_mode = not debug_mode
        print(f'Debug mode: {debug_mode}')
        continue

    print('\nprocessing...')
    result = agent.run(user_input)

    if debug_mode:
        print('\n[DEBUG] debug tool called:', result.get('tools_called'))
        print('[DEBUG] LLM calls:', result.get('llm_calls'))

        if result.get('analytics'):
            print('[DEBUG] Analytics results:')
            print( json.dumps(result.get('analytics'), indent=2))
            print()

    print(f'\nAssistant: {result["response"]}\n')
    print('-' * 60)
