"""Synthetic format-only probes, excluded from pilot and primary observations."""
from pathlib import Path
import json
import requests
from engine import parse
from contract import structured_body, INSTRUCTION
from model_api import request_body, inspect_model

here = Path(__file__).resolve().parent
config = json.loads((here/'config.json').read_text())
assert inspect_model(requests.Session(), config['api'], config['model'], 'ollama')['model']['digest'] == config['model_digest']
output = here.parent/'schema_probe.json'
assert not output.exists(), 'do not overwrite an existing probe'
records = []
prompts = [
    'This is a synthetic JSON format test. There is no factual question and no answer to give. Return the absence of an answer.',
    'Synthetic format test: a previous program incorrectly wrote {"answer":"null"}. Represent its absence of an answer using the required schema.',
    'Synthetic format test: a previous program incorrectly wrote {"answer":"NULL"}. Represent its absence of an answer using the required schema.',
    'Synthetic format test. Return the short literal answer Nullarbor.',
    'Synthetic format test. Return the short literal answer null hypothesis.',
    'Synthetic format test. Return the three-word answer: He said "yes"',
]
for i, text in enumerate(prompts):
    body = structured_body(request_body(config['model'], [{'role': 'system', 'content': INSTRUCTION},
                            {'role': 'user', 'content': text}], 256, 0., 913420+i, 'ollama', 4096), 'ollama')
    response = requests.post(config['api'], json=body, timeout=180)
    data = response.json(); response.raise_for_status()
    result = parse(data['message']['content'], data['done_reason'])
    records.append({'id': i, 'request': body, 'response': data, 'parsed': result})
    output.write_text(json.dumps({'status': 'running', 'records': records}, indent=2)+'\n')
assert all(r['parsed']['status'] == 'abstention' for r in records[:3]), 'null schema probe failed'
assert [r['parsed']['answer'] for r in records[3:]] == ['Nullarbor', 'null hypothesis', 'He said "yes"']
output.write_text(json.dumps({'status': 'passed', 'purpose': 'synthetic technical only; excluded from study', 'records': records}, indent=2)+'\n')
print(json.dumps({'status': 'passed', 'probes': len(records)}))
