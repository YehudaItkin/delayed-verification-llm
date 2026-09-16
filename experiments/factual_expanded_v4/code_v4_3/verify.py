"""Offline verification of a complete phase, including full prompt-chain replay."""
from pathlib import Path
import collections
import hashlib
import json
import sys
from engine import digest, seed, parse, initial, trajectory, amplitude_bounds
from model_api import request_body
from contract import structured_body
from scorer import score


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    spec = manifest['specification']; config = manifest['config']; phase = manifest['phase']
    code = directory.parents[1]/'code'
    for name, sha in manifest['code_sha256'].items():
        assert hashlib.sha256((code/name).read_bytes()).hexdigest() == sha, name
    requests = [json.loads(s) for s in (directory/'requests.jsonl').read_text().splitlines()]
    runs = [json.loads(s) for s in (directory/'runs.jsonl').read_text().splitlines()]
    journal = {r['id']: r for r in requests}; saved = {r['id']: r for r in runs}
    assert len(journal) == len(requests) and len(saved) == len(runs)
    expected_count = len(spec['items'])*(3+sum((spec['T']-1)*3*(1+c['verifier_on']) for c in spec['cells']))
    assert len(requests) == expected_count and len(runs) == len(spec['items'])*len(spec['cells'])
    seen = set(); identities = set()
    for item in spec['items']:
        def call(context, messages, temperature):
            identity = {'phase': phase, 'dataset': item['dataset'], 'q': item['q'], **context}
            rid = digest(identity); seen.add(rid); record = journal[rid]
            expected = structured_body(request_body(config['model'], messages, 256, temperature,
                                       seed(identity), config['backend'], 4096), config['backend'])
            assert record['identity'] == identity and record['seed'] == seed(identity)
            assert record['request'] == expected, 'prompt chain changed'
            response = record['response']; assert response['done'] is True
            parsed = parse(response['message']['content'], response['done_reason'])
            assert record['parsed'] == parsed
            assert record['usage']['prompt_tokens'] == response['prompt_eval_count']
            assert record['usage']['completion_tokens'] == response.get('eval_count', 0)
            assert sum(record['usage'].values()) <= 4096 and record['usage']['prompt_tokens'] <= 3500
            return parsed
        start = initial(item, call)
        for cell in spec['cells']:
            ident = {'phase': phase, 'dataset': item['dataset'], 'q': item['q'], **cell}
            rid = digest(ident); identities.add(rid)
            def condition_call(context, messages, temperature):
                return call({**cell, **context}, messages, temperature)
            actual = trajectory(item, start, cell['delay'], cell['verifier_on'], spec['T'], condition_call)
            row = saved[rid]
            assert all(row[k] == actual[k] for k in actual)
            assert row['manifest_sha256'] == digest(manifest) and row['identity'] == ident
            assert row['history_indices'] == [max(0, t-1-cell['delay']) for t in range(1, spec['T'])]
            labels = [[score(item, a['answer'], a['status'], item['catalog']) for a in step] for step in actual['states']]
            assert row['labels'] == labels
            bounds = amplitude_bounds([[a['reference_error'] for a in step] for step in labels[spec['T']//2:]])
            assert row['amplitude_bounds'] == bounds
    assert seen == set(journal) and identities == set(saved)
    report = {'status': 'passed', 'phase': phase, 'runs': len(runs), 'requests': len(requests),
              'technical_status_counts': dict(collections.Counter(r['parsed']['status'] for r in requests)),
              'max_prompt_tokens': max(r['usage']['prompt_tokens'] for r in requests),
              'manifest_sha256': digest(manifest),
              'files_sha256': {name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ('manifest.json','requests.jsonl','runs.jsonl')}}
    (directory/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    return report


if __name__ == '__main__': verify(sys.argv[1])
