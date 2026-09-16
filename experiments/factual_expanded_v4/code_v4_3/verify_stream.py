"""Verify a finished phase without loading its entire request journal in memory."""
from pathlib import Path
import collections
import hashlib
import json
import math
import sys
from engine import digest, seed, parse, initial, trajectory, amplitude_bounds
from contract import structured_body
from model_api import request_body
from scorer import score


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    spec = manifest['specification']; config = manifest['config']; phase = manifest['phase']
    assert config['backend'] == 'ollama'
    code = directory.parents[1]/'code'
    for name, sha in manifest['code_sha256'].items():
        assert file_hash(code/name) == sha, name
    mh = digest(manifest)
    counts = collections.Counter(); labels_by_dataset = collections.defaultdict(collections.Counter)
    total_requests = 0; total_runs = 0; max_prompt = 0; pairs = []
    with (directory/'requests.jsonl').open() as requests, (directory/'runs.jsonl').open() as runs:
        for item in spec['items']:
            def call(context, messages, temperature):
                nonlocal total_requests, max_prompt
                identity = {'phase': phase, 'dataset': item['dataset'], 'q': item['q'], **context}
                record = json.loads(next(requests))
                expected = structured_body(request_body(config['model'], messages, 256, temperature,
                                           seed(identity), 'ollama', 4096), 'ollama')
                assert record['id'] == digest(identity) and record['identity'] == identity
                assert record['seed'] == seed(identity) and record['request'] == expected
                response = record['response']; assert response['done'] is True
                parsed = parse(response['message']['content'], response['done_reason'])
                assert record['parsed'] == parsed
                assert record['usage'] == {'prompt_tokens': response['prompt_eval_count'], 'completion_tokens': response.get('eval_count', 0)}
                assert sum(record['usage'].values()) <= 4096 and record['usage']['prompt_tokens'] <= 3500
                counts[parsed['status']] += 1; total_requests += 1
                max_prompt = max(max_prompt, record['usage']['prompt_tokens'])
                return parsed
            start = initial(item, call)
            conditions = {}
            for cell in sorted(spec['cells'], key=lambda c: digest([item['id'], c, 'condition-order-v4'])):
                identity = {'phase': phase, 'dataset': item['dataset'], 'q': item['q'], **cell}
                def condition_call(context, messages, temperature):
                    return call({**cell, **context}, messages, temperature)
                actual = trajectory(item, start, cell['delay'], cell['verifier_on'], spec['T'], condition_call)
                row = json.loads(next(runs)); total_runs += 1
                assert row['id'] == digest(identity) and row['identity'] == identity and row['manifest_sha256'] == mh
                assert all(row[k] == actual[k] for k in actual)
                assert row['history_indices'] == [max(0, t-1-cell['delay']) for t in range(1, spec['T'])]
                labels = [[score(item, a['answer'], a['status'], item['catalog']) for a in step] for step in actual['states']]
                assert labels == row['labels']
                tail = labels[spec['T']//2:]
                bounds = amplitude_bounds([[a['reference_error'] for a in step] for step in tail])
                assert row['amplitude_bounds'] == bounds
                labels_by_dataset[item['dataset']].update(a['label'] for step in labels for a in step)
                availability = [sum(a['status'] == 'valid' for a in step)/3 for step in actual['states'][spec['T']//2:]]
                mean = sum(availability)/len(availability)
                availability_amplitude = math.sqrt(sum((v-mean)**2 for v in availability)/len(availability))
                conditions[cell['delay'], cell['verifier_on']] = {'bounds': bounds, 'availability_amplitude': availability_amplitude}
            if phase == 'primary':
                a, b = conditions[0, True], conditions[5, True]
                pairs.append({'dataset': item['dataset'], 'question_id': item['id'], 'question_cluster': item['question_cluster'],
                              'difference_bounds': [b['bounds'][0]-a['bounds'][1], b['bounds'][1]-a['bounds'][0]],
                              'availability_amplitude_difference': b['availability_amplitude']-a['availability_amplitude']})
        assert requests.read().strip() == '' and runs.read().strip() == '', 'unexpected extra records'
    expected = len(spec['items'])*(3+sum((spec['T']-1)*3*(1+c['verifier_on']) for c in spec['cells']))
    assert total_requests == expected and total_runs == len(spec['items'])*len(spec['cells'])
    report = {'status': 'passed', 'phase': phase, 'runs': total_runs, 'requests': total_requests,
              'technical_status_counts': dict(counts), 'max_prompt_tokens': max_prompt, 'manifest_sha256': mh,
              'verifier_sha256': file_hash(Path(__file__)),
              'files_sha256': {n: file_hash(directory/n) for n in ('manifest.json', 'requests.jsonl', 'runs.jsonl')}}
    if phase == 'primary':
        report['reference_label_counts'] = {d: dict(v) for d, v in labels_by_dataset.items()}
        report['datasets'] = {}
        for d in ('psilo', 'tqa'):
            selected = [p for p in pairs if p['dataset'] == d]
            assert len(selected) == 200
            report['datasets'][d] = {'questions': len(selected), 'clusters': len({p['question_cluster'] for p in selected}),
                'mean_difference_outer_bounds': [sum(p['difference_bounds'][i] for p in selected)/len(selected) for i in range(2)],
                'mean_availability_amplitude_difference': sum(p['availability_amplitude_difference'] for p in selected)/len(selected)}
        report['interpretation'] = 'Complete fixed grid. Conservative bounds for unresolved reference labels; no imputed factual labels, no point-estimate significance test, no independence claim for shared templates.'
        (directory/'paired_ranges.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
    (directory/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    return report


if __name__ == '__main__': verify(sys.argv[1])
