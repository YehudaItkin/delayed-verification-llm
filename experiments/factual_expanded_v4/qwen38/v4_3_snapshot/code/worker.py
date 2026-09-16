"""Durable, replayable expanded-study worker; no historical runner imports."""
from pathlib import Path
import collections
import datetime
import fcntl
import hashlib
import json
import os
import platform
import sys
import time
import requests
from contract import structured_body
from engine import digest, seed, parse, initial, trajectory, amplitude_bounds
from model_api import inspect_model, request_body
from scorer import score, validate_entry


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def append(path, obj):
    with path.open('a') as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())


def phase(config, specification, phase_name, out):
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    source = Path(__file__).resolve().parent
    metadata = inspect_model(requests.Session(), config['api'], config['model'], config['backend'])
    assert metadata['model']['digest'] == config['model_digest']
    for item in specification['items']:
        validate_entry(item['catalog'])
        assert len(item['wrong']) <= 80 and item['wrong'].strip()
        assert score(item, item['gold'], 'valid', item['catalog'])['label'] == 'matches_reference'
        assert score(item, item['wrong'], 'valid', item['catalog'])['label'] == 'contradicts_reference'
    files = [source/name for name in ('worker.py', 'engine.py', 'contract.py',
                                     'model_api.py', 'scorer.py', 'PROTOCOL_RU.md')]
    manifest = {'version': 'expanded-v4.3', 'phase': phase_name, 'specification': specification,
                'config': config, 'server': metadata, 'python': platform.python_version(),
                'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'reference_evaluation': 'question-bound source-relative; unresolved is not an error label',
                'max_output_tokens': 256, 'num_ctx': 4096}
    if (out/'manifest.json').exists():
        assert json.loads((out/'manifest.json').read_text()) == manifest, 'manifest changed'
    else:
        atomic(out/'manifest.json', manifest)
    state = {'status': 'running', 'phase': phase_name, 'pid': os.getpid(), 'started_or_resumed_utc': now()}
    atomic(out/'status.json', state)
    cache = {}
    for r in rows(out/'requests.jsonl'):
        assert r['id'] == digest(r['identity']) and r['id'] not in cache
        assert r['seed'] == seed(r['identity'])
        assert r['parsed'] == parse(r['response']['message']['content'], r['response']['done_reason'])
        assert r['usage']['prompt_tokens']+r['usage']['completion_tokens'] <= 4096
        assert r['usage']['prompt_tokens'] <= 3500
        cache[r['id']] = r
    completed = rows(out/'runs.jsonl')
    done = {r['id']: r for r in completed}
    assert len(done) == len(completed)
    session = requests.Session()
    seen = set()
    try:
        for item in specification['items']:
            assert inspect_model(session, config['api'], config['model'], config['backend']) == metadata
            def call(context, messages, temperature):
                identity = {'phase': phase_name, 'dataset': item['dataset'], 'q': item['q'], **context}
                rid = digest(identity)
                body = structured_body(request_body(config['model'], messages, 256, temperature,
                        seed(identity), config['backend'], 4096), config['backend'])
                seen.add(rid)
                if rid in cache:
                    record = cache[rid]
                    assert record['identity'] == identity and record['request'] == body, 'replay request mismatch'
                    return record['parsed']
                data = None
                started = time.monotonic()
                try:
                    response = session.post(config['api'], json=body, timeout=180)
                    try: data = response.json()
                    except ValueError: pass
                    response.raise_for_status()
                    if not isinstance(data, dict) or data.get('done') is not True:
                        raise ValueError('incomplete response')
                    content = data['message']['content']
                    reason = data['done_reason']
                    if not isinstance(content, str) or reason not in ('stop', 'length'):
                        raise ValueError('invalid response shape or termination')
                    usage = {'prompt_tokens': data['prompt_eval_count'], 'completion_tokens': data.get('eval_count', 0)}
                    record = {'id': rid, 'identity': identity, 'seed': seed(identity), 'request': body,
                              'response': data, 'parsed': parse(content, reason), 'usage': usage,
                              'seconds': time.monotonic()-started, 'finished_utc': now()}
                    append(out/'requests.jsonl', record)
                    cache[rid] = record
                    # Save even a problematic response before raising a context error.
                    if usage['prompt_tokens']+usage['completion_tokens'] > 4096 or usage['prompt_tokens'] > 3500:
                        raise ValueError('context safety limit exceeded')
                    state.update(last_request_utc=now(), requests=len(cache), completed_runs=len(done),
                                 current_question_id=item['id'])
                    atomic(out/'status.json', state)
                    return record['parsed']
                except Exception as error:
                    append(out/'errors.jsonl', {'identity': identity, 'request': body, 'response': data,
                                              'error_type': type(error).__name__, 'time_utc': now()})
                    raise
            start = initial(item, call)
            cells = sorted(specification['cells'], key=lambda c: digest([item['id'], c, 'condition-order-v4']))
            for cell in cells:
                ident = {'phase': phase_name, 'dataset': item['dataset'], 'q': item['q'], **cell}
                rid = digest(ident)
                def condition_call(context, messages, temperature):
                    return call({**cell, **context}, messages, temperature)
                # Rebuild all preceding states from raw responses even for completed rows.
                row = trajectory(item, start, cell['delay'], cell['verifier_on'], specification['T'], condition_call)
                labels = [[score(item, a['answer'], a['status'], item['catalog']) for a in step] for step in row['states']]
                errors = [[a['reference_error'] for a in step] for step in labels]
                row.update(id=rid, identity=ident, labels=labels,
                           amplitude_bounds=amplitude_bounds(errors[specification['T']//2:]),
                           manifest_sha256=digest(manifest))
                if rid in done:
                    assert done[rid] == row, 'saved trajectory differs from raw replay'
                else:
                    append(out/'runs.jsonl', row); done[rid] = row
                state.update(completed_runs=len(done), updated_utc=now())
                atomic(out/'status.json', state)
                print(json.dumps({'phase': phase_name, 'completed_runs': len(done), 'requests': len(cache)}), flush=True)
        assert seen == set(cache), 'unexpected requests'
        assert len(done) == len(specification['items'])*len(specification['cells'])
        expected = len(specification['items'])*(3+sum((specification['T']-1)*3*(1+c['verifier_on']) for c in specification['cells']))
        assert len(cache) == expected
        statuses = dict(collections.Counter(r['parsed']['status'] for r in cache.values()))
        summary = {'status': 'complete', 'phase': phase_name, 'runs': len(done), 'requests': len(cache),
                   'technical_status_counts': statuses, 'max_prompt_tokens': max(r['usage']['prompt_tokens'] for r in cache.values()),
                   'finished_utc': now(), 'manifest_sha256': digest(manifest),
                   'errors': len(rows(out/'errors.jsonl'))}
        atomic(out/'summary.json', summary)
        state.update(status='complete', finished_utc=now(), completed_runs=len(done), requests=len(cache))
        atomic(out/'status.json', state)
        return summary
    except Exception as error:
        state.update(status='stopped_for_review', error_type=type(error).__name__, stopped_utc=now())
        atomic(out/'status.json', state)
        raise
    finally:
        lock.close()


def main():
    source = Path(__file__).resolve().parent
    config = json.loads((source/'config.json').read_text())
    study = json.loads((source/'study.json').read_text())
    base = Path(config['out']); base.mkdir(parents=True, exist_ok=True)
    lock = (base/'queue.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        atomic(base/'queue_status.json', {'status': 'running', 'phase': 'pilot', 'updated_utc': now()})
        pilot = phase(config, study['pilot'], 'pilot', base/'pilot')
        assert pilot['errors'] == 0 and set(pilot['technical_status_counts']) <= {'valid', 'abstention'}, 'pilot gate failed'
        atomic(base/'pilot_gate.json', {'status': 'passed', 'pilot_summary_sha256': digest(pilot), 'time_utc': now()})
        if not study.get('primary_ready'):
            atomic(base/'queue_status.json', {'status': 'waiting_for_frozen_primary', 'phase': 'pilot_complete', 'updated_utc': now()})
            return
        assert collections.Counter(i['dataset'] for i in study['primary']['items']) == {'psilo': 200, 'tqa': 200}
        assert study['primary']['T'] == 48 and study['primary']['cells'] == [{'delay': 0, 'verifier_on': True}, {'delay': 5, 'verifier_on': True}]
        atomic(base/'queue_status.json', {'status': 'running', 'phase': 'primary', 'updated_utc': now()})
        primary = phase(config, study['primary'], 'primary', base/'primary')
        atomic(base/'queue_status.json', {'status': 'complete', 'phase': 'primary', 'summary': primary, 'updated_utc': now()})
    except Exception as error:
        atomic(base/'queue_status.json', {'status': 'stopped_for_review', 'error_type': type(error).__name__, 'updated_utc': now()})
        raise


if __name__ == '__main__': main()
