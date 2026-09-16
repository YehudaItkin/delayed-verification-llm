"""One-time activation after complete local pilot replay and frozen-input checks."""
from pathlib import Path
import collections
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import tarfile

BASE = Path('/home/itkin/factual-expanded-v4-3-20260913')
PYTHON = '/home/itkin/factual-recheck-qwen38-20260910/.venv/bin/python'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def main():
    results = BASE / 'results'
    with (results / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert read(results / 'queue_status.json')['status'] == 'waiting_for_frozen_primary'
        assert not (BASE / 'primary_activation.json').exists(), 'already activated'
        assert not (results / 'primary/manifest.json').exists(), 'primary already exists'
        proof = read(BASE / 'pilot_v4_3_verification.json')
        assert proof['status'] == 'passed' and proof['phase'] == 'pilot'
        assert proof['requests'] == 900 and proof['runs'] == 24
        assert set(proof['technical_status_counts']) <= {'valid', 'abstention'}
        for name, expected in proof['files_sha256'].items():
            assert name in ('manifest.json', 'requests.jsonl', 'runs.jsonl')
            assert sha(results / 'pilot' / name) == expected, name
        summary = read(results / 'pilot/summary.json')
        assert summary['errors'] == 0 and summary['requests'] == 900 and summary['runs'] == 24
        assert summary['technical_status_counts'] == proof['technical_status_counts']
        assert read(results / 'pilot_gate.json')['status'] == 'passed'
        for name, expected in read(results / 'pilot/manifest.json')['code_sha256'].items():
            assert sha(BASE / 'code' / name) == expected, name
        payload = read(BASE / 'primary_v4_3_payload_manifest.json')
        archive = BASE.with_name(BASE.name + '_primary_payload.tar.gz')
        assert sha(archive) == payload['archive_sha256']
        target = BASE / 'primary_inputs'
        target.mkdir(exist_ok=False)
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            assert {m.name for m in members} == set(payload['files'])
            assert all(m.isfile() and Path(m.name).name == m.name for m in members)
            for member in members:
                (target / member.name).write_bytes(tar.extractfile(member).read())
        for name, expected in payload['files'].items():
            assert sha(target / name) == expected, name
        preflight = read(target / 'primary_preflight.json')
        assert preflight['status'] == 'passed' and preflight['historical_overlap'] == 0
        assert preflight['questions'] == 400 and preflight['reference_catalogs_checked'] == 400
        assert sha(target / 'primary_frozen.json') == preflight['primary_frozen_sha256']
        assert sha(target / 'SELECTION_AND_ANALYSIS_RU.md') == preflight['selection_and_analysis_sha256']
        old = read(BASE / 'code/study.json')
        study = read(target / 'study_primary_v4_3.json')
        assert old['pilot'] == study['pilot'] and not old['primary_ready']
        assert study['version'] == 'expanded-v4.3' and study['primary_ready'] is True
        frozen = read(target / 'primary_frozen.json')
        assert all(study['primary'][key] == value for key, value in frozen.items())
        assert collections.Counter(i['dataset'] for i in frozen['items']) == {'psilo': 200, 'tqa': 200}
        assert frozen['T'] == 48 and frozen['cells'] == [{'delay': 0, 'verifier_on': True}, {'delay': 5, 'verifier_on': True}]
        assert study['primary']['launch_provenance']['primary_frozen_sha256'] == sha(target / 'primary_frozen.json')
        (BASE / 'study.pilot_only.json').write_bytes((BASE / 'code/study.json').read_bytes())
        atomic(BASE / 'code/study.json', study)
        record = {'activated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'payload': payload, 'pilot_verification': proof,
                  'study_sha256': sha(BASE / 'code/study.json'),
                  'activation_script_sha256': sha(Path(__file__)),
                  'expected_primary_questions': 400, 'expected_primary_runs': 800,
                  'expected_primary_requests': 226800}
        atomic(BASE / 'primary_activation.json', record)
    # Release the old queue lock before starting its new owner.
    with (BASE / 'worker.log').open('ab') as log:
        process = subprocess.Popen([PYTHON, '-u', str(BASE / 'code/worker.py')],
            cwd=BASE / 'code', stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    pid = {'pid': process.pid, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic(BASE / 'primary_worker_pid.json', pid)
    print(json.dumps(pid), flush=True)


if __name__ == '__main__':
    main()
