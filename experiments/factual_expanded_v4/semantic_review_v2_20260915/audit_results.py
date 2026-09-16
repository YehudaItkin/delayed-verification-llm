"""Verify actual v2 results against the preserved v1 ledger and trajectories."""
from pathlib import Path
import collections
import datetime
import hashlib
import itertools
import json

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / 'semantic_review_20260915'

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    old = json.loads((V1 / 'review_frozen.json').read_text())
    new = json.loads((HERE / 'review_frozen.json').read_text())
    for folder, review in ((V1, old), (HERE, new)):
        for name, digest in review['input_sha256'].items():
            assert sha(folder / name) == digest, name
    assert new['created_utc'] < json.loads((HERE/'results/summary.json').read_text())['created_utc']
    old_answers = {a['pair_id']: a for a in old['answers']}
    new_answers = {a['pair_id']: a for a in new['answers']}
    assert old_answers.keys() == new_answers.keys()
    transitions = collections.Counter()
    for pair_id, a in old_answers.items():
        b = new_answers[pair_id]
        assert (a['question_id'], a['answer']) == (b['question_id'], b['answer'])
        assert b['v1_label'] == a['label']
        if a['label'] != 'unresolved':
            for key, value in a.items():
                assert b[key] == value, (pair_id, key)
        transitions[a['label'] + ' -> ' + b['label']] += 1
    rows = states = 0
    state_transitions = collections.defaultdict(collections.Counter)
    with (V1/'results/runs.jsonl').open() as f, (HERE/'results/runs.jsonl').open() as g:
        for a, b in itertools.zip_longest(f, g):
            assert a is not None and b is not None
            a, b = json.loads(a), json.loads(b)
            assert a['id'] == b['id'] and a['identity'] == b['identity']
            assert a['question_id'] == b['question_id']
            assert b['amplitude_bounds'][0] >= a['amplitude_bounds'][0] - 1e-12
            assert b['amplitude_bounds'][1] <= a['amplitude_bounds'][1] + 1e-12
            assert len(a['labels']) == len(b['labels']) == 48
            for x, y in zip(a['labels'], b['labels']):
                assert len(x) == len(y) == 3
                for u, v in zip(x, y):
                    if u['label'] != 'unresolved':
                        assert (u['label'], u['reference_error']) == (v['label'], v['reference_error'])
                    state_transitions[a['identity']['dataset']][u['label']+' -> '+v['label']] += 1
                    states += 1
            rows += 1
    assert rows == 800 and states == 115200
    summaries = {version: json.loads((folder/'results/summary.json').read_text())
                 for version, folder in [('v1', V1), ('v2', HERE)]}
    for folder, s in ((V1, summaries['v1']), (HERE, summaries['v2'])):
        assert sha(folder/'review_frozen.json') == s['review_sha256']
        for name, digest in s['derived_files_sha256'].items():
            assert sha(folder/'results'/name) == digest
    s = summaries['v2']
    tail = collections.Counter()
    for counts in s['tail_state_label_counts'].values():
        tail.update(counts)
    unknown = tail['unresolved'] + tail['abstention'] + tail['invalid_output']
    result = {
        'status': 'passed', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'v1_review_sha256': sha(V1/'review_frozen.json'),
        'v2_review_sha256': sha(HERE/'review_frozen.json'),
        'rows_compared': rows, 'states_compared': states,
        'all_prior_decided_pairs_preserved': True,
        'all_original_absence_and_format_labels_preserved': True,
        'every_v2_trajectory_bound_nested_in_v1': True,
        'unique_pair_transitions_v1_to_v2': dict(transitions),
        'state_transitions_v1_to_v2': dict(state_transitions),
        'tail_unknown_cells': unknown,
        'tail_abstention_share_of_unknown_cells': tail['abstention']/unknown,
        'audit_code_sha256': sha(Path(__file__)),
    }
    (HERE/'results/audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
