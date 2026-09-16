"""Apply frozen semantic decisions without changing collection or original labels."""
from pathlib import Path
import collections, datetime, hashlib, json, sys
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SOURCE = BASE/'qwen38/v4_3_snapshot/results/primary'
sys.path.insert(0, str(BASE/'code_v4_3'))
from engine import amplitude_bounds

def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def apply_state(question_id, answer, original, decisions):
    if original['label'] != 'unresolved':
        return {'label': original['label'], 'reference_error': original['reference_error'], 'origin': 'original'}
    decision = decisions[question_id, answer['answer']]
    assert decision['question_id'] == question_id and decision['answer'] == answer['answer']
    label = decision['label']
    return {'label': label, 'reference_error': {'matches_reference': 0, 'contradicts_reference': 1}.get(label),
            'origin': 'semantic-assistant-v2', 'pair_id': decision['pair_id']}

def main():
    review = json.loads((HERE/'review_frozen.json').read_text())
    for name, expected in review['input_sha256'].items(): assert sha(HERE/name) == expected, name
    verification = json.loads((BASE/'primary_v4_3_verification.json').read_text())
    assert verification['status'] == 'passed' and verification['runs'] == 800 and verification['requests'] == 226800
    for name, expected in verification['files_sha256'].items(): assert sha(SOURCE/name) == expected, name
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    items = {(i['dataset'], i['q']): i for i in manifest['specification']['items']}
    decisions = {(a['question_id'], a['answer']): a for a in review['answers']}
    assert len(decisions) == review['pairs'] == 3604
    results = HERE/'results'; results.mkdir(exist_ok=False)
    counts = collections.defaultdict(collections.Counter); tailcounts = collections.defaultdict(collections.Counter)
    transitions = collections.defaultdict(collections.Counter); cells = collections.defaultdict(dict)
    seen = set(); seen_rows = set(); used = set()
    with (SOURCE/'runs.jsonl').open() as f, (results/'runs.jsonl').open('w') as output:
        for line in f:
            row = json.loads(line); identity = row['identity']; dataset = identity['dataset']
            item = items[dataset, identity['q']]; qid = item['id']
            assert row['id'] not in seen_rows; seen_rows.add(row['id'])
            labels = []
            for state, old_labels in zip(row['states'], row['labels']):
                changed = []
                for answer, original in zip(state, old_labels):
                    label = apply_state(qid, answer, original, decisions)
                    if original['label'] == 'unresolved': used.add((qid, answer['answer']))
                    changed.append(label)
                    counts[dataset][label['label']] += 1
                    transitions[dataset][original['label']+' -> '+label['label']] += 1
                labels.append(changed)
            assert len(labels) == 48 and all(len(s)==3 for s in labels)
            errors = [[a['reference_error'] for a in step] for step in labels[24:]]
            tailcounts[dataset].update(a['label'] for step in labels[24:] for a in step)
            bounds = amplitude_bounds(errors)
            old = row['amplitude_bounds']
            assert bounds[0]+1e-12 >= old[0] and bounds[1] <= old[1]+1e-12
            key = (dataset, qid)
            assert identity['delay'] not in cells[key] and identity['verifier_on'] is True
            cells[key][identity['delay']] = {'bounds': bounds, 'unknown_tail_states': sum(x is None for s in errors for x in s)}
            output.write(json.dumps({'id': row['id'], 'identity': identity, 'question_id': qid,
                'labels': labels, 'amplitude_bounds': bounds, 'original_amplitude_bounds': old}, ensure_ascii=False)+'\n')
            seen.add(key)
    assert used == set(decisions) and len(seen) == 400 and len(seen_rows) == 800
    pairs = []
    for (dataset, qid), conditions in cells.items():
        assert set(conditions) == {0, 5}
        a, b = conditions[0]['bounds'], conditions[5]['bounds']
        pairs.append({'dataset': dataset, 'question_id': qid,
            'difference_outer_bounds': [b[0]-a[1], b[1]-a[0]],
            'fully_labelled_tail': all(c['unknown_tail_states']==0 for c in conditions.values())})
    report = {'status': 'passed', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'review_sha256': sha(HERE/'review_frozen.json'), 'analysis_code_sha256': sha(Path(__file__)),
        'source_files_sha256': verification['files_sha256'], 'questions': 400, 'runs': 800,
        'reviewed_unique_pairs': 3604, 'unique_pair_decisions': review['counts'],
        'all_state_label_counts': dict(counts), 'tail_state_label_counts': dict(tailcounts),
        'label_transitions': dict(transitions), 'datasets': {}}
    for dataset in ('psilo','tqa'):
        selected = [p for p in pairs if p['dataset'] == dataset]; assert len(selected)==200
        means = [sum(p['difference_outer_bounds'][i] for p in selected)/200 for i in (0,1)]
        report['datasets'][dataset] = {'questions': 200,
            'mean_difference_outer_bounds': means,
            'original_mean_difference_outer_bounds': verification['datasets'][dataset]['mean_difference_outer_bounds'],
            'fully_labelled_question_tails': sum(p['fully_labelled_tail'] for p in selected),
            'clusters': verification['datasets'][dataset]['clusters']}
        assert sum(counts[dataset].values())==57600 and sum(tailcounts[dataset].values())==28800
    report['interpretation'] = ('All questions retained. Same reviewed answer receives same label in all occurrences. '
        'Remaining unknowns/abstentions have conservative independent-cell completion outer bounds, not confidence intervals '
        'or sharp bounds under repeated-answer truth constraints. No imputed-label significance test; no independent-human validation claim.')
    (results/'pairs.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
    report['derived_files_sha256'] = {n: sha(results/n) for n in ('runs.jsonl','pairs.jsonl')}
    (results/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
