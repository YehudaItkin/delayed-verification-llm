"""Record explicitly inspected batches; never infer unreviewed labels."""
from pathlib import Path
import datetime, hashlib, json, sys

HERE = Path(__file__).resolve().parent
LABELS = {'M': 'matches_reference', 'C': 'contradicts_reference', 'U': 'unresolved', 'A': 'abstention'}

def expand(text):
    out = set()
    for part in text.split(','):
        if not part: continue
        values = part.split('-')
        out.update(range(int(values[0]), int(values[-1])+1))
    return out

def main():
    batch_path = Path(sys.argv[1]); batch = json.loads(batch_path.read_text())
    packet = json.loads((HERE/'packet.json').read_text())
    decisions_path = HERE/'decisions.json'
    decisions = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}
    for number, spec in batch.items():
        q = packet[int(number)-1]
        assert spec['default'] in LABELS and spec['reason'].strip()
        assert number not in decisions, 'refuse silent overwrite'
        overrides = {}; valid = {a['number'] for a in q['answers']}
        for label in LABELS:
            for a in expand(spec.get(label, '')):
                assert a in valid and a not in overrides
                overrides[a] = label
        decisions[number] = {'question_id': q['question_id'], 'reviewer': 'assistant',
            'reviewed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'batch_sha256': hashlib.sha256(batch_path.read_bytes()).hexdigest(),
            'reason': spec['reason'],
            'answers': [{'pair_id': a['pair_id'], 'answer': a['text'],
                         'label': LABELS[overrides.get(a['number'], spec['default'])]}
                        for a in q['answers']]}
    tmp = decisions_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(decisions, ensure_ascii=False, indent=2)+'\n'); tmp.replace(decisions_path)
    print('Recorded questions:', len(decisions), 'pairs:', sum(len(d['answers']) for d in decisions.values()))

if __name__ == '__main__': main()
