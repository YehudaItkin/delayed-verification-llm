from pathlib import Path
import collections, datetime, hashlib, json, sys
from record import expand, LABELS
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'code_v4_3'))
from scorer import normalize

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out = HERE/'review_frozen.json'
    assert not out.exists(), 'refuse overwrite frozen review'
    packet = json.loads((HERE/'packet.json').read_text())
    decisions = json.loads((HERE/'decisions.json').read_text())
    amendments = json.loads((HERE/'amendments.json').read_text())
    assert set(decisions) == {str(q['number']) for q in packet}
    final = []
    for q in packet:
        d = decisions[str(q['number'])]
        assert d['question_id'] == q['question_id'] and len(d['answers']) == len(q['answers'])
        amendment = amendments.get(str(q['number']), {})
        overrides = {}
        for label in LABELS:
            for number in expand(amendment.get(label, '')):
                assert 1 <= number <= len(q['answers']) and number not in overrides
                overrides[number] = LABELS[label]
        normalized = {}
        for a, recorded in zip(q['answers'], d['answers']):
            assert a['pair_id'] == recorded['pair_id'] and a['text'] == recorded['answer']
            label = overrides.get(a['number'], recorded['label'])
            key = normalize(a['text'])
            assert key not in normalized or normalized[key] == label, (q['number'], key)
            normalized[key] = label
            final.append({'pair_id': a['pair_id'], 'question_id': q['question_id'],
                'answer': a['text'], 'label': label, 'first_pass_label': recorded['label'],
                'basis': amendment['reason'] if a['number'] in overrides else d['reason'],
                'sources': amendment.get('sources', []) if a['number'] in overrides else ['frozen_packet:'+str(q['number'])],
                'review_group': q['number']})
    assert len(final) == 3604 and len({a['pair_id'] for a in final}) == 3604
    files = [HERE/n for n in ('packet.json','decisions.json','amendments.json','PROTOCOL_RU.md')]
    files += sorted(HERE.glob('batch*.json'))
    report = {'version': 'semantic-assistant-v1', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reviewer': 'same assistant; condition-hidden packet, not independent human adjudication',
        'questions': len(packet), 'pairs': len(final), 'counts': dict(collections.Counter(a['label'] for a in final)),
        'input_sha256': {p.name: sha(p) for p in files}, 'answers': final}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'answers'}))

if __name__ == '__main__': main()
