"""Freeze the residual-pair review, preserving the complete v1 ledger."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'semantic_review_20260915' / 'review_frozen.json'
sys.path.insert(0, str(HERE.parent / 'code_v4_3'))
from scorer import normalize

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def merge_review(previous, packet, decisions):
    assert set(decisions) == {str(q['number']) for q in packet}
    residual = {a['pair_id']: a for a in previous['answers'] if a['label'] == 'unresolved'}
    reviewed = {}
    for q in packet:
        decision = decisions[str(q['number'])]
        assert decision['question_id'] == q['question_id']
        assert len(q['answers']) == len(decision['answers'])
        for a, d in zip(q['answers'], decision['answers']):
            assert a['pair_id'] == d['pair_id'] and a['text'] == d['answer']
            assert a['pair_id'] not in reviewed
            old = residual[a['pair_id']]
            assert old['question_id'] == q['question_id'] and old['answer'] == a['text']
            assert d['label'] in ('matches_reference', 'contradicts_reference', 'unresolved')
            reviewed[a['pair_id']] = (d, decision['reason'])
    assert set(reviewed) == set(residual)
    final = []
    normalized = {}
    for old in previous['answers']:
        new = dict(old)
        new['v1_label'] = old['label']
        new['v2_reviewed'] = old['pair_id'] in reviewed
        if new['v2_reviewed']:
            d, reason = reviewed[old['pair_id']]
            new.update(label=d['label'], basis=reason,
                       sources=['SOURCES.json', 'frozen_packet:' + str(old['review_group'])],
                       v1_basis=old['basis'], v1_sources=old['sources'])
        key = (new['question_id'], normalize(new['answer']))
        assert key not in normalized or normalized[key] == new['label'], (new['review_group'], key)
        normalized[key] = new['label']
        final.append(new)
    assert len(final) == len({a['pair_id'] for a in final}) == len(previous['answers'])
    return final

def main():
    out = HERE / 'review_frozen.json'
    assert not out.exists(), 'refuse overwrite frozen review'
    previous = json.loads(PREVIOUS.read_text())
    assert sha(PREVIOUS) == '63ec170a5af9374199a64deb0d8a62e4bc94dc0270c96e596a55cba81bd2fbb5'
    packet = json.loads((HERE / 'packet.json').read_text())
    decisions = json.loads((HERE / 'decisions.json').read_text())
    final = merge_review(previous, packet, decisions)
    assert len(packet) == 150 and len(final) == 3604
    changed = [a for a in final if a['v2_reviewed']]
    assert len(changed) == 1027
    inputs = [HERE / n for n in ('packet.json', 'decisions.json', 'PROTOCOL_RU.md', 'SOURCES.json', 'freeze.py', 'record.py')]
    inputs += sorted(HERE.glob('batch*.json'))
    hashes = {p.name: sha(p) for p in inputs}
    hashes['../semantic_review_20260915/review_frozen.json'] = sha(PREVIOUS)
    result = {
        'version': 'semantic-assistant-v2',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reviewer': 'same assistant; post-hoc semantic refinement, not independent human adjudication',
        'questions': previous['questions'], 'pairs': len(final),
        'v2_reviewed_questions': len(packet), 'v2_reviewed_pairs': len(changed),
        'counts': dict(collections.Counter(a['label'] for a in final)),
        'v2_residual_decisions': dict(collections.Counter(a['label'] for a in changed)),
        'input_sha256': hashes, 'answers': final,
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (HERE / 'remaining_unresolved.json').write_text(json.dumps(
        [a for a in final if a['label'] == 'unresolved'], ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'answers'}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
