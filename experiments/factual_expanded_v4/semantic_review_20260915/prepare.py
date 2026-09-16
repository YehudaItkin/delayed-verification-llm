from pathlib import Path
import collections, hashlib, json

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SOURCE = BASE / 'qwen38/v4_3_snapshot/results/primary'

def digest(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def main():
    assert not (HERE/'packet.json').exists()
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    items = {(i['dataset'], i['q']): i for i in manifest['specification']['items']}
    answers = collections.defaultdict(set)
    with (SOURCE/'runs.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            for state, labels in zip(row['states'], row['labels']):
                for answer, label in zip(state, labels):
                    if label['label'] == 'unresolved':
                        answers[row['identity']['dataset'], row['identity']['q']].add(answer['answer'])
    packet = []
    for number, key in enumerate(sorted(answers, key=lambda k: digest(['semantic-review-order-v1', k])), 1):
        item = items[key]
        packet.append({'number': number, 'question_id': item['id'], 'question': item['q'],
            'reference': item['gold'], 'evidence': item['evidence'],
            'accepted': [x['text'] for x in item['catalog']['accepted']],
            'rejected': [x['text'] for x in item['catalog']['rejected']],
            'answers': [{'number': n, 'pair_id': digest([item['id'], a]), 'text': a}
                for n, a in enumerate(sorted(answers[key], key=lambda a: digest(['answer-order-v1', item['id'], a])), 1)]})
    (HERE/'packet.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2)+'\n')
    meta = {'questions': len(packet), 'unique_pairs': sum(len(q['answers']) for q in packet),
        'packet_sha256': hashlib.sha256((HERE/'packet.json').read_bytes()).hexdigest(),
        'source_verification': json.loads((BASE/'primary_v4_3_verification.json').read_text()),
        'blinding': 'No model, dataset label, condition, step, agent, frequency, or amplitude in review packet. Reviewer previously saw aggregate results and selected examples; not independent fully blinded human adjudication.'}
    (HERE/'preparation.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps({k: meta[k] for k in ('questions', 'unique_pairs', 'packet_sha256')}))

if __name__ == '__main__': main()
