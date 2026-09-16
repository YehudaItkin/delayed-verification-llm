"""Pure corrected debate state machine; transport and scoring are external."""
import hashlib
import json
from contract import INSTRUCTION, parse_answer
from scorer import ABSTENTIONS, normalize


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def parse(content, reason):
    result = parse_answer(content, reason)
    if result['status'] == 'valid':
        key = normalize(result['answer'])
        if key == 'null':
            return {'status': 'invalid_placeholder', 'answer': None}
        if key in ABSTENTIONS:
            return {'status': 'abstention', 'answer': None}
    return result


def seed(identity):
    return int(digest(['expanded-v4', identity])[:8], 16) % (2**31-1)


def prompt(text):
    return [{'role': 'system', 'content': INSTRUCTION}, {'role': 'user', 'content': text}]


def initial(item, call):
    return [call({'role': 'initial', 'agent': i}, prompt('Question: '+item['q']), .7)
            for i in range(3)]


def trajectory(item, start, delay, verifier_on, horizon, call):
    if len(start) != 3 or delay < 0 or horizon < 2:
        raise ValueError('invalid state dimensions')
    states = [[dict(s) for s in start]]
    notes = []
    indices = []
    for t in range(1, horizon):
        index = max(0, t-1-delay)
        indices.append(index)
        current = states[-1]
        stale = states[index]
        verdicts = []
        for i in range(3):
            if verifier_on:
                text = ('Question: '+item['q']+'\nReference evidence: '+item['evidence_excerpt']
                        +'\nClaim to check: '+json.dumps(stale[i], ensure_ascii=False)
                        +'\nReturn the answer supported by this evidence; choose abstention if it does not determine an answer.')
                note = call({'role': 'verifier', 'step': t, 'agent': i}, prompt(text), 0.)
            else:
                note = {'status': 'disabled', 'answer': None}
            verdicts.append(note)
        updates = []
        for i in range(3):
            peers = [current[j] for j in range(3) if j != i]
            peers += [{'status': 'valid', 'answer': item['wrong']} for _ in range(4)]
            text = ('Question: '+item['q']+'\nYour previous answer: '+json.dumps(current[i], ensure_ascii=False)
                    +'\nPeers\' latest answers: '+json.dumps(peers, ensure_ascii=False)
                    +'\nVerifier answer (may refer to an EARLIER claim): '+json.dumps(verdicts[i], ensure_ascii=False)
                    +'\nGive your own current short answer. A null answer or unavailable response is not a factual claim.')
            updates.append(call({'role': 'agent', 'step': t, 'agent': i}, prompt(text), .7))
        notes.append(verdicts)
        states.append(updates)
    return {'states': states, 'verifier_notes': notes, 'history_indices': indices}


def amplitude_bounds(labels):
    """Exact min/max population SD over all binary completions (three agents)."""
    options = []
    for row in labels:
        if len(row) != 3 or any(x not in (0, 1, None) for x in row):
            raise ValueError('expected three binary or unknown labels')
        lo = sum(x == 1 for x in row)
        options.append(range(lo, lo+sum(x is None for x in row)+1))
    if not options:
        raise ValueError('empty trajectory')
    minimum = {0: 0}
    maximum = {0: 0}
    for values in options:
        low, high = {}, {}
        for total in minimum:
            for x in values:
                s = total+x
                low[s] = min(low.get(s, float('inf')), minimum[total]+x*x)
                high[s] = max(high.get(s, -1), maximum[total]+x*x)
        minimum, maximum = low, high
    n = len(options)
    def variance(s, ss):
        return max(0., ss/(9*n)-(s/(3*n))**2)
    return [min(variance(s, ss) for s, ss in minimum.items())**.5,
            max(variance(s, ss) for s, ss in maximum.items())**.5]
