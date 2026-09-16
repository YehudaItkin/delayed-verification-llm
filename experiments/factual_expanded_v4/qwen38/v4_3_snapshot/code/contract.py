"""Explicit answer/abstention alternatives, without regex-based JSON grammar."""
import json
import unicodedata

SCHEMA = {'oneOf': [
    {'type': 'object', 'properties': {'status': {'const': 'abstention'}},
     'required': ['status'], 'additionalProperties': False},
    {'type': 'object', 'properties': {'status': {'const': 'answer'},
                                    'answer': {'type': 'string', 'minLength': 1, 'maxLength': 80}},
     'required': ['status', 'answer'], 'additionalProperties': False},
]}
INSTRUCTION = (
    'Return only JSON matching this schema: '+json.dumps(SCHEMA, separators=(',', ':'))
    + '. If you can answer, return {"status":"answer","answer":"your short factual answer"}. '
    'If you cannot answer, return exactly {"status":"abstention"} without an answer field. '
    'Do not put a missing-value placeholder in the answer field. '
    'Do not copy a peer answer merely because it appears in the prompt. No explanation.'
)


def structured_body(base, backend):
    body = dict(base)
    if backend == 'ollama': body['format'] = SCHEMA
    elif backend == 'vllm': body['structured_outputs'] = {'json': SCHEMA}
    else: raise ValueError('unsupported backend')
    return body


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('duplicate key')
        result[key] = value
    return result


def parse_answer(content, finish_reason):
    if finish_reason == 'length': return {'status': 'truncated', 'answer': None}
    if finish_reason != 'stop': return {'status': 'unexpected_finish', 'answer': None}
    if not isinstance(content, str): return {'status': 'non_text', 'answer': None}
    if not content.strip(): return {'status': 'empty_generation', 'answer': None}
    try:
        value = json.loads(content, object_pairs_hook=_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('not JSON')))
    except (ValueError, TypeError): return {'status': 'invalid_json', 'answer': None}
    if isinstance(value, dict) and value == {'status': 'abstention'}:
        return {'status': 'abstention', 'answer': None}
    if not isinstance(value, dict) or set(value) != {'status', 'answer'} or value['status'] != 'answer':
        return {'status': 'invalid_schema', 'answer': None}
    answer = value['answer']
    if (not isinstance(answer, str) or not 1 <= len(answer) <= 80 or not answer.strip()
        or any(unicodedata.category(c).startswith('C') or c in '\r\n\u2028\u2029' for c in answer)):
        return {'status': 'invalid_answer', 'answer': None}
    return {'status': 'valid', 'answer': answer.strip()}
