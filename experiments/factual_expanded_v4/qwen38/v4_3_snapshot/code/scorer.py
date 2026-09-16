"""Conservative, question-bound reference scoring; not a universal truth judge.

Unknown paraphrases remain unresolved. NLI probabilities never determine labels.
"""
import hashlib
import json
import unicodedata

VERSION = 'reference-v1'
ABSTENTIONS = frozenset({"i don't know", 'i do not know', 'i have no comment',
                        'i cannot answer', "i can't answer"})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize(text):
    if not isinstance(text, str):
        raise TypeError('answer must be a string')
    text = unicodedata.normalize('NFC', text).replace('\u2019', "'")
    text = ' '.join(text.casefold().split())
    # Preserve negation, signs, decimal separators, accents and uncertainty marks.
    if text.endswith('.') and not text.endswith('..'):
        text = text[:-1].rstrip()
    return text


def identity(item):
    return {k: item[k] for k in ('dataset', 'q', 'gold', 'evidence')}


def validate_entry(entry):
    if entry['item_sha256'] != digest(entry['identity']):
        raise ValueError('catalog identity checksum mismatch')
    if entry['version'] != VERSION:
        raise ValueError('unsupported reference catalog version')
    maps = []
    for key in ('accepted', 'rejected'):
        mapping = {}
        for alias in entry[key]:
            value = normalize(alias['text'])
            if not value or value == 'null' or not alias.get('basis'):
                raise ValueError('invalid or undocumented catalog alias')
            mapping[value] = alias
        maps.append(mapping)
    if set(maps[0]) & set(maps[1]):
        raise ValueError('reference aliases have conflicting labels')
    if normalize(entry['identity']['gold']) not in maps[0]:
        raise ValueError('catalog must retain the supplied reference')
    return maps


def score(item, answer, technical_status, entry):
    accepted, rejected = validate_entry(entry)
    if identity(item) != entry['identity']:
        raise ValueError('reference catalog belongs to a different question or source')
    result = {'scorer_version': VERSION, 'catalog_entry_sha256': digest(entry),
              'label': None, 'reference_error': None, 'basis': None}
    def finish(label, basis, error=None):
        return dict(result, label=label, basis=basis, reference_error=error)
    if technical_status == 'abstention':
        if answer is not None:
            raise ValueError('abstention must have a null parsed answer')
        return finish('abstention', 'explicit_json_null')
    if technical_status != 'valid':
        return finish('invalid_output', 'technical_status:'+technical_status)
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('valid technical output requires a nonempty string')
    key = normalize(answer)
    if key == 'null':
        return finish('invalid_output', 'quoted_null_placeholder')
    # A source may call a refusal "correct"; track response availability separately.
    if key in ABSTENTIONS:
        return finish('abstention', 'explicit_textual_abstention')
    if key in accepted:
        return finish('matches_reference', accepted[key]['basis'], 0)
    if key in rejected:
        return finish('contradicts_reference', rejected[key]['basis'], 1)
    return finish('unresolved', 'not_in_question_bound_reference_catalog')
