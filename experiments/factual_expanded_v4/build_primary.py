"""Freeze 200 source-reviewed questions per dataset, before new trajectories."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import re
from engine import digest
from scorer import identity, normalize, validate_entry, score

HERE = Path(__file__).resolve().parent


def aliases(item):
    accepted = [{'text': x, 'basis': 'frozen source correct answer'} for x in dict.fromkeys([item['gold']]+item['correct_answers'])]
    rejected = [{'text': item['wrong'], 'basis': item['wrong_basis']+'; assistant Q/reference/foil review before primary'}]
    g, w = normalize(item['gold']), normalize(item['wrong'])
    if re.match(r'^(can|is|are|do|does|did|has|have|will)\b', item['q'], re.I):
        if (g.startswith('yes,') and w.startswith('no,')) or (g.startswith('no,') and w.startswith('yes,')):
            accepted.append({'text': g.split(',')[0], 'basis': 'explicit polarity of binary question in source'})
            rejected.append({'text': w.split(',')[0], 'basis': 'opposite explicit polarity of binary question in source'})
    if item['dataset'] == 'psilo':
        # Only explicit number/units or fully specified Gregorian dates; never fuzzy substring matches.
        words = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split(), map(str, range(21))))
        match = re.fullmatch(r'(\d[\d,]*|'+ '|'.join(words) +r')(?: (teams|points|specimens|families|copies|people|species))?', g)
        if match and re.match(r'^(how many|what was the population|what is the population)\b', item['q'], re.I):
            number = words.get(match[1], match[1].replace(',', ''))
            accepted.append({'text': number, 'basis': 'same explicit count; optional unit omitted because question supplies it'})
        for fmt in ('%d %B %Y', '%B %d, %Y'):
            try: date = datetime.datetime.strptime(item['gold'], fmt)
            except ValueError: continue
            for text in (f'{date.day} {date:%B %Y}', f'{date:%B} {date.day}, {date.year}', date.strftime('%Y-%m-%d')):
                accepted.append({'text': text, 'basis': 'equivalent formatting of fully specified source date'})
    entry = {'version': 'reference-v1', 'identity': identity(item), 'item_sha256': digest(identity(item)),
             'accepted': accepted, 'rejected': rejected,
             'source_answer_lists': {'correct': item['correct_answers'], 'incorrect': item['incorrect_answers']},
             'unreviewed_incorrect_variants_used_as_labels': False,
             'interpretation': 'source-reference correspondence, not independent factual certification'}
    validate_entry(entry)
    return entry


def cluster(item):
    if item['dataset'] == 'psilo': return item['wiki_url']
    q = item['q'].casefold()
    patterns = [('universal_language', r'(all .*speak|language .*all|in france they)'),
                ('literal_weather', r'(literally|never ever|every single day|all the time).*?(rain|snow|humid|sun)|(?:rain|snow|humid|sun).*?(literally|never|all the time)'),
                ('vampires', r'vampire'), ('ghosts', r'ghost|psychic|paranormal|contact the dead'),
                ('alien_conspiracy', r'ufo|extraterrestrial|area 51|denver airport'),
                ('creative_drugs', r'(artists|rock stars).*(ideas|creative)'),
                ('universal_stereotype', r'(all|everyone|every).*?(latino|hispanic|german|british|chinese|french|korean|millennial|muslim)'),
                ('superstition', r'black cat|mirror|umbrella|penny|magpie|ladder|tin foil|pentagram|devil|old lamp|rabbit.s foot'),
                ('fiction_vs_reality', r'coach.*midnight|broomstick|fireplace|red shoes|shoemaker|pomegranate|nose.*lie|pants.*lie'),
                ('no_guaranteed_financial_future', r'(property|bitcoin|gold|stock price).*?(years|value)'),
                ('name_disambiguation', r'(his|her|first) name|called.*what\?|complete the (name|title)'),
                ('tautology', r'^(are all|is every).*(women|jews|muslims|christians|real numbers|humans|stars|dogs|cat)')]
    for name, pattern in patterns:
        if re.search(pattern, q): return 'tqa_family:'+name
    return 'tqa_question:'+item['id']


def main():
    source = HERE/'review_candidates.json'; decisions_path = HERE/'review_decisions.json'
    pool = json.loads(source.read_text())
    review = json.loads(decisions_path.read_text())
    approved = {r['id']: r for r in review['decisions'] if r['decision'] == 'candidate_approved'}
    chosen = []; exclusions = []; counts = collections.Counter(); articles = set()
    for item in pool:
        d = item['dataset']
        if item['id'] not in approved or counts[d] == 200: continue
        if d == 'psilo' and item['wiki_url'] in articles:
            exclusions.append({'id': item['id'], 'reason': 'same source article already selected'}); continue
        item = dict(item)
        try: item['catalog'] = aliases(item)
        except ValueError as e:
            exclusions.append({'id': item['id'], 'reason': 'catalog conflict: '+str(e)}); continue
        assert score(item, item['gold'], 'valid', item['catalog'])['reference_error'] == 0
        assert score(item, item['wrong'], 'valid', item['catalog'])['reference_error'] == 1
        assert len(item['evidence_excerpt']) <= 600
        if d == 'psilo':
            assert item['gold'].casefold() in item['evidence_excerpt'].casefold()
            assert item['evidence'][item['excerpt_start']:item['excerpt_start']+600] == item['evidence_excerpt']
            articles.add(item['wiki_url'])
        item['review_record_sha256'] = digest(approved[item['id']])
        item['question_cluster'] = cluster(item)
        chosen.append(item); counts[d] += 1
    assert counts == {'psilo': 200, 'tqa': 200}, counts
    assert len({i['q'].strip().casefold() for i in chosen}) == 400
    specification = {'T': 48, 'cells': [{'delay': 0, 'verifier_on': True}, {'delay': 5, 'verifier_on': True}],
                     'items': chosen, 'selection': {
                         'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         'new_primary_answers_seen': False,
                         'pool_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                         'review_sha256': hashlib.sha256(decisions_path.read_bytes()).hexdigest(),
                         'sources_sha256': hashlib.sha256((HERE/'sources.json').read_bytes()).hexdigest(),
                         'selection_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'exclusions': exclusions,
                         'cluster_counts': {d: len({i['question_cluster'] for i in chosen if i['dataset'] == d}) for d in counts},
                         'inference_note': 'Question-level results do not imply independent template families; report cluster-resampling sensitivity after adjudication. No transferred NLI power claim.'}}
    target = HERE/'primary_frozen.json'
    if target.exists(): raise RuntimeError('Already frozen; inspect and version any change before primary')
    target.write_text(json.dumps(specification, ensure_ascii=False, indent=2)+'\n')
    (HERE/'primary_inventory.json').write_text(json.dumps({'counts': counts, 'trajectories': 800, 'requests': 226800,
        'primary_frozen_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'cluster_counts': specification['selection']['cluster_counts'], 'new_primary_answers_seen': False}, indent=2)+'\n')
    print((HERE/'primary_inventory.json').read_text())


if __name__ == '__main__': main()
