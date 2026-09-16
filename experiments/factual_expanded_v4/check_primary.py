"""Check frozen question origins, exclusions, reviewed foils, and source catalogs."""
from pathlib import Path
import collections
import hashlib
import json
import pyarrow.parquet as pq
from engine import digest
from scorer import validate_entry, score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    study = json.loads((HERE/'primary_frozen.json').read_text())
    sources = json.loads((HERE/'sources.json').read_text())
    tables = {}
    for d, metadata in sources.items():
        source = ROOT/metadata['local_path']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == metadata['sha256']
        tables[d] = pq.read_table(source).to_pylist()
    historical = {r['q'].strip().casefold() for p in (ROOT/'paper/remote').glob('selected*.json') for r in json.loads(p.read_text())}
    historical.update(r['q'].strip().casefold() for p in (ROOT/'paper/remote').glob('f1_*.json') for r in json.loads(p.read_text())['runs'])
    review = {r['id']: r for r in json.loads((HERE/'review_decisions.json').read_text())['decisions']}
    assert collections.Counter(i['dataset'] for i in study['items']) == {'psilo': 200, 'tqa': 200}
    assert len({i['id'] for i in study['items']}) == 400
    articles = set()
    for item in study['items']:
        d = item['dataset']; raw = tables[d][item['source_row']]
        assert item['q'].strip().casefold() not in historical
        assert raw['question'].strip() == item['q']
        assert raw['golden_answer' if d == 'psilo' else 'best_answer'].strip() == item['gold']
        assert item['source_revision'] == sources[d]['revision']
        assert review[item['id']]['decision'] == 'candidate_approved'
        assert digest(review[item['id']]) == item['review_record_sha256']
        validate_entry(item['catalog'])
        assert score(item, item['gold'], 'valid', item['catalog'])['reference_error'] == 0
        assert score(item, item['wrong'], 'valid', item['catalog'])['reference_error'] == 1
        if d == 'psilo':
            assert item['wiki_url'] == raw['wiki_url'] and item['wiki_url'] not in articles
            articles.add(item['wiki_url'])
            assert item['evidence'] == raw['wiki_passage']
            assert item['evidence_excerpt'] == item['evidence'][item['excerpt_start']:item['excerpt_start']+600]
            assert item['wrong'] in raw['annotated_span']
        else:
            assert item['correct_answers'] == list(raw['correct_answers'])
            assert item['incorrect_answers'] == list(raw['incorrect_answers'])
            assert item['wrong'] in raw['incorrect_answers']
    report = {'status': 'passed', 'questions': 400, 'historical_overlap': 0,
              'unique_psilo_articles': len(articles), 'reference_catalogs_checked': 400,
              'source_hashes_verified': {d: m['sha256'] for d, m in sources.items()},
              'primary_frozen_sha256': hashlib.sha256((HERE/'primary_frozen.json').read_bytes()).hexdigest(),
              'selection_and_analysis_sha256': hashlib.sha256((HERE/'SELECTION_AND_ANALYSIS_RU.md').read_bytes()).hexdigest()}
    (HERE/'primary_preflight.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
