"""Prepare source-only review candidates; never uses new model responses."""
from pathlib import Path
import hashlib
import json
import re
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    exclusions = {r['q'].strip().casefold() for p in (ROOT/'paper/remote').glob('selected*.json')
                  for r in json.loads(p.read_text())}
    exclusions.update(r['q'].strip().casefold() for p in (ROOT/'paper/remote').glob('f1_*.json')
                      for r in json.loads(p.read_text())['runs'])
    result = []; rejected = []; sources = {}
    for dataset in ('psilo', 'tqa'):
        meta = json.loads((ROOT/f'experiments/factual_recheck/source_metadata/{dataset}_download.json').read_text())
        path = ROOT/meta['local_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['sha256']
        sources[dataset] = meta
        table = pq.read_table(path).to_pylist()
        if dataset == 'psilo':
            source_rows = [json.loads(s)['source_row'] for s in (ROOT/'output/datasets/factual_recheck/psilo_candidates.jsonl').read_text().splitlines()]
        else:
            source_rows = list(range(len(table)))
        seen = set()
        for source_row in source_rows:
            raw = table[source_row]
            q = raw['question'].strip()
            gold = raw['golden_answer' if dataset == 'psilo' else 'best_answer'].strip()
            reason = None
            key = q.casefold()
            if key in exclusions: reason = 'historical_question'
            elif key in seen: reason = 'duplicate_question'
            elif not q or len(q) > 500 or not gold or len(gold) > 80: reason = 'short_answer_contract'
            seen.add(key)
            item = {'dataset': dataset, 'q': q, 'gold': gold, 'source_row': source_row,
                    'source_revision': meta['revision']}
            if dataset == 'psilo':
                evidence = raw['wiki_passage']
                spans = re.findall(r'\[HAL\](.*?)\[/HAL\]', raw['annotated_span'], re.S)
                if len(spans) != 1 or not 1 <= len(spans[0]) <= 80 or '\n' in spans[0]:
                    reason = reason or 'no_single_short_source_hallucination_span'
                wrong = spans[0] if spans else ''
                accepted = [gold]; rejected_answers = [wrong]
                start = evidence.casefold().find(gold.casefold())
                if start < 0: reason = reason or 'gold_not_literal_in_passage'
                lo = max(0, start-200)
                excerpt = evidence[lo:lo+600]
                item.update(source_id=raw['id'], wiki_url=raw['wiki_url'], wiki_title=raw['wiki_title'],
                            source_llm_answer=raw['llm_answer'], source_annotated_span=raw['annotated_span'],
                            wrong_basis='single PsiloQA source HAL span; requires question-specific review', excerpt_start=lo)
            else:
                accepted = list(raw['correct_answers'])
                rejected_answers = list(raw['incorrect_answers'])
                keys = {x.casefold().strip().rstrip('.') for x in accepted+[gold]}
                alternatives = [s for s in rejected_answers if 1 <= len(s) <= 80 and s.casefold().strip().rstrip('.') not in keys]
                wrong = alternatives[0] if alternatives else ''
                if not wrong: reason = reason or 'no_short_source_distractor'
                evidence = f"The correct answer is {gold}. Also acceptable: {'; '.join(accepted[:4])}."
                excerpt = evidence[:600]
                item.update(wrong_basis='TruthfulQA incorrect_answers; requires question-specific review', excerpt_start=0)
            if wrong.casefold().strip().rstrip('.') == gold.casefold().strip().rstrip('.'):
                reason = reason or 'same_gold_and_distractor'
            if gold.casefold().strip().rstrip('.') in {'i have no comment', 'the question is ambiguous'}:
                reason = reason or 'nonfactual_or_unanswerable_reference'
            if re.search(r'\b(current|currently|right now|still banned|next for|dating who|most recent|over time)\b', q, re.I):
                reason = reason or 'undated_time_sensitive_question'
            item.update(evidence=evidence, evidence_excerpt=excerpt, correct_answers=accepted,
                        incorrect_answers=rejected_answers, wrong=wrong)
            item['id'] = hashlib.sha256((dataset+'\n'+q).encode()).hexdigest()
            item['order'] = hashlib.sha256(('expanded-v4-20260913\n'+item['id']).encode()).hexdigest()
            if reason: rejected.append({'dataset': dataset, 'source_row': source_row, 'q': q, 'reason': reason})
            else: result.append(item)
    result.sort(key=lambda x: x['order'])
    (HERE/'review_candidates.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (HERE/'pool_exclusions.json').write_text(json.dumps(rejected, ensure_ascii=False, indent=2)+'\n')
    (HERE/'sources.json').write_text(json.dumps(sources, indent=2)+'\n')
    for d in ('psilo', 'tqa'):
        items = [x for x in result if x['dataset'] == d]
        (HERE/(d+'_review.txt')).write_text(''.join(f"{i} [{x['source_row']}] {x['q']} | {x['gold']} | {x['wrong']}\n" for i,x in enumerate(items)))
        print(d, len(items))


if __name__ == '__main__': main()
