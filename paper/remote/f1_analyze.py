"""Exploratory F1 analysis from trajectories, paired by question.

The archived experiment deviates from PREREG_factual_oscillation.md. Labels
0 and 1 both have lag 0, and label 6 has lag 5. No confirmatory decision is made.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.stats import wilcoxon


def analyze(data):
    cells, seen = {}, set()
    for r in data['runs']:
        key = (r['n_faulty'], r['temp'], r['delta'])
        ident = (*key, r['q'], r['seed'])
        if ident in seen:
            raise ValueError(f'duplicate run: {ident}')
        seen.add(ident)
        vals = []
        for field in ('mean_traj', 'mean_s_traj'):
            x = np.asarray(r[field], float)
            if len(x) != data['meta']['T'] or not np.isfinite(x).all():
                raise ValueError(f'invalid trajectory: {ident}')
            vals.append(float(x[len(x)//2:].std()))
        cells.setdefault(key, {}).setdefault(r['q'], []).append(vals)
    qcells = {k: {q: np.mean(v, axis=0) for q, v in qs.items()} for k, qs in cells.items()}
    result = {'analysis': 'exploratory; unadjusted one-sided Wilcoxon, greater',
              'delay_convention': data['meta'].get('delay_convention', 'legacy'),
              'cells': {}, 'contrasts': {}}
    for key, qs in sorted(qcells.items()):
        x = np.array(list(qs.values()))
        result['cells'][str(key)] = {'n_questions': len(x), 'mean': x.mean(0).tolist(),
                                    'median': np.median(x, axis=0).tolist(), 'max': x.max(0).tolist()}
    for nf, temp in sorted({(k[0], k[1]) for k in cells}):
        if (nf,temp,0) not in qcells or (nf,temp,6) not in qcells:
            raise ValueError('both delay labels 0 and 6 are required')
        c0, c6 = qcells[nf,temp,0], qcells[nf,temp,6]
        if c0.keys() != c6.keys():
            raise ValueError('unmatched questions in the delay contrast')
        x = np.array([c6[q][0]-c0[q][0] for q in sorted(c0)])
        p = float(wilcoxon(x, alternative='greater').pvalue) if np.any(x) else 1.
        result['contrasts'][str((nf,temp))] = {'n_questions': len(x), 'mean_difference': float(x.mean()),
                                             'positive': int(sum(x>0)), 'p_unadjusted': p}
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*', type=Path)
    a = ap.parse_args()
    files = a.files or [Path(__file__).with_name(n) for n in ('f1_psilo.json', 'f1_tqa.json')]
    print(json.dumps({p.name: analyze(json.loads(p.read_text())) for p in files}, indent=2))
