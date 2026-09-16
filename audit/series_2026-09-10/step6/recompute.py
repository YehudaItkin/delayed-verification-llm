"""Recompute both historical estimands; never rewrite the source trajectories."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
out, hashes = {}, {}
for f in sorted((ROOT / 'paper/remote').glob('expA_v3*.json')):
    if 'smoke' in f.name:
        continue
    data = json.loads(f.read_text())
    runs = data['runs']
    hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    result = {}
    for name, moved in [('all_runs', False), ('moved_only', True)]:
        pairs = []
        for q in sorted({r['q'] for r in runs}):
            cells = [[r['amp'] for r in runs if r['q'] == q and r['alpha'] == .5
                      and r['delta'] == d and (not moved or r['moved'])] for d in (1, 6)]
            if all(cells):
                pairs.append([np.mean(c) for c in cells])
        x = np.asarray(pairs)
        result[name] = {'pairs': len(x), 'means': x.mean(0).tolist() if len(x) else None,
                        'p': float(wilcoxon(x[:, 1], x[:, 0], alternative='greater').pvalue) if len(x) else None}
    result['cells'] = {}
    for a, d in sorted({(r['alpha'], r['delta']) for r in runs}):
        cell = [r for r in runs if (r['alpha'], r['delta']) == (a, d)]
        result['cells'][str((a, d))] = {'amp': float(np.mean([r['amp'] for r in cell])),
            'moved': sum(r['moved'] for r in cell), 'n': len(cell),
            'overshoot': float(np.mean([min(r['e']) < -.02 and max(r['e']) > .02 for r in cell]))}
    result['meta'] = data['meta']
    out[f.name] = result
(HERE / 'empirical_results.json').write_text(json.dumps(out, indent=2) + '\n')
(HERE / 'source_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
print(json.dumps({k: {s: v[s] for s in ('all_runs', 'moved_only')} for k, v in out.items()}, indent=2))
