"""Fresh audit of every archived experiment; never modify source JSON."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import sys
import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'paper/remote'))
from f1_analyze import analyze


def main():
    hashes = json.loads((HERE/'raw_hashes.json').read_text())
    for name, sha in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha, name
    out = {'integrity': {}, 'f1': {}, 'rq2': {}, 'v3': {}}
    for path in sorted((ROOT/'paper/remote').glob('*.json')):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or 'runs' not in data:
            continue
        runs, meta = data['runs'], data['meta']
        seen = set()
        for r in runs:
            ident = tuple((k, r[k]) for k in ('q','alpha','kappa','delta','n_faulty','temp','seed') if k in r)
            assert ident not in seen, (path.name, ident)
            seen.add(ident)
            x = np.array(r.get('mean_traj', r.get('e')), float)
            assert len(x) == meta['T'] and np.isfinite(x).all(), path.name
            if 'mean_traj' in r:
                assert np.all((0 <= x) & (x <= 1))
            if 'conv' in r:
                assert r['conv'] == int(x[-1] < .1)
                assert abs(r['final']-x[-1]) < 1e-12
            if 'amp_s' in r:
                assert abs(r['amp']-np.std(x[len(x)//2:])) < 1e-12
                sx = np.array(r['mean_s_traj'])
                assert np.all(np.abs(sx)<=1) and len(sx)==len(x)
                assert abs(r['amp_s']-np.std(sx[len(sx)//2:])) < 1e-12
            if 'traj' in r:
                np.testing.assert_allclose(np.mean(r['traj'], axis=1), x)
        nq = len({r['q'] for r in runs})
        assert nq == meta['n_items']
        cells = defaultdict(list)
        for r in runs:
            key = tuple((k, r[k]) for k in ('alpha','kappa','delta','n_faulty','temp') if k in r)
            cells[key].append(r)
        assert all(len(c)==nq*meta.get('seeds',1) for c in cells.values())
        out['integrity'][path.name] = {'runs':len(runs), 'questions':nq, 'states':meta['T'],
                                      'cells':len(cells), 'replicates':meta.get('seeds',1)}
        if path.name.startswith('f1_'):
            out['f1'][path.name] = analyze(data)
        if 'kappa' in runs[0]:
            result = {}
            for key, cell in cells.items():
                row = {'n':len(cell), 'endpoint_success':sum(r['mean_traj'][-1]<.1 for r in cell)/len(cell)}
                if 'majs' in cell[0]:
                    flips = []
                    for r in cell:
                        m = [x.lower().strip() for x in r['majs'][2:]]
                        f = sum(a!=b for a,b in zip(m,m[1:]))/(len(m)-1)
                        assert abs(f-r['flip_rate']) <= .0005+1e-12
                        flips.append(f)
                    row['string_flip_rate'] = float(np.mean(flips))
                result[str(key)] = row
            out['rq2'][path.name] = result
        if path.name.startswith('expA_v3'):
            errors, ambiguous, inconsistent = [], [], []
            for r in runs:
                x = np.array(r['e'][3:]); amp = float(x.std()); span=float(np.ptp(x))
                errors.append(abs(amp-r['amp']))
                # std is 1-Lipschitz in RMS; e rounding <=.0005 and amp rounding <=.0005.
                assert errors[-1] <= .001+1e-10, (path.name, errors[-1])
                assert abs(span-r['rng']) <= .0015+1e-10
                if abs(span-.15)<=.001+1e-10:
                    ambiguous.append([r['q'],r['alpha'],r['delta'],r['seed']])
                elif int(span>.15)!=r['moved']:
                    inconsistent.append(r)
            assert not inconsistent
            contrasts = {}
            for metric in ('stored','trajectory'):
                for moved in (False,True):
                    pairs=[]
                    for q in sorted({r['q'] for r in runs}):
                        cs=[[r for r in runs if r['q']==q and r['alpha']==.5 and r['delta']==d
                             and (not moved or r['moved'])] for d in (1,6)]
                        if all(cs):
                            pairs.append([np.mean([r['amp'] if metric=='stored' else np.std(r['e'][3:]) for r in c]) for c in cs])
                    if not pairs:
                        contrasts[f'{metric}_{"moved" if moved else "all"}'] = {'n_questions':0,'means':None,'p':None}
                        continue
                    x=np.array(pairs); diff=x[:,1]-x[:,0]
                    contrasts[f'{metric}_{"moved" if moved else "all"}'] = {
                        'n_questions':len(x), 'means':x.mean(0).tolist(),
                        'p':float(wilcoxon(diff,alternative='greater').pvalue) if np.any(diff) else 1.}
            out['v3'][path.name]={'max_amp_discrepancy':max(errors),
                                  'movement_ambiguous_after_rounding':ambiguous, 'contrasts':contrasts}
    (HERE/'experiment_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(f"Integrity verified: {len(hashes)} unchanged JSON files; {len(out['integrity'])} experiment archives.")
    print('F1 strong-forcing contrasts:', {k:v['contrasts'] for k,v in out['f1'].items()})
    print('V3 rounding:', {k:(v['max_amp_discrepancy'],len(v['movement_ambiguous_after_rounding'])) for k,v in out['v3'].items()})


if __name__ == '__main__':
    main()
