"""Exploratory lag-aware AR fits; not evidence of coefficient/mechanism recovery.

Legacy labels 1/4 mean lags 0/3. At lag zero, a and b are not separately
identifiable: only their sum is fitted. Constant/rank-deficient series do not
identify a recurrence and are explicitly excluded. No classification precision
or recall is interpreted as coefficient accuracy.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from history_index import effective_delay


def fit_run(r, convention='legacy'):
    lag = effective_delay(r['delta'], convention)
    e = np.asarray(r['mean_traj'], float)
    row = {'q':r['q'], 'kappa':r['kappa'], 'label':r['delta'], 'lag':lag,
           'replicate':r.get('seed',0)}
    if not np.isfinite(e).all() or len(e)<lag+5:
        return {**row, 'status':'invalid or insufficient trajectory'}
    if np.ptp(e) < 1e-9:
        return {**row, 'status':'constant trajectory; dynamics unidentifiable'}
    t = np.arange(lag,len(e)-1)
    X = np.column_stack([e[t],np.ones(len(t))] if lag==0 else [e[t],e[t-lag],np.ones(len(t))])
    y = e[t+1]
    coef,_,rank,singular=np.linalg.lstsq(X,y,rcond=None)
    if rank < X.shape[1]:
        return {**row, 'status':'rank-deficient; dynamics unidentifiable', 'rank':int(rank)}
    pred = X@coef
    denom = np.sum((y-y.mean())**2)
    if lag==0:
        coefs={'combined_a_plus_b':float(coef[0]),'intercept':float(coef[1])}
        roots=[coef[0]]
    else:
        coefs=dict(zip(('a','b','intercept'), map(float,coef)))
        poly=np.zeros(lag+2);poly[0]=1;poly[1]-=coef[0];poly[-1]-=coef[1]
        roots=np.roots(poly)
    return {**row,'status':'descriptive unconstrained OLS fit','coefficients':coefs,
            'n_rows':len(y),'rank':int(rank),'condition_number':float(singular[0]/singular[-1]),
            'in_sample_r2':float(1-np.sum((y-pred)**2)/denom) if denom>1e-16 else None,
            'fitted_homogeneous_root_radius':float(np.max(np.abs(roots)))}


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('file',nargs='?',type=Path,default=Path(__file__).with_name('expA_results.json'))
    a=ap.parse_args();data=json.loads(a.file.read_text())
    print(json.dumps({'interpretation':'descriptive fits; no independent validation or mechanism identification',
                      'runs':[fit_run(r,data['meta'].get('delay_convention','legacy')) for r in data['runs']]},indent=2))
