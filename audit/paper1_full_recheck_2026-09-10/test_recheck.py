import ast
from pathlib import Path
import copy
import json
import re
import sys
import unittest
import numpy as np
from scipy.optimize import brentq

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'paper/remote'))
from history_index import history_index, effective_delay
from f1_analyze import analyze
from sysid_expA import fit_run


def functions(path, names, **extra):
    tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns={'np':np,'re':re,**extra}
    exec(compile(tree,str(path),'exec'),ns)
    return ns


class Checks(unittest.TestCase):
    def test_f1_legacy_alias_executed(self):
        # Exercise the historical harness, using distinct state labels each round.
        def run(label):
            seen=[]; counter=iter(range(1,100))
            ns=functions(HERE/'before/debate_f1.py',{'run_f1'},
                         cold_answer=lambda q:0,_dist=lambda a,b:0,signed=lambda *args:0,
                         verifier=lambda stale,ev:seen.append(stale),
                         agent_update=lambda *args:next(counter))
            ns['run_f1']({'q':'q','gold':'g','evidence':'e','wrong':'w'},1,0,10,label,0)
            return seen
        self.assertEqual(run(0),run(1))
        self.assertEqual(run(6),[0,0,0,0,0,0,1,2,3])

    def test_correct_f1_temperature_and_lag(self):
        seen=[]; temps=[]; counter=iter(range(1,100))
        ns=functions(ROOT/'paper/remote/debate_f1.py',{'run_f1'},
                     history_index=history_index,effective_delay=effective_delay,
                     cold_answer=lambda q,t:temps.append(t) or 0,
                     _dist=lambda *args:0,signed=lambda *args:0,
                     verifier=lambda stale,ev:seen.append(stale),
                     agent_update=lambda *args:next(counter))
        r=ns['run_f1']({'q':'q','gold':'g','evidence':'e','wrong':'w'},1,0,5,1,0)
        self.assertEqual(seen,[0,0,1,2]);self.assertEqual(temps,[0])
        self.assertEqual(r['answers'],[[0],[1],[2],[3],[4]])

    def test_history_bounds_and_inputs(self):
        for t in range(1,30):
            for d in range(10):
                self.assertEqual(history_index(t,d),max(0,t-1-d))
                self.assertLess(history_index(t,d,'legacy'),t)
        for d in (-1,True,1.5):
            with self.assertRaises(ValueError):history_index(3,d)

    def test_parser_counterexamples_and_fix(self):
        old=functions(HERE/'before/debate_v3.py',{'parse_num'})['parse_num']
        new=functions(ROOT/'paper/remote/debate_v3.py',{'parse_num'})['parse_num']
        self.assertEqual(old('ANSWER: 1e3'),3.)
        self.assertEqual(old('ANSWER: -.5'),5.)
        for s,v in [('ANSWER: 1e3',1000),('ANSWER: -.5',-.5),('1,234.5',1234.5),('ANSWER: +2.5E-3',.0025)]:
            self.assertEqual(new(s),v)
        for s in ('1,2','nan','inf','1e999','answer is 4 or 6','ANSWER: 3\nconfidence: .9'):
            self.assertIsNone(new(s))

    def test_f1_order_invariance_and_missing_pair(self):
        data=json.loads((ROOT/'paper/remote/f1_tqa.json').read_text())
        expected=analyze(data)['contrasts']
        data['runs'].reverse()
        self.assertEqual(analyze(data)['contrasts'],expected)
        data['runs']=[r for r in data['runs'] if not (r['delta']==6 and r['q']==data['runs'][0]['q'])]
        with self.assertRaises(ValueError):analyze(data)

    def test_lag_zero_identifiability(self):
        e=[1.]
        for _ in range(15):e.append(.7*e[-1]+.02)
        r=fit_run({'mean_traj':e,'delta':1,'q':'q','kappa':'strong'})
        self.assertAlmostEqual(r['coefficients']['combined_a_plus_b'],.7)
        self.assertNotIn('a',r['coefficients'])
        r=fit_run({'mean_traj':[.5]*18,'delta':4,'q':'q','kappa':'strong'})
        self.assertIn('unidentifiable',r['status'])

    def test_delayed_gossip_and_noncommutation(self):
        roots=np.roots([1,-1,0,20/27])
        self.assertGreater(max(abs(roots)),1)
        self.assertLess(abs(1-20/27),1)
        L=np.array([[2,-1],[-1,2]]);K=np.diag([1,2])
        np.testing.assert_array_equal(L@K-K@L,[[0,-1],[1,0]])

    def test_zero_delay_companion_and_uncapped_ceiling(self):
        ns=functions(ROOT/'paper/validate.py',{'companion','mode_stable','kappa_max'})
        np.testing.assert_allclose(ns['companion'](np.array([[.8]]),2,.1,0),[[.6]])
        self.assertTrue(ns['mode_stable'](.8,2,.1,0))
        self.assertFalse(ns['mode_stable'](.8,20,.1,0))
        self.assertAlmostEqual(ns['kappa_max'](.5,.01,1),100,places=5)
        self.assertAlmostEqual(ns['kappa_max'](.5,.01,0),150,places=5)

    def test_independent_phase_boundary_and_roots(self):
        bc=functions(ROOT/'paper/demo.py',{'chebU','beta_c'})['beta_c']
        # Independent phase equation, rather than Chebyshev implementation.
        for delay in range(1,13):
            for a in (0,.01,.2,.5,.9,.999):
                theta=brentq(lambda t:delay*t+np.arctan2(np.sin(t),np.cos(t)-a)-np.pi,
                             1e-12,np.pi-1e-12)
                beta=abs(np.exp(1j*theta)-a)
                self.assertAlmostEqual(bc(a,delay),beta,places=9)
                for factor in (.98,1.02):
                    c=np.zeros(delay+2);c[0]=1;c[1]=-a;c[-1]+=beta*factor
                    self.assertEqual(max(abs(np.roots(c)))<1,factor<1)

    def test_all_future_harnesses_compile(self):
        for name in ('debate_exp','debate_expA','debate_B','debate_f1','debate_v3'):
            compile((ROOT/f'paper/remote/{name}.py').read_text(),name,'exec')


if __name__=='__main__':unittest.main(verbosity=2)
