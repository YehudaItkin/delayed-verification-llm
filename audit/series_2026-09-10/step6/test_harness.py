import ast
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[3]

def load_run(path):
    tree=ast.parse(path.read_text())
    module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_v3'],type_ignores=[])
    ns={'np':np,'estimate':lambda q,prev,peers,suggestion,temp:suggestion}
    exec(compile(module,str(path),'exec'),ns)
    return ns['run_v3']

class HarnessChecks(unittest.TestCase):
    def test_legacy_zero_lag(self):
        run=load_run(Path(__file__).with_name('debate_v3_before.py'))
        item={'q':'test','truth':0,'wrong':80,'scale':80}
        e=run(item,.5,1,T=5)
        np.testing.assert_allclose(e,[1,.5,.25,.125,.0625])

    def test_correct_theory_delay_one(self):
        run=load_run(ROOT/'paper/remote/debate_v3.py')
        item={'q':'test','truth':0,'wrong':80,'scale':80}
        np.testing.assert_allclose(run(item,.5,1,T=5),[1,.5,0,-.25,-.25])

    def test_legacy_mode_preserved(self):
        old=load_run(Path(__file__).with_name('debate_v3_before.py'))
        new=load_run(ROOT/'paper/remote/debate_v3.py')
        item={'q':'test','truth':0,'wrong':80,'scale':80}
        for d in (1,6):
            np.testing.assert_array_equal(old(item,.5,d),new(item,.5,d,delay_convention='legacy'))

    def test_high_gain_zero_lag_is_damped(self):
        run=load_run(ROOT/'paper/remote/debate_v3.py')
        item={'q':'test','truth':0,'wrong':80,'scale':80}
        np.testing.assert_allclose(run(item,1.5,0,T=5),[1,-.5,.25,-.125,.0625])

if __name__=='__main__': unittest.main(verbosity=2)
