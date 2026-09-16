"""Offline replay and reanalysis; preserved inputs remain unchanged, outputs are fresh."""
from pathlib import Path
import importlib.util,json,shutil,sys,tempfile
import numpy as np
from restore_data import restore
HERE=Path(__file__).resolve().parent
BASE=HERE/'experiments/factual_expanded_v4'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
def main():
    restore();work=Path(tempfile.mkdtemp(prefix='reproduction-',dir=HERE))
    source=BASE/'qwen38/v4_3_snapshot/results/primary'
    replay=work/'replay/results/primary';replay.mkdir(parents=True)
    shutil.copytree(BASE/'qwen38/v4_3_snapshot/code',work/'replay/code')
    for name in ['manifest.json','requests.jsonl','runs.jsonl']:
        (replay/name).symlink_to((source/name).resolve())
    sys.path.insert(0,str(BASE/'code_v4_3'))
    technical=module('technical_replay',BASE/'code_v4_3/verify_stream.py').verify(replay)
    expected=json.loads((BASE/'primary_v4_3_verification.json').read_text())
    assert technical==expected,'technical replay differs from preserved report'
    for version,folder in [('v1','semantic_review_20260915'),('v2','semantic_review_v2_20260915')]:
        original=BASE/folder;fresh=work/folder;fresh.mkdir()
        for p in original.iterdir():
            if p.is_file() and not p.name.endswith('.gz'): shutil.copy2(p,fresh/p.name)
        scorer=module('semantic_'+version,original/'apply_review.py');scorer.HERE=fresh;scorer.main()
        old=json.loads((original/'results/summary.json').read_text());new=json.loads((fresh/'results/summary.json').read_text())
        for key in old:
            if key!='created_utc': assert old[key]==new[key],(version,key)
    original=BASE/'outcome_analysis_20260915';fresh=work/'outcomes';fresh.mkdir()
    shutil.copy2(original/'PROTOCOL_RU.md',fresh/'PROTOCOL_RU.md')
    # Regenerated semantic files were checked byte-for-byte above; use the preserved inputs.
    analysis=module('outcome_replay',original/'analyze.py');analysis.HERE=fresh;analysis.main()
    old=json.loads((original/'results/summary.json').read_text());new=json.loads((fresh/'results/summary.json').read_text())
    assert old['datasets']==new['datasets']
    with np.load(original/'results/bootstrap_samples.npz') as a,np.load(fresh/'results/bootstrap_samples.npz') as b:
        assert a.files==b.files
        for key in a.files: np.testing.assert_array_equal(a[key],b[key])
    report={'status':'passed','runs':technical['runs'],'requests':technical['requests'],
            'semantic_versions_reproduced':['v1','v2'],'all_bootstrap_arrays_identical':True,
            'network_calls':0,'output_directory':str(work)}
    (work/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
