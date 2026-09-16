"""Restore compressed research data and verify every payload file; no network calls."""
from pathlib import Path
import gzip, hashlib, json, shutil
HERE=Path(__file__).resolve().parent
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def restore():
    manifest=json.loads((HERE/'RESEARCH_MANIFEST.json').read_text());root=HERE
    for row in manifest['files']:
        source=root/row['archive_path'];target=root/row['restore_path']
        assert source.resolve().is_relative_to(root.resolve()) and target.resolve().is_relative_to(root.resolve())
        assert sha(source)==row['stored_sha256'],row['archive_path']
        if row['encoding']=='gzip' and not target.exists():
            temp=target.with_name(target.name+'.restoring')
            with gzip.open(source,'rb') as src,temp.open('wb') as dst: shutil.copyfileobj(src,dst)
            assert temp.stat().st_size==row['bytes'] and sha(temp)==row['sha256'];temp.replace(target)
        assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256'],row['restore_path']
    print(f"Verified {len(manifest['files'])} payload files; complete data restored without changing bytes.")
if __name__=='__main__': restore()
