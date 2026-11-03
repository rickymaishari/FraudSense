"""Bind saved artifacts, inference schema/code and configuration with checksums."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib

def main():
    bundle=joblib.load(ROOT/'artifacts/champion.joblib')
    paths=list((ROOT/'artifacts').glob('*.joblib'))
    paths+=list((ROOT/'fraudsense').glob('*.py'))+list((ROOT/'backend').glob('*.py'))
    paths += [ROOT/'requirements.txt',ROOT/'requirements-lock.txt',ROOT/'reports/new_experiments/frozen_development_decisions.json']
    manifest={'model_version':bundle['version'],'schema_version':'2.0','files':{
        p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (ROOT/'artifacts/manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Manifest created for',len(paths),'model, source and configuration files')

if __name__=='__main__':main()
