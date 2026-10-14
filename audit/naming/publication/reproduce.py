"""Re-execute sealed Stage B unchanged except output destinations; never overwrite originals."""
import os,sys,subprocess,json,hashlib
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1'
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
if len(sys.argv)>1:
 sys.path.insert(0,str(BASE))
 source=(BASE/'stage_b.py').read_text()
 # Redirect only the two result filenames, including the existing-result safety assertion.
 for suffix in ['scores.json','figure_masks.npz']:
  old="f'{key}_"+suffix+"'";new="f'publication/reproduction/{key}_"+suffix+"'"
  assert source.count(old)==(2 if suffix=='scores.json' else 1)
  source=source.replace(old,new)
 sys.argv=[str(BASE/'stage_b.py'),*sys.argv[1:]]
 exec(compile(source,str(BASE/'stage_b.py'),'exec'))
else:
 results=[]
 for fit in ['inductive','transductive']:
  for seed in [0,1,2]:
   key=f'{fit}_s{seed}';print('START',key,flush=True)
   with (HERE/'reproduction'/f'{key}.log').open('x') as log:
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--fit',fit,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,check=True,env=os.environ.copy())
   prior=json.loads((BASE/f'{key}_scores.json').read_text());now=json.loads((HERE/'reproduction'/f'{key}_scores.json').read_text())
   prior.pop('seconds');now.pop('seconds');assert prior==now,key
   results.append(dict(fit=fit,seed=seed,all_scores_rankings_agreement_exact=True));print('EXACT',key,flush=True)
 (HERE/'reproduction_verification.json').write_text(json.dumps(dict(producer='publication/reproduce.py',only_changed='output destinations; elapsed time excluded from equality',runs=results),indent=2)+'\n')
