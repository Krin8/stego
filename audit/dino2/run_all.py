"""Sequential fixed protocol. Any genuine local-load failure skips that arm only."""
from pathlib import Path
import os,sys,subprocess,json
H=Path(__file__).resolve().parent;os.environ['PYTHONDONTWRITEBYTECODE']='1'
def run(script,args,name):
 cmd=[sys.executable,str(H/script),*args]
 with (H/'commands.txt').open('a') as f:f.write(' '.join(cmd)+'\n')
 print('START',name,flush=True)
 with (H/f'{name}.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',name,flush=True)
assert json.loads((H/'weight_verification.json').read_text())['passed']
for arm in ['D2','D3']:
 if not (H/f'{arm}_extract.json').exists():run('extract.py',['--arm',arm],f'{arm}_extract_run')
 if (H/f'{arm}_FAILED.json').exists():continue
 run('figures.py',['--arm',arm],f'{arm}_figures')
for arm in ['D2','D3']:
 if not (H/f'{arm}_extract.json').exists():continue
 assert (H/f'{arm}_checks.json').is_file() and (H/f'{arm}_figures.json').is_file()
 for k in [8,2]:
  for s in [0,1,2]:run('fit_score.py',['--arm',arm,'--k',str(k),'--seed',str(s)],f'{arm}_k{k}_s{s}')
if any((H/f'{arm}_extract.json').exists() for arm in ['D2','D3']):
 run('pair_affinities.py',[],'pair_affinities');run('pair_score.py',[],'pair_score')
print('ALL COMPLETE',flush=True)
