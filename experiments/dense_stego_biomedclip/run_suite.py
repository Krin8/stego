"""Sequential six-run controller. Stops on any overlapping training or failure."""
import os,sys,subprocess,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='')
assert (HERE/'configs/locked.json').exists(),'Training protocol is not locked; resolve pending user choices first.'
c=json.loads((HERE/'configs/locked.json').read_text());assert c['status']=='LOCKED'
def overlap():
 result=subprocess.run(['ps','-Ao','pid,stat,etime,command'],capture_output=True,text=True,check=True);matches=[]
 for line in result.stdout.splitlines()[1:]:
  p=line.strip().split(None,3)
  if len(p)!=4:continue
  pid,status,elapsed,command=p
  if int(pid)==os.getpid():continue
  if re.search(r'(?i)(?:^|/)python(?:[0-9.]*)(?:\s|$)',command) and ('train_segmentation.py' in command or 'dense_stego_biomedclip/train.py' in command):matches.append(dict(pid=int(pid),status=status,elapsed=elapsed,command=command))
 return matches
for arm in c['arms']:
 for seed in c['seeds']:
  key=f'{arm}_seed{seed}';matches=overlap();(HERE/f'logs/{key}_process_check.json').write_text(json.dumps(matches,indent=2)+'\n')
  if matches:raise RuntimeError('STOP: overlapping/suspended training found: '+json.dumps(matches))
  command=[sys.executable,str(HERE/'train.py'),'--arm',arm,'--seed',str(seed)]
  print('START',command,flush=True)
  with (HERE/f'logs/{key}.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,env=os.environ.copy())
  for stage in ['fit','name','predict']:
   with (HERE/f'logs/{key}_{stage}.log').open('x') as log:subprocess.run([sys.executable,str(HERE/'stages.py'),stage,'--arm',arm,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,check=True,env=os.environ.copy())
  print('FINISHED',key,flush=True)
for script in ['seal_predictions.py','evaluate.py','report.py']:
 with (HERE/f'logs/{script}.log').open('x') as log:subprocess.run([sys.executable,str(HERE/script)],stdout=log,stderr=subprocess.STDOUT,check=True,env=os.environ.copy())
print('ALL SIX RUNS SEALED, EVALUATED AND REPORTED',flush=True)
