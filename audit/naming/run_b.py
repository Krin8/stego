from pathlib import Path
import subprocess,sys,os
H=Path(__file__).resolve().parent;os.environ['PYTHONDONTWRITEBYTECODE']='1'
for fit in ['inductive','transductive']:
 for seed in [0,1,2]:
  key=f'{fit}_s{seed}';cmd=[sys.executable,str(H/'stage_b.py'),'--fit',fit,'--seed',str(seed)]
  print('START',key,flush=True)
  with (H/f'{key}_stageB.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',key,flush=True)
