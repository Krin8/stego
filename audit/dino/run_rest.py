from pathlib import Path
import subprocess,sys,os
H=Path(__file__).resolve().parent;os.environ['PYTHONDONTWRITEBYTECODE']='1'
steps=[('fit_score.py',['--k',str(k),'--seed',str(s)],f'D1_k{k}_s{s}') for k in [8,2] for s in [0,1,2]]+[('pair_affinities.py',[],'pair_affinities'),('pair_score.py',[],'pair_score'),('figures.py',[],'figures')]
for script,args,name in steps:
 print('START',name,flush=True);cmd=[sys.executable,str(H/script),*args]
 with (H/'commands.txt').open('a') as f:f.write(' '.join(cmd)+'\n')
 with (H/f'{name}.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',name,flush=True)
