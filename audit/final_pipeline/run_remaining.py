"""Sequential CPU execution; completed steps are retained, never silently overwritten."""
from pathlib import Path
import subprocess,sys,os
H=Path(__file__).resolve().parent
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
for seed in [0,1,2]:
    for task,scope,done in [('fit','inductive',f'inductive_seed{seed}_fit.json'),('mapping','inductive',f'inductive_seed{seed}.json'),('evaluate','inductive',f'inductive_seed{seed}_results.json'),('mapping','all20',f'all20_seed{seed}.json')]:
        if (H/done).exists():continue
        cmd=[sys.executable,str(H/'study.py'),task,'--seed',str(seed),'--scope',scope]
        with (H/'commands.txt').open('a') as f:f.write(' '.join(cmd)+'\n')
        print('START',task,scope,seed,flush=True)
        with (H/f'{task}_{scope}_{seed}.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('DONE',task,scope,seed,flush=True)
with (H/'package_check.log').open('x') as f:subprocess.run([sys.executable,str(H/'package_check.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
print('ALL COMPLETE',flush=True)
