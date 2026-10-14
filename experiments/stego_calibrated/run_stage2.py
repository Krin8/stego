"""Sequential bounded stage 2; never retries or resumes an interrupted training."""
import os,sys,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
decision=json.loads((HERE/'screen_decision.json').read_text())
assert decision['D448']['passed'] and not decision['C448']['passed']
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'',
     'STEGO_RUN_DIR':str(HERE/'D448/stage2')}
logs=HERE/'D448/stage2/logs'
def run(script,args,name,local_env=env):
    command=[sys.executable,'-B',str(HERE/script),*args]
    print('START',name,' '.join(command),flush=True)
    with (logs/(name+'.log')).open('x') as stream:
        subprocess.run(command,cwd=ROOT,env=local_env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('COMPLETE',name,flush=True)
for seed in [0,1,2]:
    run('train.py',['--seed',str(seed)],f'seed{seed}_stdout')
for seed in [0,1,2]:
    for stage in ['fit','name','predict']:
        run('stages.py',[stage,'--seed',str(seed)],f'seed{seed}_{stage}_stdout')
run('seal_all.py',[],'seal_all',local_env={**env,'STEGO_RUN_DIR':str(HERE)})
run('evaluate.py',[],'evaluate')
print('BOUNDED ROUND COMPLETE',flush=True)
