"""Sequential predeclared screens/stage2; no retry, resume or label-based choice."""
import argparse,json,os,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['screen','stage2']);args=ap.parse_args()
def read(p):return json.loads(p.read_text())
def run(script,argv,folder,name):
    command=[sys.executable,'-B',str(P/script),*argv]
    env={**os.environ,'STEGO_RUN_DIR':str(folder),'PYTHONDONTWRITEBYTECODE':'1',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':''}
    print('START',str(folder.relative_to(P)),name,flush=True)
    with (folder/'logs'/f'{name}.log').open('x') as log:
        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('COMPLETE',str(folder.relative_to(P)),name,flush=True)
cal=read(ROOT/'audit/colocated/amended_calibration.json')
assert cal['selected_target'] is not None
selected=next(x for x in cal['summary'] if x['target']==cal['selected_target']);assert selected['admissible']
if args.stage=='screen':
    assert not(P/'screen_decision.json').exists()
    decision={}
    for arm in ['C448','P1']:
        results=[]
        for seed in [0,1,2]:
            folder=P/arm/'stage1';run('train.py',['--seed',str(seed)],folder,f'seed{seed}_stdout')
            results.append(read(folder/f'logs/seed{seed}_complete.json'))
        decision[arm]=dict(passed=all(r['screen_pass'] for r in results),
            seeds={str(r['seed']):r['screen_pass'] for r in results},labels_used=False)
    (P/'screen_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print('SCREEN DECISION',json.dumps(decision),flush=True)
else:
    decision=read(P/'screen_decision.json');arms=[arm for arm in ['C448','P1'] if decision[arm]['passed']]
    assert arms,'No arm passed screen'
    for arm in arms:
        for seed in [0,1,2]:
            run('train.py',['--seed',str(seed)],P/arm/'stage2',f'seed{seed}_stdout')
    for arm in arms:
        for seed in [0,1,2]:
            for stage in ['fit','name','predict']:
                run('stages.py',[stage,'--seed',str(seed)],P/arm/'stage2',f'seed{seed}_{stage}_stdout')
    run('seal_all.py',[],P,'seal_all')
    for arm in arms:run('evaluate.py',[],P/arm/'stage2','evaluate')
    print('ALL SELECTED RUNS SEALED AND SCORED',flush=True)
