"""Sequential CPU suite. Refuses any overlapping training process."""
from pathlib import Path
import subprocess,sys,os,json,time
HERE=Path(__file__).resolve().parent

def overlap():
    result=subprocess.run(['ps','-Ao','pid=,ppid=,stat=,command='],text=True,capture_output=True,check=True)
    found=[]
    for line in result.stdout.splitlines():
        parts=line.strip().split(None,3)
        if len(parts)!=4:continue
        pid,ppid,state,command=parts
        if int(pid) in [os.getpid(),os.getppid()] or command.startswith('/bin/zsh -c'):continue
        if ('train_segmentation.py' in command or
            ('python' in command.lower() and ('/train.py' in command or 'run_suite.py' in command))):found.append(line)
    if found:raise RuntimeError('Overlapping training process: '+'\n'.join(found))
    return dict(utc=time.time(),overlap=False)

def run(file,*args):
    log=HERE/'logs'/('suite_'+file.replace('.py','')+'_'+'_'.join(args).replace('--','')+'.log')
    with log.open('x') as stream:
        subprocess.run([sys.executable,'-B',str(HERE/file),*args],stdout=stream,
                       stderr=subprocess.STDOUT,check=True,cwd=HERE.parents[1])

if __name__=='__main__':
    for seed in [0,1,2]:
        record=overlap();(HERE/f'logs/seed{seed}_overlap.json').write_text(json.dumps(record)+'\n')
        run('train.py','--seed',str(seed))
        for stage in ['fit','name','predict']:run('stages.py',stage,'--seed',str(seed))
    run('seal_predictions.py')
    run('evaluate.py')
    run('report.py')
