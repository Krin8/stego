"""Bounded CPU checks: no network or writes outside this turn's scratch/report."""
import sys,os,signal,tempfile,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SCRATCH=ROOT/'audit/scratch/fixes2'
REPORT=ROOT/'audit/FIXES2.md'
sys.dont_write_bytecode=True
tempfile.tempdir='/private/tmp'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false',TORCHINDUCTOR_CACHE_DIR=str(SCRATCH),CUDA_VISIBLE_DEVICES='')
signal.alarm(110)
def audit(event,args):
    if event=='open':
        path,mode,flags=args
        write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if write and not isinstance(path,int):
            p=Path(path).resolve()
            if p!=REPORT and not p.is_relative_to(SCRATCH):raise PermissionError('Forbidden check write: '+str(p))
    elif event=='os.mkdir':
        p=Path(args[0]).resolve()
        if not p.exists() and not p.is_relative_to(SCRATCH):raise PermissionError('Forbidden mkdir')
    elif event in ('os.rename','os.remove','os.rmdir'):
        for x in args[:2] if event=='os.rename' else args[:1]:
            if not Path(x).resolve().is_relative_to(SCRATCH):raise PermissionError('Forbidden mutation')
    elif event in ('os.chmod','os.utime','os.link','os.symlink','socket.connect'):raise PermissionError(event)
sys.addaudithook(audit)
def definitions(path,names,namespace):
    tree=ast.parse((ROOT/path).read_text())
    nodes=[n for n in tree.body if getattr(n,'name',None) in names or isinstance(n,ast.Assign) and any(getattr(t,'id',None) in names for t in ast.walk(n.targets[0]))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/path),'exec'),namespace)
    return namespace
def record(text):
    with REPORT.open('a') as f:f.write('\n'+text+'\n')
