"""Read-only investigation guard; permits only this report and new audit images."""
import os,sys,tempfile,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'audit/DECISIONS.md'
SCRATCH=ROOT/'audit/scratch'
tempfile.tempdir='/private/tmp'
sys.dont_write_bytecode=True
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false',TORCHINDUCTOR_CACHE_DIR=str(SCRATCH))
def audit(event,args):
    if event=='open':
        path,mode,flags=args
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing and not isinstance(path,int):
            p=Path(path).resolve()
            if p != REPORT and not (p.is_relative_to(SCRATCH/'overlays') and p.suffix=='.png' and not p.exists()):
                raise PermissionError('Blocked write: '+str(p))
    elif event=='os.mkdir':
        p=Path(args[0]).resolve()
        if not p.is_dir() and p != SCRATCH/'overlays':raise PermissionError('Blocked directory creation')
    elif event in ('os.remove','os.rename','os.rmdir','os.chmod','os.utime','os.link','os.symlink','socket.connect'):
        raise PermissionError('Blocked '+event)
sys.addaudithook(audit)
signal.alarm(110)
def append(text):
    with REPORT.open('a') as f:f.write('\n'+text+'\n')
