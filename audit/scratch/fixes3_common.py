"""CPU/offline/write-constrained setup for bounded fixes3 checks and experiments."""
from pathlib import Path
import os,sys,tempfile
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'audit/scratch/fixes3'
os.environ.update(MPLBACKEND='Agg',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',MPLCONFIGDIR=str(OUT/'mpl'),TOKENIZERS_PARALLELISM='false')
sys.dont_write_bytecode=True;tempfile.tempdir=str(OUT)
def guard(event,args):
    if event=='open':
        path,mode,flags=args
        write=isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
        if write and not isinstance(path,int):
            p=Path(os.fsdecode(path)).resolve()
            if not p.is_relative_to(OUT) and p!=ROOT/'audit/FIXES3.md':raise PermissionError('Write denied: '+str(p))
    elif event=='os.mkdir':
        p=Path(os.fsdecode(args[0])).resolve()
        if not p.exists() and not p.is_relative_to(OUT):raise PermissionError('mkdir denied')
    elif event in ('os.remove','os.rmdir','os.rename'):
        for p in args[:2] if event=='os.rename' else args[:1]:
            if not Path(os.fsdecode(p)).resolve().is_relative_to(OUT):raise PermissionError('Mutation denied')
    elif event in ('socket.connect','os.chmod','os.utime','os.link','os.symlink'):raise PermissionError(event)
sys.addaudithook(guard)
sys.path.insert(0,str(ROOT/'src'))
import torch,numpy as np
torch.set_num_threads(4)
def record(t):
    with (ROOT/'audit/FIXES3.md').open('a') as f:f.write('\n'+t+'\n')
