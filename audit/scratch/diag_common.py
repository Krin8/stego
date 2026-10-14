"""Shared bounded, offline, CPU-only setup for the September 30 diagnostic."""
from pathlib import Path
import os,sys,signal,tempfile,json,time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'audit/scratch/diag'
SEED=20260930
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',MPLCONFIGDIR=str(OUT/'mpl'),TOKENIZERS_PARALLELISM='false')
sys.dont_write_bytecode=True
tempfile.tempdir=str(OUT)
signal.alarm(570)
START=time.monotonic()
def guard(event,args):
    if event=='open':
        path,mode,flags=args
        writing=isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
        if writing and not isinstance(path,int):
            p=Path(path).resolve()
            if not p.is_relative_to(ROOT/'audit/scratch') and p!=ROOT/'audit/DIAG.md':raise PermissionError('Diagnostic write denied: '+str(p))
    elif event=='os.mkdir':
        p=Path(args[0]).resolve()
        if not p.exists() and not p.is_relative_to(ROOT/'audit/scratch'):raise PermissionError('Diagnostic mkdir denied')
    elif event in ('os.remove','os.rmdir','os.rename'):
        for p in args[:2] if event=='os.rename' else args[:1]:
            if not Path(p).resolve().is_relative_to(ROOT/'audit/scratch'):raise PermissionError('Diagnostic mutation denied')
    elif event in ('socket.connect','os.chmod','os.utime','os.link','os.symlink'):raise PermissionError(event)
sys.addaudithook(guard)
sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def cfg():
    from omegaconf import OmegaConf
    return OmegaConf.load(ROOT/'logs/chaos/chaos_chaos_v2_ct_date_Sep30_15-52-07/lightning_logs/version_0/hparams.yaml').cfg
def dataset(split):
    from data import CHAOS
    from chaos_preprocessing import get_chaos_transform
    return CHAOS(str(ROOT/'src/pytorch_data_dir'),'CT',split,get_chaos_transform(),get_chaos_transform(is_label=True),2,True)
def elapsed():return round(time.monotonic()-START,2)
