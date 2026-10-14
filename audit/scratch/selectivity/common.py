"""Read-only CPU pair diagnostics. Utilities copied from BASELINE; only output scope and metadata paths changed."""
from pathlib import Path
import os, sys, signal, time, tempfile, json, ast, hashlib
ROOT=Path(__file__).resolve().parents[3]; OUT=Path(__file__).resolve().parent
BASE=ROOT/'audit/scratch/baseline'; PAIR=ROOT/'audit/scratch/pairs'; FORE=ROOT/'audit/scratch/forensics'; SEED=20260930; START=time.monotonic()
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='offline',MPLBACKEND='Agg',MPLCONFIGDIR=str(OUT/'mpl'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
sys.dont_write_bytecode=True; tempfile.tempdir=str(OUT); signal.alarm(570)
def guard(event,args):
 if event=='open':
  path,mode,flags=args
  writing=isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
  if writing and not isinstance(path,int):
   p=Path(os.fsdecode(path)).resolve()
   if not p.is_relative_to(OUT) and p!=ROOT/'audit/SELECTIVITY.md':raise PermissionError('Read-only selectivity diagnostic: '+str(p))
  if not writing and not isinstance(path,int) and '/preprocessed_v2/train/labels/' in str(path):raise PermissionError('Training labels forbidden in this investigation')
 elif event=='os.mkdir':
  p=Path(os.fsdecode(args[0])).resolve()
  if not p.exists() and not p.is_relative_to(OUT):raise PermissionError(str(p))
 elif event in ['os.remove','os.rmdir','os.rename']:
  for p in args[:2] if event=='os.rename' else args[:1]:
   if not Path(os.fsdecode(p)).resolve().is_relative_to(OUT):raise PermissionError(str(p))
 elif event in ['socket.connect','os.chmod','os.utime','os.link','os.symlink']:raise PermissionError(event)
sys.addaudithook(guard); sys.path.insert(0,str(ROOT/'src'))
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_segmentation import LitUnsupervisedSegmenter
from chaos_preprocessing import get_chaos_transform
from data import ContrastiveSegDataset
from collapse_monitor import CollapseMonitor
from unlabeled_training import ImagesOnlyDataset
torch.set_num_threads(4); torch.manual_seed(SEED); np.random.seed(SEED)
def dump(name,x):(OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def elapsed():return time.monotonic()-START
def note(text):
 with (ROOT/'audit/SELECTIVITY.md').open('a') as f:f.write('\n'+text+'\n')
# Execute only the original function definitions, avoiding FORENSICS' output guard.
source=ast.parse((FORE/'common.py').read_text())
for name in ['load','dataset','scores','mapping_report']:
 node=next(x for x in source.body if isinstance(x,ast.FunctionDef) and x.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(FORE/'common.py'),'exec'),globals())
