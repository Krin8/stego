"""Two-thread offline runtime, confined output and validation-only annotation access."""
import os,sys,json,ast,time,tempfile,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];START=time.monotonic()
sys.dont_write_bytecode=True;tempfile.tempdir=str(OUT)
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg',MPLCONFIGDIR=str(OUT/'mpl'),XDG_CACHE_HOME=str(OUT/'.cache'))
READS=set();ALLOW_LABELS=False
def guard(event,args):
 if event=='open' and not isinstance(args[0],int):
  p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:]
  write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
  if write and not p.is_relative_to(OUT):raise PermissionError('Writes confined to stego_ideas: '+str(p))
  if not write:
   s=str(p);annotation=any(t in s for t in ['/labels/','/Ground/','scoring_masks','_labels.npy'])
   if annotation and not (ALLOW_LABELS and '/preprocessed_v2/val/labels/' in s):raise PermissionError('Only validation PNG labels allowed in scoring: '+s)
   READS.add(s)
 elif event in ['os.mkdir','os.remove','os.rmdir','os.rename']:
  for v in args[:2] if event=='os.rename' else args[:1]:
   p=Path(os.fsdecode(v)).resolve()
   if not p.is_relative_to(OUT) and not(event=='os.mkdir' and p.exists()):raise PermissionError(str(p))
 elif event=='socket.connect':raise PermissionError('No network')
sys.addaudithook(guard)
import numpy as np,torch
import torch.nn.functional as F
from types import SimpleNamespace
torch.set_num_threads(2);torch.set_grad_enabled(False)
sys.path.insert(0,str(ROOT/'experiments/stego_biomedclip'))
from reference_loss import ContrastiveCorrelationLoss,sample
from model import objective
def read(path):return json.loads(Path(path).read_text())
def dump(name,x):(OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def reuse(path,name,namespace):
 tree=ast.parse(Path(path).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
def ledger(name):dump(name,dict(script=sys.argv[0],threads=torch.get_num_threads(),seconds=time.monotonic()-START,opened_read_paths=sorted(READS),annotations_allowed=ALLOW_LABELS))
RR=read(ROOT/'audit/naming/image_records.json');TRAIN=np.array([i for i,r in enumerate(RR) if r['split']=='train']);VAL=np.array([i for i,r in enumerate(RR) if r['split']=='val'])
FEATURES=np.load(ROOT/'audit/improve/B2_448_features.npy',mmap_mode='r',allow_pickle=False)
CFG=read(ROOT/'experiments/stego_biomedclip/configs/locked.json')
assert len(TRAIN)==1869 and len(VAL)==472 and FEATURES.shape==(2341,784,768)
def feature_batch(global_ids):
 return torch.from_numpy(np.stack([FEATURES[i] for i in global_ids])).reshape(-1,28,28,768).permute(0,3,1,2).contiguous()
