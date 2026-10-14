"""Isolated offline CPU runtime. Masks forbidden until a completed prediction seal exists."""
from pathlib import Path
import os,sys,json,hashlib,time,ast,tempfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];BASE=ROOT/'audit/naming';OLD=ROOT/'audit/improve'
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'));tempfile.tempdir=str(HERE/'logs')
os.environ.setdefault('HF_HOME',str(Path.home()/'.cache/huggingface'))
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(HERE/'logs/mpl'),XDG_CACHE_HOME=str(HERE/'logs/cache'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
STAGE=os.environ.get('STEGO_STAGE','development');READS=set();FORBIDDEN=[]
def guard(event,args):
 if event=='open' and not isinstance(args[0],int):
  p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:];write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
  if write and not p.is_relative_to(HERE):raise PermissionError('Experiment writes only: '+str(p))
  if not write:
   s=str(p);bad=('/Ground/' in s or '/labels/' in s or '/masks/' in s or 'scoring_masks' in s or p.name.endswith(('_scores.json','_results.json','_labels.npy')) or (p.parent==ROOT/'audit/final_pipeline' and p.name.startswith(('all20_seed','inductive_seed')) and p.suffix=='.json' and not p.name.endswith('_fit.json')))
   if bad and STAGE!='evaluation':FORBIDDEN.append(s);raise PermissionError('No labels/reference scores before evaluation: '+s)
   if STAGE in ['training','naming','fitting','development'] and '/preprocessed_v2/val/' in s:raise PermissionError('Validation images excluded from '+STAGE+': '+s)
   READS.add(s)
 elif event=='socket.connect':raise PermissionError('Network forbidden')
 elif event in ['os.remove','os.rmdir','os.mkdir','os.rename']:
  for v in args[:2] if event=='os.rename' else args[:1]:
   p=Path(os.fsdecode(v)).resolve()
   if not p.is_relative_to(HERE) and not(event=='os.mkdir' and p.exists()):raise PermissionError('Mutation outside experiment: '+str(p))
sys.addaudithook(guard)
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from torchvision.transforms import functional as TF,InterpolationMode
from chaos_preprocessing import MEAN,STD
from types import SimpleNamespace
torch.set_num_threads(4)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 p=HERE/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(Path(p).read_text())
def reuse(path,names):
 tree=ast.parse(Path(path).read_text())
 for name in names:
  node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),globals())
reuse(ROOT/'audit/final_pipeline/segment_ct.py',['body_box','crop_image','image_tensor','inverse_grid','resize224','largest','guided','distances','contract'])
# Original distance formula, generalized only from its hard-coded768 to centre dimension.
_distance_node=next(n for n in ast.parse((ROOT/'audit/final_pipeline/segment_ct.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='distances')
_distance_source=ast.unparse(_distance_node);assert _distance_source.count('reshape(-1, 768)')==1
exec(compile(_distance_source.replace('reshape(-1, 768)','reshape(-1, centers.shape[1])'),'baseline distances: feature-dimension-only generalization','exec'),globals())
def records():
 rr=read(BASE/'image_records.json');train={r['patient'] for r in rr if r['split']=='train'};val={r['patient'] for r in rr if r['split']=='val'};assert not train&val and len(train)==16 and len(val)==4;return rr
def load_image(r):
 with Image.open(r['image']) as im:a=np.array(im,np.float32)/65535.
 assert a.shape==(512,512);return a
def seed_all(seed):
 import random
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
def frozen_backbone():
 from modules import BiomedCLIPFeaturizer
 net=BiomedCLIPFeaturizer(70,SimpleNamespace(projection_type=None,dropout=False,use_text_prompts=False)).cpu().eval().requires_grad_(False)
 assert sum(p.numel() for p in net.parameters() if p.requires_grad)==0
 return net
def tokens(net,x):
 with torch.no_grad():f,_=net(x)
 assert f.shape[1:]==(768,28,28);return F.normalize(f,dim=1)
def baseline_seal():
 p=BASE/'stageA_manifest.json';m=read(p);assert sha(p)==(BASE/'stageA_manifest.sha256').read_text().split()[0]
 for rel,digest in m['output_sha256'].items():assert sha(BASE/rel)==digest,rel
 return sha(p)
def ledger(name):dump(name,dict(stage=STAGE,opened_read_paths=sorted(READS),blocked=FORBIDDEN,labels_allowed=STAGE=='evaluation'))
