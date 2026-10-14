from pathlib import Path
import os,sys,time,json,ast,hashlib,tempfile
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];OLD=ROOT/'audit/improve';FINAL=ROOT/'audit/final_pipeline';START=time.monotonic()
sys.dont_write_bytecode=True;tempfile.tempdir=str(OUT)
os.environ.setdefault('HF_HOME',str(Path.home()/'.cache/huggingface'))
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(OUT/'mpl'),XDG_CACHE_HOME=str(OUT/'.cache'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
READS=set();BLOCKED=[]
def guard(event,args):
 if event=='open':
  path,mode,flags=args
  if isinstance(path,int):return
  p=Path(os.fsdecode(path)).resolve();writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
  if writing and not p.is_relative_to(OUT):raise PermissionError('Only audit/naming writes: '+str(p))
  if not writing and os.environ.get('NAMING_STAGE')=='A':
   s=str(p);bad=('/labels/' in s or '/Ground/' in s or 'scoring_masks' in s or p.name.endswith('_labels.npy') or p.name.endswith('_scores.json') or p.name.endswith('_results.json') or p.name in ['check_reference.json','summary.json','FINAL_BASELINE.md','IMPROVE2.md'])
   if p.parent==FINAL and p.suffix=='.json' and p.name.startswith(('all20_seed','inductive_seed')) and not p.name.endswith('_fit.json'):bad=True
   if bad:BLOCKED.append(str(p));raise PermissionError('Stage A forbids label/score/mapping files: '+s)
   READS.add(str(p))
 elif event=='os.mkdir':
  p=Path(os.fsdecode(args[0])).resolve()
  if not p.exists() and not p.is_relative_to(OUT):raise PermissionError(str(p))
 elif event in ['os.remove','os.rmdir','os.rename']:
  for path in args[:2] if event=='os.rename' else args[:1]:
   if not Path(os.fsdecode(path)).resolve().is_relative_to(OUT):raise PermissionError(str(path))
 elif event=='socket.connect':raise PermissionError('Network forbidden')
sys.addaudithook(guard);sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from torchvision.transforms import functional as TF,InterpolationMode
from chaos_preprocessing import MEAN,STD
from types import SimpleNamespace
torch.set_num_threads(4);torch.manual_seed(20260930)
def reuse(path,names):
 source=path.read_text();tree=ast.parse(source)
 for name in names:
  node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),globals())
reuse(FINAL/'segment_ct.py',['crop_image','image_tensor','inverse_grid','resize224','largest','guided','distances','predict_features','contract'])
reuse(OLD/'pipeline.py',['patient_stats'])
reuse(ROOT/'audit/scratch/forensics/common.py',['scores'])
node=next(n for n in ast.walk(ast.parse((ROOT/'audit/scratch/forensics/task1.py').read_text())) if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='conf')
expr=compile(ast.unparse(node.value).replace('minlength=16','minlength=2*K').replace('reshape(8, 2)','reshape(K, 2)'),'FORENSICS confusion','eval')
def confusion(pr,y,K):return eval(expr,dict(np=np,pr=np.asarray(pr,np.int64),y=np.asarray(y,np.int64),K=K))
def dump(name,value):(OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def elapsed():return time.monotonic()-START
def raw_image(record):
 with Image.open(record['image']) as im:return np.array(im,np.float32)/65535.
def sealed_manifest():
 p=OUT/'stageA_manifest.json';expected=(OUT/'stageA_manifest.sha256').read_text().split()[0];assert sha(p)==expected
 m=json.loads(p.read_text())
 for rel,expected in m['output_sha256'].items():assert sha(OUT/rel)==expected,rel
 return m
