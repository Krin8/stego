from pathlib import Path
import os,sys,time,json,ast,hashlib,tempfile
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];OLD=ROOT/'audit/improve';SELECT=ROOT/'audit/scratch/selectivity';SEED=20260930;START=time.monotonic()
sys.dont_write_bytecode=True;tempfile.tempdir=str(OUT)
os.environ.setdefault('HF_HOME',str(Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'huggingface'))
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(OUT/'mpl'),XDG_CACHE_HOME=str(OUT/'.cache'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
def guard(event,args):
 if event=='open':
  path,mode,flags=args
  if isinstance(path,int):return
  p=Path(os.fsdecode(path)).resolve();writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
  if writing and (not p.is_relative_to(OUT) or p.is_relative_to(OUT/'weights')):raise PermissionError('Writes only under audit/dino2: '+str(p))
  if os.environ.get('LABEL_FREE')=='1' and ('/labels/' in str(p) or 'scoring_masks' in str(p) or p.name.endswith('_labels.npy')):raise PermissionError('Labels forbidden in extraction/fitting/affinities')
 elif event=='os.mkdir':
  p=Path(os.fsdecode(args[0])).resolve()
  if not p.exists() and (not p.is_relative_to(OUT) or p.is_relative_to(OUT/'weights')):raise PermissionError(str(p))
 elif event in ['os.remove','os.rmdir','os.rename']:
  for path in args[:2] if event=='os.rename' else args[:1]:
   if (not Path(os.fsdecode(path)).resolve().is_relative_to(OUT) or Path(os.fsdecode(path)).resolve().is_relative_to(OUT/'weights')):raise PermissionError(str(path))
 elif event=='socket.connect':raise PermissionError('Network forbidden')
sys.addaudithook(guard);sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from PIL import Image
from torchvision.transforms import functional as TF,InterpolationMode
torch.set_num_threads(4);torch.manual_seed(SEED)
MEAN=(.485,.456,.406);STD=(.229,.224,.225)
def reuse(path,names,namespace=None):
 source=path.read_text();tree=ast.parse(source)
 for name in names:
  node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),globals() if namespace is None else namespace)
reuse(OLD/'pipeline.py',['raw_image','body_box','crop_image','image_tensor','inverse_grid','resize224','patient_stats'])
reuse(ROOT/'audit/scratch/forensics/common.py',['scores','mapping_report'])
node=next(n for n in ast.walk(ast.parse((ROOT/'audit/scratch/forensics/task1.py').read_text())) if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='conf')
expr=compile(ast.unparse(node.value).replace('minlength=16','minlength=2*K').replace('reshape(8, 2)','reshape(K, 2)'),'FORENSICS confusion','eval')
def confusion(pr,y,K):return eval(expr,dict(np=np,pr=np.asarray(pr,np.int64),y=np.asarray(y,np.int64),K=K))
def dump(name,value):(OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:sorted(x) if isinstance(x,set) else str(x))+'\n')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def elapsed():return time.monotonic()-START
def records():return json.loads((OLD/'metadata.json').read_text())['records']
