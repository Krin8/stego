from pathlib import Path
import os,sys,time,tempfile,json,ast,hashlib
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;OLD=ROOT/'audit/improve';START=time.monotonic()
sys.dont_write_bytecode=True;tempfile.tempdir=str(OUT)
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(OUT/'mpl'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
def guard(event,args):
 if event=='open':
  path,mode,flags=args
  if isinstance(path,int):return
  p=Path(os.fsdecode(path)).resolve()
  writing=isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
  if writing and not p.is_relative_to(OUT):raise PermissionError('Writes restricted to improve2: '+str(p))
  if not writing and p.suffix in ['.ckpt','.pth','.safetensors']:raise PermissionError('No model loading: '+str(p))
 elif event=='os.mkdir':
  p=Path(os.fsdecode(args[0])).resolve()
  if not p.exists() and not p.is_relative_to(OUT):raise PermissionError(str(p))
 elif event in ['os.remove','os.rmdir','os.rename']:
  for path in args[:2] if event=='os.rename' else args[:1]:
   if not Path(os.fsdecode(path)).resolve().is_relative_to(OUT):raise PermissionError(str(path))
 elif event in ['socket.connect','os.chmod','os.utime','os.link','os.symlink']:raise PermissionError(event)
sys.addaudithook(guard)
import numpy as np,torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from PIL import Image
torch.set_num_threads(4)
def reuse(path,names):
 tree=ast.parse(path.read_text())
 for name in names:
  node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),globals())
reuse(OLD/'pipeline.py',['raw_image','inverse_grid','native_prediction','resize224','patient_stats'])
reuse(OLD/'postprocess.py',['largest','guided'])
reuse(ROOT/'audit/scratch/forensics/common.py',['scores'])
node=next(n for n in ast.walk(ast.parse((ROOT/'audit/scratch/forensics/task1.py').read_text())) if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='conf')
expr=ast.unparse(node.value).replace('minlength=16','minlength=2*K').replace('reshape(8, 2)','reshape(K, 2)')
confusion_expression=compile(expr,'FORENSICS confusion','eval')
reuse(OLD/'common.py',['confusion'])
def dump(name,x):(OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def elapsed():return time.monotonic()-START
def metadata():return json.loads((OLD/'metadata.json').read_text())
def surface(mask):return mask & ~ndi.binary_erosion(mask,structure=ndi.generate_binary_structure(2,1),border_value=0)
def boundary_distances(pred,truth,truth_points=None):
 pp=np.argwhere(surface(pred));tt=np.argwhere(surface(truth)) if truth_points is None else truth_points
 if not len(pp) and not len(tt):return np.empty(0), 'both_empty'
 if not len(pp) or not len(tt):return np.full(max(len(pp),len(tt)),np.inf),'one_empty'
 return np.concatenate([cKDTree(tt).query(pp)[0],cKDTree(pp).query(tt)[0]]),'both_present'
def boundary_summary(arrays):
 values=np.concatenate(arrays) if arrays else np.empty(0);finite=values[np.isfinite(values)]
 # np.percentile interpolation near inf can yield NaN: define extended-real result explicitly.
 if len(values):
  rank=.95*(len(values)-1);upper=int(np.ceil(rank))
  hd=float(np.percentile(finite,100*rank/(len(finite)-1))) if upper<len(finite) and len(finite)>1 else (float(finite[0]) if len(values)==1 and len(finite)==1 else None)
  mean=float(values.mean()) if len(finite)==len(values) else None
 else:hd=mean=None
 return dict(hd95=hd,mean_surface_distance=mean,nonfinite_boundary_points=int(len(values)-len(finite)),surface_points=int(len(values)),
  finite_only_hd95=float(np.percentile(finite,95)) if len(finite) else None,
  finite_only_mean_surface_distance=float(finite.mean()) if len(finite) else None)
