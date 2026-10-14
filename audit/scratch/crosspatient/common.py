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
   if not p.is_relative_to(OUT) and p!=ROOT/'audit/CROSSPATIENT.md':raise PermissionError('Read-only cross-patient diagnostic: '+str(p))
  if not writing and not isinstance(path,int) and '/preprocessed_v2/train/labels/' in str(path) and os.environ.get('ALLOW_TRAIN_LABEL_DIAGNOSTIC')!='1':raise PermissionError('Training labels permitted only in explicit diagnostic phase')
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
 with (ROOT/'audit/CROSSPATIENT.md').open('a') as f:f.write('\n'+text+'\n')
# Execute only the original function definitions, avoiding FORENSICS' output guard.
source=ast.parse((FORE/'common.py').read_text())
for name in ['load','dataset','scores','mapping_report']:
 node=next(x for x in source.body if isinstance(x,ast.FunctionDef) and x.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(FORE/'common.py'),'exec'),globals())
# The original exhaustive mapping report has K=8 constants. Generalize ONLY these
# two dimension constants, preserving definitions, strict ties and subset ordering.
class Dimensions(ast.NodeTransformer):
 def visit_Constant(self,node):
  if node.value==8:return ast.parse('len(conf)',mode='eval').body
  if node.value==256:return ast.parse('2**len(conf)',mode='eval').body
  return node
node=next(x for x in ast.parse((FORE/'common.py').read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='mapping_report')
node=Dimensions().visit(node); node.name='mapping_any_k'
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(FORE/'common.py'),'exec'),globals())
# Reuse Task1's actual bincount expression, changing only K-dependent dimensions.
node=next(x for x in ast.walk(ast.parse((FORE/'task1.py').read_text())) if isinstance(x,ast.AugAssign) and isinstance(x.target,ast.Name) and x.target.id=='conf')
expr=ast.unparse(node.value).replace('minlength=16','minlength=2*K').replace('reshape(8, 2)','reshape(K, 2)')
confusion_expression=compile(expr,str(FORE/'task1.py'),'eval')
def confusion(pr,y,K):return eval(confusion_expression,dict(np=np,pr=np.asarray(pr,dtype=np.int64),y=np.asarray(y,dtype=np.int64),K=K))
def evaluate(pred,K):
 labels=np.load(FORE/'val_labels.npy',mmap_mode='r'); meta=json.loads((BASE/'samples.json').read_text())
 patients=meta['val_patients']; confs={p:np.zeros((K,2),dtype=np.int64) for p in sorted(set(patients),key=int)}
 for i,p in enumerate(patients):confs[p]+=confusion(pred[i],labels[i],K)
 total=sum(confs.values()); result=mapping_any_k(total)
 if K==8:assert result==mapping_report(total)
 result['patient_confusions']={p:c.tolist() for p,c in confs.items()}; result['lopo']={}
 for p,conf in confs.items():
  fit=mapping_any_k(total-conf)['optimal'];score=scores(conf,fit['clusters'])
  result['lopo'][p]=dict(mapping_fit_patients=[q for q in confs if q!=p],mapping_fit_iou=fit['iou'],heldout=score)
 result['lopo_summary']={m:{'mean':float(np.mean([v['heldout'][m] for v in result['lopo'].values()])), 'min':float(min(v['heldout'][m] for v in result['lopo'].values()))} for m in ['iou','dice']}
 pooled={key:sum(v['heldout'][key] for v in result['lopo'].values()) for key in ['tp','fp','fn']}
 result['lopo_pooled']=dict(iou=pooled['tp']/max(1,sum(pooled.values())),dice=2*pooled['tp']/max(1,2*pooled['tp']+pooled['fp']+pooled['fn']),**pooled)
 return result
