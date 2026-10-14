"""Offline CPU forensics,570s hard timeout; writes confined to new forensic artifacts."""
from pathlib import Path
import os,sys,signal,time,tempfile,json
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;SEED=20260930;START=time.monotonic()
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='offline',MPLBACKEND='Agg',MPLCONFIGDIR=str(OUT/'mpl'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
sys.dont_write_bytecode=True;tempfile.tempdir=str(OUT);signal.alarm(570)
def guard(event,args):
 if event=='open':
  path,mode,flags=args;writing=isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
  if writing and not isinstance(path,int):
   p=Path(os.fsdecode(path)).resolve()
   if not p.is_relative_to(OUT) and p!=ROOT/'audit/FORENSICS.md':raise PermissionError('Read-only audit: '+str(p))
 elif event=='os.mkdir':
  p=Path(os.fsdecode(args[0])).resolve()
  if not p.exists() and not p.is_relative_to(OUT):raise PermissionError(str(p))
 elif event in ['os.remove','os.rmdir','os.rename']:
  for p in args[:2] if event=='os.rename' else args[:1]:
   if not Path(os.fsdecode(p)).resolve().is_relative_to(OUT):raise PermissionError(str(p))
 elif event in ['socket.connect','os.chmod','os.utime','os.link','os.symlink']:raise PermissionError(event)
sys.addaudithook(guard);sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_segmentation import LitUnsupervisedSegmenter
from chaos_preprocessing import get_chaos_transform
from data import ContrastiveSegDataset
from collapse_monitor import CollapseMonitor
torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
def dump(name,x):(OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def elapsed():return time.monotonic()-START
def inventory():
 paths={f'seed{i}':json.loads((ROOT/f'audit/scratch/fixes3/fixes3_C_seed{i}_result.json').read_text())['checkpoint'] for i in range(3)}
 ev=json.loads((ROOT/'audit/run_records/fixes3_full_C/postrun_evidence.json').read_text());final=ev['checkpoints'][-1];cb=next(v for v in final['callbacks'].values() if v.get('monitor')=='test/cluster/mIoU')
 paths.update(best600=cb['best_model_path'],final=final['path'])
 return {k:dict(path=v,exists=Path(v).is_file(),bytes=Path(v).stat().st_size if Path(v).is_file() else None) for k,v in paths.items()}
def load(path,model=None):
 ck=torch.load(path,map_location='cpu',mmap=True,weights_only=False);cfg=ck['hyper_parameters']['cfg']
 if model is None:model=LitUnsupervisedSegmenter(ck['hyper_parameters']['n_classes'],cfg).cpu()
 model.load_state_dict(ck['state_dict'],strict=True);model.eval();model.requires_grad_(False);model.cluster_metrics.reset()
 return model,ck
def dataset(cfg,split='val',neighbors=False):
 return ContrastiveSegDataset(pytorch_data_dir=cfg.pytorch_data_dir,dataset_name=cfg.dataset_name,crop_type=None,image_set=split,transform=get_chaos_transform(cfg.res,False,'center'),target_transform=get_chaos_transform(cfg.res,True,'center'),cfg=cfg,mask=True,num_neighbors=cfg.num_neighbors if neighbors else None,pos_images=neighbors,pos_labels=False)
def scores(conf,ids):
 ids=[int(i) for i in ids];tp=int(conf[ids,1].sum()) if ids else 0;fp=int(conf[ids,0].sum()) if ids else 0;fn=int(conf[:,1].sum())-tp
 return dict(clusters=ids,iou=tp/max(1,tp+fp+fn),dice=2*tp/max(1,2*tp+fp+fn),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),tp=tp,fp=fp,fn=fn)
def mapping_report(conf):
 conf=np.asarray(conf,dtype=np.int64);tot=int(conf.sum());liver=int(conf[:,1].sum())
 rows=[dict(cluster=i,pixel_fraction=float(c.sum()/tot),liver_pixels=int(c[1]),background_pixels=int(c[0]),precision=float(c[1]/max(1,c.sum())),recall=float(c[1]/max(1,liver))) for i,c in enumerate(conf)]
 subsets=[scores(conf,[i for i in range(8) if mask>>i&1]) for mask in range(256)]
 single=[scores(conf,[i]) for i in range(8)]
 return dict(confusion=conf.tolist(),pixels=tot,liver_pixels=liver,clusters=rows,majority=scores(conf,np.flatnonzero(conf[:,1]>conf[:,0])),optimal=max(subsets,key=lambda s:s['iou']),thresholds={str(x):scores(conf,[i for i,r in enumerate(rows) if r['precision']>x/100]) for x in [10,20,30,40]},best_single=max(single,key=lambda s:s['iou']),active=sum(r['pixel_fraction']>=.01 for r in rows))
def note(text):
 with (ROOT/'audit/FORENSICS.md').open('a') as f:f.write('\n'+text+'\n')
