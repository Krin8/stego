"""CPU inference only, exact saved checkpoints, actual validation preprocessing/metrics."""
from pathlib import Path
import os,sys,json,time
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'audit/run_records/fixes3_full_C'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='offline',CUDA_VISIBLE_DEVICES='',MPLBACKEND='Agg',MPLCONFIGDIR=str(ROOT/'audit/scratch/fixes3/mpl'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
sys.path.insert(0,str(ROOT/'src'));os.chdir(ROOT/'src')
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_segmentation import LitUnsupervisedSegmenter
from data import ContrastiveSegDataset
from chaos_preprocessing import get_chaos_transform
from collapse_monitor import CollapseMonitor
from validation_visuals import fixed_validation_indices,plot_fixed_slices
import matplotlib.pyplot as plt
torch.set_num_threads(4);torch.manual_seed(0)
ev=json.loads((OUT/'postrun_evidence.json').read_text());last=ev['checkpoints'][-1]
cb=next(v for v in last['callbacks'].values() if v.get('monitor')=='test/cluster/mIoU')
paths={'best':cb['best_model_path'],'final':last['path']};results={}
for name,path in paths.items():
 start=time.monotonic();ckpt=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
 cfg=ckpt['hyper_parameters']['cfg'];model=LitUnsupervisedSegmenter(ckpt['hyper_parameters']['n_classes'],cfg).cpu()
 keys=model.load_state_dict(ckpt['state_dict'],strict=True);assert not keys.missing_keys and not keys.unexpected_keys
 model.eval();model.requires_grad_(False);model.cluster_metrics.reset()
 ds=ContrastiveSegDataset(pytorch_data_dir=cfg.pytorch_data_dir,dataset_name=cfg.dataset_name,crop_type=None,image_set='val',transform=get_chaos_transform(cfg.res,False,'center'),target_transform=get_chaos_transform(cfg.res,True,'center'),mask=True,cfg=cfg)
 assert len(ds)==472
 selected=fixed_validation_indices(ds.dataset.records);outputs=[];monitor=CollapseMonitor(8)
 with torch.inference_mode():
  for batch in DataLoader(ds,batch_size=16,shuffle=False,num_workers=0):
   _,code=model.net(batch['img']);up=F.interpolate(code,batch['label'].shape[-2:],mode='bilinear',align_corners=False)
   _,probs=model.cluster_probe(up,None);preds=probs.argmax(1)
   model.cluster_metrics.update(preds,batch['label']);monitor.update(code,preds)
   for i,index in enumerate(batch['ind'].tolist()):
    if index in selected:outputs.append(dict(ind=index,sample_id=ds.dataset.records[index]['stem'],img=batch['img'][i].cpu(),label=batch['label'][i].cpu(),cluster_preds=preds[i].cpu()))
  metrics=model.cluster_metrics.compute();lookup=model.cluster_metrics.map_clusters(torch.arange(8));stats=monitor.compute()
  confusion=model.cluster_metrics.stats.tolist()
 fig=plot_fixed_slices(outputs,lookup);overlay=OUT/f'{name}_checkpoint_overlay.png';assert not overlay.exists();fig.savefig(overlay);plt.close(fig)
 results[name]=dict(checkpoint=path,global_step=ckpt['global_step'],batches=ckpt['global_step']//3,strict_load=True,eval_mode=not model.training,images=len(ds),monitors=stats,metrics={k:float(v) for k,v in metrics.items()},mapping=lookup.tolist(),confusion=confusion,overlay=str(overlay),seconds=time.monotonic()-start)
 print(name,json.dumps(results[name]),flush=True)
 del model,ckpt
(OUT/'saved_checkpoint_evaluation.json').write_text(json.dumps({'script':str(Path(__file__).resolve()),'protocol':'Actual training validation: CPU/eval/no gradients; same472 val slices,224x224, raw clusters, bilinear code upsample align_cornersFalse, evaluation-only many-to-one mapping, no CRF. No training.','results':results},indent=2)+'\n')
