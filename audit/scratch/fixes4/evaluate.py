"""Actual full-cohort inference, BASELINE/FORENSICS mappings and patient-LOPO."""
from common import *
import ast,json,argparse,signal,time
from torch.utils.data import DataLoader
import torch.nn.functional as F
from train_segmentation import LitUnsupervisedSegmenter
from chaos_preprocessing import get_chaos_transform
from data import ContrastiveSegDataset
from collapse_monitor import CollapseMonitor
from validation_visuals import fixed_validation_indices,plot_fixed_slices
import matplotlib.pyplot as plt
signal.alarm(570);start=time.monotonic();FORE=ROOT/'audit/scratch/forensics';BASE=ROOT/'audit/scratch/baseline'
# Compile original implementations exactly; no scoring-definition copy or rewrite.
original=(BASE/'common.py').read_text()
section=original[original.index('# Execute only the original function definitions'):]
evaluation_namespace=dict(globals(),OUT=BASE)
exec(compile(section,str(BASE/'common.py'),'exec'),evaluation_namespace)
load=evaluation_namespace['load'];dataset=evaluation_namespace['dataset'];evaluate=evaluation_namespace['evaluate']
WRITE=ROOT/'audit/scratch/fixes4'
p=argparse.ArgumentParser();p.add_argument('--candidate',choices=['D','E'],required=True);p.add_argument('--seed',type=int,choices=[0,1,2],required=True);args=p.parse_args()
name=f'fixes4_{args.candidate}_seed{args.seed}';runroot=WRITE/'experiments'/name
resultpath=WRITE/f'{name}_evaluation.json';assert not resultpath.exists()
terminal=json.loads((runroot/'result.json').read_text());index=terminal['checkpoints'];results=[];model=None
def no_train_labels(event,a):
 if event=='open' and not isinstance(a[0],int) and '/preprocessed_v2/train/labels/' in str(Path(os.fsdecode(a[0])).resolve()):raise PermissionError('Training annotations forbidden')
sys.addaudithook(no_train_labels)
for batch in [100,250,500]:
 if str(batch) not in index:
  results.append(dict(candidate=args.candidate,seed=args.seed,batch=batch,status='MISSING: run stopped before checkpoint',early_stopped=terminal['early_stopped']));continue
 path=Path(index[str(batch)]);model,ck=load(path,model);assert ck['global_step']==3*batch
 assert model.cfg.extra_clusters==6 and model.cfg.max_steps==5000 and model.cfg.seed==args.seed
 assert ck['lr_schedulers'][0]['T_max']==5000 and ck['lr_schedulers'][0]['last_epoch']==batch
 ds=dataset(model.cfg);assert len(ds)==472
 assert [r['sample_id'] for r in ds.dataset.records]==json.loads((BASE/'samples.json').read_text())['val_ids']
 pred=np.empty((472,224,224),dtype=np.uint8);monitor=CollapseMonitor(8);fixed=set(fixed_validation_indices(ds.dataset.records));panels=[]
 with torch.inference_mode():
  for data in DataLoader(ds,batch_size=16,shuffle=False,num_workers=0):
   _,code=model.net(data['img']);up=F.interpolate(code,data['label'].shape[-2:],mode='bilinear',align_corners=False)
   _,prob=model.cluster_probe(up,None);ids=prob.argmax(1);monitor.update(code,ids);model.cluster_metrics.update(ids,data['label'])
   indices=data['ind'].numpy();pred[indices]=ids.numpy()
   for j,i in enumerate(indices):
    if i in fixed:panels.append(dict(ind=int(i),sample_id=ds.dataset.records[i]['stem'],img=data['img'][j],label=data['label'][j],cluster_preds=ids[j]))
  actual=model.cluster_metrics.compute()
 r=evaluate(pred,8);assert abs(r['majority']['iou']*100-float(actual['test/cluster/liver_IoU']))<1e-5
 stats=monitor.compute();unhealthy=stats['cross_image_cosine']>.99 or stats['code_std']<.01
 r.update(candidate=args.candidate,seed=args.seed,batch=batch,status='EVALUATED',checkpoint=str(path),checkpoint_bytes=path.stat().st_size,monitors=stats,unhealthy=unhealthy,run_early_stopped=terminal['early_stopped'],script=__file__,labels='Validation only: majority/oracle maps fit+score all472; LOPO fits three patients and scores held-out fourth, no held-out labels fit that fold mapping.')
 if batch==500:
  lookup=torch.zeros(8,dtype=torch.long);lookup[r['majority']['clusters']]=1
  fig=plot_fixed_slices(panels,lookup);overlay=runroot/'evaluation_fixed_batch_000500.png';fig.savefig(overlay);plt.close(fig);r['evaluation_overlay']=str(overlay)
  assert abs(stats['code_std']-terminal['monitors']['code_std'])<1e-6
  assert abs(stats['cross_image_cosine']-terminal['monitors']['cross_image_cosine'])<1e-6
 results.append(r);resultpath.write_text(json.dumps(dict(script=__file__,results=results,elapsed_seconds=time.monotonic()-start),indent=2)+'\n')
 print(name,batch,'majority',r['majority']['iou'],'optimal',r['optimal']['iou'],'LOPO',r['lopo_summary'],flush=True)
resultpath.write_text(json.dumps(dict(script=__file__,results=results,elapsed_seconds=time.monotonic()-start),indent=2)+'\n')
