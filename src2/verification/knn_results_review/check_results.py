"""Verify saved artifacts and score saved masks only; no model fitting or training."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from src2.utils import sha256,read_json,resolve_path
from src2.metrics import counts,scores,aggregate
from src2.preprocessing import resize224,load_image,inverse_grid

torch.set_num_threads(2)
root=Path.cwd(); base=root/'src2/runs'; out=Path(__file__).resolve().parent
training=base/'knn_seed0_training';ev=base/'knn_seed0_evaluation';fit=base/'knn_seed0_fit'
complete=read_json(training/'complete.json');manifest=read_json(ev/'prediction_manifest.json');cfg=read_json(ev/'resolved_config.json');metrics=read_json(ev/'metrics.json')
assert sha256(training/'final.pt')==complete['checkpoint_sha256']==manifest['checkpoint_sha256']
assert sha256(fit/'fit.json')==manifest['fit_sha256']
assert sha256(ev/'prediction_manifest.json')==metrics['manifest_sha256']
for name,h in manifest['outputs'].items():assert sha256(ev/name)==h
rows=[json.loads(s) for s in (training/'train.jsonl').read_text().splitlines()]
assert [r['batch'] for r in rows]==list(range(1,501))
assert all(np.isfinite(r['loss']) for r in rows)
labels=np.load(resolve_path(cfg,cfg['paths']['scoring_masks']),mmap_mode='r')
raw=np.load(ev/'raw_clusters.npy',mmap_mode='r');predictions={k:np.load(ev/f'N2_{k}.npy',mmap_mode='r') for k in ('hard','refined')}
report=dict(artifact_hashes_verified=True,batches=len(rows),validation_slices=len(manifest['indices']),
    monitor_checks=len(complete['monitors']),all_monitor_checks_healthy=all(r['healthy'] for r in complete['monitors']),
    selected_cluster=manifest['selected_cluster'],labels_use='validation labels for score verification and GT display only',
    empty_slice_diagnostics={},metric_values_verified=True,checkpoint=str(training/'final.pt'),checkpoint_bytes=(training/'final.pt').stat().st_size)
for kind,preds in predictions.items():
 for scale in (512,224):
  result=[]
  for j,i in enumerate(manifest['indices']):
   p=preds[j];y=labels[i]==1
   if scale==224:p,y=resize224(p),resize224(y)
   result.append(dict(index=i,sample_id=manifest['records'][j]['sample_id'],patient=manifest['records'][j]['patient'],**scores(counts(p,y))))
  assert aggregate(result)==metrics['results'][kind][str(scale)]
 native=metrics['results'][kind]['512']['slices'];empty=[r for r in native if r['tp']+r['fn']==0]
 report['empty_slice_diagnostics'][kind]=dict(slices=len(empty),with_false_positives=sum(r['fp']>0 for r in empty),mean_fp_pixels=float(np.mean([r['fp'] for r in empty])) if empty else None)

fixed=[1916,2005,2095,2242];lookup={i:j for j,i in enumerate(manifest['indices'])}
cmap=ListedColormap([*plt.get_cmap('tab10').colors[:8],(0,0,0)]);norm=BoundaryNorm(np.arange(-.5,9.5),9)
fig,axes=plt.subplots(4,5,figsize=(17,13))
for row,i in enumerate(fixed):
 j=lookup[i];r=manifest['records'][j];im=load_image(resolve_path(cfg,r['image']),cfg);y=labels[i]==1
 native=inverse_grid(raw[j],r,mode='nearest',outside=8)
 panels=[im,y,native,predictions['hard'][j],predictions['refined'][j]]
 for col,p in enumerate(panels):
  ax=axes[row,col]
  if col==2:ax.imshow(p,cmap=cmap,norm=norm,interpolation='nearest')
  else:ax.imshow(p,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
  if col in (3,4):
   s=scores(counts(p,y));ax.set_title(f'IoU {100*s["liver_iou"]:.2f}% / Dice {100*s["liver_dice"]:.2f}%')
   if y.any():ax.contour(y,levels=[.5],colors=['lime'],linewidths=.6)
  else:ax.set_title(['CT','Ground truth','Raw K=8'][col])
  ax.set_axis_off()
 axes[row,0].set_title(f'Patient {r["patient"]} | index {i}')
fig.suptitle('Feature-kNN seed 0 — actual native masks, N2 selects C2\nColumns: CT | ground truth | raw clusters | hard C2 mask | guided refinement\nWhite = predicted liver; green outline = ground truth. Fixed slices, not chosen by score.',fontsize=12)
fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/'fixed_slices.png',dpi=140);plt.close(fig)
report['figure']=str(out/'fixed_slices.png')
(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
