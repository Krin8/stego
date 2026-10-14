"""StageB: open labels only after immutable three-run predictions are verified."""
import os
os.environ['STEGO_STAGE']='evaluation'
from runtime import *
from datetime import datetime,timezone
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
import runtime as rt
rt.reuse(ROOT/'audit/scratch/forensics/common.py',['scores']);rt.reuse(OLD/'pipeline.py',['patient_stats'])
scores=rt.scores;patient_stats=rt.patient_stats
def confusion(p,y):return np.bincount((np.asarray(p,np.int64)*2+np.asarray(y,np.int64)).ravel(),minlength=4).reshape(2,2)
manifest_path=HERE/'predictions/stageA_manifest.json';assert sha(manifest_path)==(HERE/'predictions/stageA_manifest.sha256').read_text().strip();seal=read(manifest_path)
for path,digest in seal['outputs'].items():assert sha(HERE/path)==digest,path
for record in seal['runs'].values():
 manifest=read(HERE/record['manifest']);out=(HERE/record['manifest']).parent
 for name,digest in manifest['outputs'].items():assert sha(out/name)==digest,name
assert not(HERE/'metrics/results.json').exists();dump('metrics/label_access_start.json',dict(utc=datetime.now(timezone.utc).isoformat(),prediction_seal_sha256=sha(manifest_path)))
# First annotation reads occur only below the completed seal checks.
y={512:np.load(OLD/'scoring_masks_native.npy',mmap_mode='r'),224:np.load(OLD/'scoring_masks_224.npy',mmap_mode='r')};rr=records();fixed=read(OLD/'figure_selection.json')['eight_indices'];results={};figures=[];colors=['#4477aa','#ee6677','#228833','#ccbb44','#66ccee','#aa3377','#bbbbbb','#663399','#111111'];cmap=ListedColormap(colors);norm=BoundaryNorm(np.arange(-.5,9.5,1),9)
for arm in ['STEGO']:
 for seed in [0,1,2]:
  key=f'seed{seed}';out=HERE/f'predictions/seed{seed}';m=read(out/'prediction_manifest.json');indices=m['indices'];patients=sorted({rr[i]['patient'] for i in indices},key=int);run=dict(arm=arm,seed=seed,grid=m['grid'],selected=m['selected'],rules={})
  for rule in ['N1','N2','N3']:
   run['rules'][rule]={}
   for kind in ['hard','F1']:
    masks=np.load(out/f'{rule}_{kind}.npy',mmap_mode='r');scales={}
    for resolution in [224,512]:
     conf={p:np.zeros((2,2),np.int64) for p in patients};per={p:[] for p in patients}
     for j,i in enumerate(indices):
      pred=masks[j] if resolution==512 else resize224(masks[j]);co=confusion(pred,y[resolution][i]);conf[rr[i]['patient']]+=co;v=scores(co,[1]);per[rr[i]['patient']].append(dict(index=i,**v))
     ps={p:dict(pooled=scores(conf[p],[1]),mean_slice={metric:float(np.mean([s[metric] for s in per[p]])) for metric in ['iou','dice']},slice_scores=per[p]) for p in patients};summary={agg:{metric:patient_stats([r[agg][metric] for r in ps.values()]) for metric in ['iou','dice']} for agg in ['pooled','mean_slice']};scales[str(resolution)]=dict(patients=ps,summary=summary)
    ss=[v for p in scales['512']['patients'].values() for v in p['slice_scores']];tiny=[v for v in ss if 0<v['tp']+v['fn']<2000];empty=[v for v in ss if v['tp']+v['fn']==0];scales['diagnostics']=dict(tiny_slices=len(tiny),tiny_mean_iou=float(np.mean([v['iou'] for v in tiny])) if tiny else None,empty_slices=len(empty),empty_with_fp=sum(v['fp']>0 for v in empty),empty_mean_fp=float(np.mean([v['fp'] for v in empty])) if empty else None);run['rules'][rule][kind]=scales
   # Actual saved arrays for every named rule; no reconstructed or invented masks.
   hard=np.load(out/f'{rule}_hard.npy',mmap_mode='r');final=np.load(out/f'{rule}_F1.npy',mmap_mode='r');raw=np.load(out/'raw_clusters_native.npy',mmap_mode='r');fig,axes=plt.subplots(8,5,figsize=(20,28))
   for row,i in enumerate(fixed):
    j=indices.index(i);im=load_image(rr[i]);arrays=[im,y[512][i],raw[j],hard[j],final[j]];titles=[rr[i]['stem'],'Ground truth',f'{arm} raw {m["grid"]} grid',f'{rule} liver {m["selected"][rule]}','Final F1']
    for col,(a,title) in enumerate(zip(arrays,titles)):
     axes[row,col].imshow(a,cmap=cmap if col==2 else 'gray',norm=norm if col==2 else None,**({} if col==2 else dict(vmin=0,vmax=1)),interpolation='nearest');axes[row,col].set_title(title,fontsize=9);axes[row,col].axis('off')
   fig.suptitle(f'{key} / {rule}: actual saved predictions, native geometry',y=.995);fig.tight_layout(rect=(0,.025,1,.99));fig.legend(handles=[Patch(facecolor=colors[k],label=f'C{k}') for k in range(8)],loc='lower center',ncol=8,frameon=False);path=HERE/f'visualizations/{key}_{rule}_five_columns.png';fig.savefig(path,dpi=120);plt.close(fig);figures.append(str(path))
   fig,ax=plt.subplots(8,3,figsize=(12,28))
   for row,i in enumerate(fixed):
    j=indices.index(i);im=load_image(rr[i]);ax[row,0].imshow(im,cmap='gray',vmin=0,vmax=1);ax[row,1].imshow(y[512][i],cmap='gray',vmin=0,vmax=1);ax[row,2].imshow(im,cmap='gray',vmin=0,vmax=1);rgba=np.zeros((512,512,4));rgba[final[j]>0]=[.1,.9,.35,.4];ax[row,2].imshow(rgba)
    for a,title in zip(ax[row],[rr[i]['stem'],'Ground truth',f'{key} {rule} final']):a.set_title(title,fontsize=9);a.axis('off')
   fig.tight_layout();path=HERE/f'visualizations/{key}_{rule}_overlay.png';fig.savefig(path,dpi=120);plt.close(fig);figures.append(str(path))
  results[key]=run
# Seed masks compared directly, label-free; all rules, both output stages.
stability=[]
for arm in ['STEGO']:
 for rule in ['N1','N2','N3']:
  for kind in ['hard','F1']:
   for a,b in [(0,1),(0,2),(1,2)]:
    pa=np.load(HERE/f'predictions/seed{a}/{rule}_{kind}.npy',mmap_mode='r');pb=np.load(HERE/f'predictions/seed{b}/{rule}_{kind}.npy',mmap_mode='r');inter=tot=diff=0
    for x,z in zip(pa,pb):inter+=int(np.count_nonzero((x>0)&(z>0)));tot+=int(x.sum()+z.sum());diff+=int(np.count_nonzero(x!=z))
    stability.append(dict(arm=arm,rule=rule,kind=kind,seeds=[a,b],interseed_dice=2*inter/tot if tot else 1.,different_pixels=diff))
# Quantify permutation-invariant cluster stability and 70-D representation similarity.
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
representations=[]
for a,b in [(0,1),(0,2),(1,2)]:
 xa=np.load(HERE/f'predictions/seed{a}/raw_clusters.npy',mmap_mode='r').reshape(-1)
 xb=np.load(HERE/f'predictions/seed{b}/raw_clusters.npy',mmap_mode='r').reshape(-1)
 fa=np.load(HERE/f'predictions/seed{a}/dense_features.npy',mmap_mode='r').reshape(-1,70)[::37].astype(np.float64)
 fb=np.load(HERE/f'predictions/seed{b}/dense_features.npy',mmap_mode='r').reshape(-1,70)[::37].astype(np.float64)
 fa-=fa.mean(0);fb-=fb.mean(0)
 cka=float(np.square(fa.T@fb).sum()/max(1e-30,np.linalg.norm(fa.T@fa)*np.linalg.norm(fb.T@fb)))
 representations.append(dict(seeds=[a,b],ARI=adjusted_rand_score(xa,xb),NMI=normalized_mutual_info_score(xa,xb),linear_CKA=cka,uses_labels=False))
dump('metrics/results.json',dict(producer='evaluate.py',seal_sha256=sha(manifest_path),results=results,seed_stability=stability,representation_stability=representations,figures=figures))
# Baseline annotations/scores accessed only in this sealed evaluation process.
import csv,shutil
rows=[]
for seed in range(3):
 baseline=read(BASE/f'inductive_s{seed}_scores.json')
 for rule in ['N1','N2','N3']:
  old=baseline['rules'][rule]['results']
  new=results[f'seed{seed}']['rules'][rule]
  def value(r,kind,metric):return 100*r[kind]['224']['summary']['pooled'][metric]['mean']
  common={'Backbone':'BiomedCLIP ViT-B/16','Frozen?':True,'K':8,'Resolution':'448 input /28grid /224score','seed':seed,'rule':rule}
  for method,r,stego in [('Existing BiomedCLIP + K-means',old,False),('BiomedCLIP + STEGO + K-means',new,True),('BiomedCLIP + STEGO + K-means + F1',new,True)]:
   rows.append(dict(Method=method,**common,**{'STEGO?':stego,'IoU':value(r,'hard','iou'),'Dice':value(r,'hard','dice'),'F1 IoU':value(r,'F1','iou'),'F1 Dice':value(r,'F1','dice')}))
with (HERE/'metrics/comparison.csv').open('w',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=['Method','Backbone','Frozen?','STEGO?','K','Resolution','IoU','Dice','F1 IoU','F1 Dice','seed','rule']);writer.writeheader();writer.writerows(rows)
summary={}
for rule in ['N1','N2','N3']:
 summary[rule]={}
 for kind in ['hard','F1']:
  summary[rule][kind]={metric:dict(mean=float(np.mean(v:=[results[f'seed{s}']['rules'][rule][kind]['224']['summary']['pooled'][metric]['mean'] for s in range(3)])),std=float(np.std(v,ddof=1))) for metric in ['iou','dice']}
dump('metrics/across_seeds.json',summary)
# Primary display is predeclared seed0/N3; every other seed/rule is saved above.
source=HERE/'visualizations/seed0_N3_five_columns.png'
shutil.copyfile(source,HERE/'visualizations/actual_predictions_five_columns.png')
with Image.open(source) as im:im.convert('RGB').save(HERE/'visualizations/actual_predictions_five_columns.jpeg',quality=95)
# PDF is the same actual-array figure, not a source for prediction reconstruction.
with Image.open(source) as im:im.convert('RGB').save(HERE/'visualizations/actual_predictions_five_columns.pdf',resolution=120)
ledger('logs/evaluation_access.json');baseline_seal();print('EVALUATION COMPLETE',flush=True)
