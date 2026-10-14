"""Validation-only scoring of previously fixed, label-free pair scores/cuts."""
from common import *
import argparse
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve
p=argparse.ArgumentParser();p.add_argument('--pool',choices=['intra','knn','random'],required=True);args=p.parse_args();pool=args.pool
meta=json.loads((OUT/'affinities.json').read_text());records=json.loads((OUT/'records.json').read_text())
labels=torch.from_numpy(np.array(np.load(FORE/'val_labels.npy',mmap_mode='r'),dtype=np.float32))[:,None]
occupancy=F.avg_pool2d(labels,16);patch_labels=(occupancy>.5)[:,0].numpy().astype(np.uint8).reshape(472,196)
a=np.load(OUT/f'{pool}_a.npy');b=np.load(OUT/f'{pool}_b.npy');ta=np.load(OUT/f'{pool}_ta.npy');tb=np.load(OUT/f'{pool}_tb.npy')
category=(2*patch_labels[a[:,None],ta]+patch_labels[b[:,None],tb]).reshape(-1)
pid=np.array([str(r['patient_id']) for r in records]);anchor_patient=np.repeat(pid[a],ta.shape[1]);del labels
def basic(cat,x,threshold):
 counts=np.bincount(cat,minlength=4);chosen=x>threshold;c=np.bincount(cat[chosen],minlength=4)
 def ratio(n,d):return float(n/d) if d else None
 return dict(threshold=float(threshold),attraction=float(chosen.mean()),liver_target_precision=ratio(c[3],c[2]+c[3]),liver_pair_recall=ratio(c[3],counts[3]),liver_background_attraction=ratio(c[2],counts[2]),same_label=ratio(c[0]+c[3],c.sum()),attracted_liver_anchor_pairs=int(c[2]+c[3]),counts_BB_BL_LB_LL=c.tolist())
def envelope(precision,recall,threshold,r,x,cat):
 available=np.flatnonzero(recall>=r)
 if not len(available):return None
 i=int(available[np.argmax(precision[available])]);t=float(threshold[i])
 return dict(target_recall=float(r),precision=float(precision[i]),actual_recall=float(recall[i]),inclusive_threshold=t,overall_attraction=float(np.mean(x>=t)))
def summarize(cat,x,cuts,reference_recall=None,detail=True):
 liver=cat>=2;y=(cat[liver]==3);scores=x[liver];n=int(len(y))
 result=dict(pairs=int(len(x)),liver_anchor_pairs=n,liver_target_pairs=int(y.sum()),liver_target_base_rate=float(y.mean()),auc=float(roc_auc_score(y,scores)),average_precision=float(average_precision_score(y,scores)),sweep=[dict(budget=c.get('budget'),**basic(cat,x,c['threshold'])) for c in cuts])
 precision,recall,threshold=precision_recall_curve(y,scores);precision=precision[:-1];recall=recall[:-1]
 targets=[.5,.75]+([] if reference_recall is None else [reference_recall]);keys=['r50','r75']+([] if reference_recall is None else ['kmeans_recall'])
 result['precision_envelopes']={key:envelope(precision,recall,threshold,r,x,cat) for key,r in zip(keys,targets)}
 if detail:
  available=np.flatnonzero(precision>.5)
  if len(available):
   i=int(available[np.argmax(recall[available])]);t=float(threshold[i]);result['largest_recall_at_majority_precision']=dict(precision=float(precision[i]),recall=float(recall[i]),inclusive_threshold=t,overall_attraction=float(np.mean(x>=t)),attracted_liver_anchor_pairs=int(np.count_nonzero(scores>=t)))
  else:result['largest_recall_at_majority_precision']=None
  result['majority_precision_sweep_budgets']=[r['budget'] for r in result['sweep'] if r['liver_target_precision'] is not None and r['liver_target_precision']>.5]
 return result
reference=np.asarray(np.load(OUT/f'scores_kmeans8_{pool}.npy',mmap_mode='r')).reshape(-1)
reference_point=basic(category,reference,.5);reference_recall=reference_point['liver_pair_recall']
results={}
for variant in meta['variants']:
 x=np.asarray(np.load(OUT/f'scores_{variant}_{pool}.npy',mmap_mode='r')).reshape(-1)
 cuts=meta['thresholds'][variant][pool];r=summarize(category,x,cuts,reference_recall)
 r['per_anchor_patient']={}
 for patient in sorted(set(pid),key=int):
  m=anchor_patient==patient
  r['per_anchor_patient'][patient]=summarize(category[m],x[m],[],None,False)
 if variant=='baseline':r['at_C_shift']=basic(category,x,meta['actual_shifts'][pool])
 results[variant]=r
 print(pool,variant,'AUC',r['auc'],'AP',r['average_precision'],'seconds',elapsed(),flush=True)
sensitivity={}
if pool=='intra':
 positions=np.load(OUT/'pair_positions.npy');same_coord=np.tile(positions//121==positions%121,len(a))
 masks={'exclude_same_coordinate':~same_coord,'exclude_same_nearest_token':(ta!=tb).reshape(-1)}
 for name,mask in masks.items():
  sensitivity[name]={}
  for variant in ['baseline','uncentered','kmeans8']:
   x=np.asarray(np.load(OUT/f'scores_{variant}_{pool}.npy',mmap_mode='r')).reshape(-1)[mask]
   budgets=[.02,.05,.1,.2,.3,.5,.75]
   cuts=[dict(budget=None,threshold=.5)] if variant=='kmeans8' else [dict(budget=r,threshold=float(t)) for r,t in zip(budgets,np.quantile(x,1-np.array(budgets)))]
   sensitivity[name][variant]=dict(excluded_fraction=float(1-mask.mean()),**summarize(category[mask],x,cuts))
result=dict(script=__file__,command=f'PYTHONDONTWRITEBYTECODE=1 python3 audit/scratch/selectivity/score.py --pool {pool}',pool=pool,labels='Only all472 validation masks, used solely for diagnostic scoring. No train labels. ROC/AP, base rates, purity, recall, precision envelopes and majority-precision thresholds use validation labels; attraction rates/quantile thresholds do not.',results=results,kmeans_reference_point=reference_point,sensitivity=sensitivity,patch_liver_fraction=float(patch_labels.mean()),mixed_occupancy_patch_fraction=float(((occupancy>0)&(occupancy<1)).float().mean()),elapsed_seconds=elapsed())
dump(f'score_{pool}.json',result);note(f'Scoring completed: `audit/scratch/selectivity/score.py --pool {pool}` → `score_{pool}.json`; labels restricted to validation diagnostic scoring. Runtime {elapsed():.2f}s.')
