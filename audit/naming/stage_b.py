"""Only now open labels: score frozen StageA names, prior assisted mappings and diagnostics."""
import os
os.environ['NAMING_STAGE']='B'
from common import *
import argparse
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--fit',choices=['inductive','transductive'],required=True);p.add_argument('--seed',type=int,choices=[0,1,2],required=True);a=p.parse_args();key=f'{a.fit}_s{a.seed}'
m=sealed_manifest();assert not (OUT/f'{key}_scores.json').exists()
if not (OUT/'stageB_started.json').exists():dump('stageB_started.json',dict(utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=sha(OUT/'stageA_manifest.json'),parameters_frozen=True))
names=json.loads((OUT/f'{key}_naming.json').read_text());rr=json.loads((OUT/'image_records.json').read_text());features=np.load(OLD/'B2_448_features.npy',mmap_mode='r');grids=np.load(OUT/f'{key}_grids.npy',mmap_mode='r');centers=np.load(OUT/f'{key}_centers.npy')
# All label reads are below this point and are recorded; none can affect sealed decisions.
y={res:np.load(OLD/f'scoring_masks_{"native" if res==512 else 224}.npy',mmap_mode='r') for res in [224,512]}
inds=[i for i,r in enumerate(rr) if a.fit=='transductive' or r['split']=='val'];pats=sorted({rr[i]['patient'] for i in inds},key=int)
if a.fit=='inductive':
 assisted=json.loads((FINAL/f'inductive_seed{a.seed}.json').read_text());train_conf=np.asarray(assisted['mapping_confusion'])
 best=max((scores(train_conf,[j for j in range(8) if bits>>j&1]) for bits in range(256)),key=lambda s:s['iou'])['clusters']
 references={res:{pat:dict(majority=assisted['liver_clusters'],optimal=best) for pat in pats} for res in [224,512]}
 reference_label_use='Stored train16 strict-majority confusion/mapping; optimal subset derived from stored train confusion, not validation labels; both resolutions same mapping as FINAL_PIPELINE'
else:
 assisted=json.loads((OLD/f'B2_448_s{a.seed}_scores.json').read_text());references={res:{pat:{rule:assisted['results'][str(res)]['folds'][pat][rule]['clusters'] for rule in ['majority','optimal']} for pat in pats} for res in [224,512]}
 reference_label_use='Stored other19 LOPO mappings at each resolution from IMPROVE; heldout patient scores only'
# Label diagnostic confusion by raw cluster at both scales, before any named masks.
cs={res:{pat:np.zeros((9,2),np.int64) for pat in pats} for res in [224,512]}
for i in inds:
 pr=inverse_grid(grids[i],rr[i],outside=8).astype(np.uint8)
 for res,small in [(512,pr),(224,resize224(pr))]:cs[res][rr[i]['patient']]+=confusion(small,y[res][i],9)
ranking={}
for res in [224,512]:
 total=sum(cs[res].values());all_liver=total[:,1].sum();rows=[]
 for c in range(8):rows.append(dict(cluster=c,liver_pixels=int(total[c,1]),background_pixels=int(total[c,0]),precision=float(total[c,1]/max(1,total[c].sum())),recall=float(total[c,1]/max(1,all_liver)),pixel_fraction=float(total[c].sum()/total.sum())))
 ranking[str(res)]=dict(order_by_precision=sorted(rows,key=lambda r:(-r['precision'],r['cluster'])),outside_crop_liver_pixels=int(total[8,1]),labels='Evaluated patients only; diagnostic ranking; never used for naming')
rules=['N1','N2','N3','reference_majority','reference_optimal']
counts={rule:{variant:{res:{pat:np.zeros((2,2),np.int64) for pat in pats} for res in [224,512]} for variant in ['hard','F1']} for rule in rules}
slice_scores={rule:{variant:{res:{pat:[] for pat in pats} for res in [224,512]} for variant in ['hard','F1']} for rule in rules}
fixed=json.loads((OLD/'figure_selection.json').read_text())['eight_indices'];panels={};agreement={}
for rule in ['N1','N2','N3']:
 chosen=names[rule]['selected_cluster'];subset=[] if chosen is None else [chosen]
 agreement[rule]={str(res):{pat:{r:subset==references[res][pat][r] for r in ['majority','optimal']} for pat in pats} for res in [224,512]}
for num,i in enumerate(inds):
 r=rr[i];pat=r['patient'];im=raw_image(r);cache={}
 for rule in rules:
  for res in [224,512]:
   chosen=names[rule]['selected_cluster'] if rule in ['N1','N2','N3'] else None
   ids=([] if chosen is None else [chosen]) if rule in ['N1','N2','N3'] else references[res][pat][rule.replace('reference_','')]
   subset=tuple(ids)
   if subset not in cache:
    # Reuse exact packaged F1, changing only the sealed semantic cluster subset.
    cache[subset]=predict_features(features[i],im,r,centers,dict(liver_clusters=list(subset),temperature=names['temperature']))
   for variant,pred in zip(['hard','F1'],cache[subset]):
    pr=pred if res==512 else resize224(pred);c=confusion(pr,y[res][i],2);counts[rule][variant][res][pat]+=c;s=scores(c,[1]);slice_scores[rule][variant][res][pat].append(dict(index=i,stem=r['stem'],**s))
    if i in fixed and res==512:panels[f'{rule}_{variant}_{i}']=pred.astype(np.uint8)
 if num%400==0:print(key,num,len(inds),elapsed(),flush=True)
result=dict(producer='stage_b.py',fit=a.fit,seed=a.seed,manifest_sha256=sha(OUT/'stageA_manifest.json'),patients=pats,slices=len(inds),labels='Masks read only in StageB for scoring and precision/recall diagnostics; reference mappings label-assisted as separately stated',reference_label_use=reference_label_use,rules={},cluster_ranking=ranking,seconds=elapsed())
for rule in rules:
 chosen=names[rule]['selected_cluster'] if rule in names else None
 row=dict(selected_cluster=chosen if rule in names else 'patient-dependent reference',abstained=chosen is None if rule in names else None,agreement=agreement.get(rule),results={})
 for variant in ['hard','F1']:
  row['results'][variant]={}
  for res in [224,512]:
   per={pat:dict(pooled=scores(counts[rule][variant][res][pat],[1]),mean_slice={metric:float(np.mean([s[metric] for s in slice_scores[rule][variant][res][pat]])) for metric in ['iou','dice']},slice_count=len(slice_scores[rule][variant][res][pat]),slice_scores=slice_scores[rule][variant][res][pat]) for pat in pats}
   summary={agg:{metric:patient_stats([v[agg][metric] for v in per.values()]) for metric in ['iou','dice']} for agg in ['pooled','mean_slice']}
   row['results'][variant][str(res)]=dict(patients=per,summary=summary)
 result['rules'][rule]=row
np.savez_compressed(OUT/f'{key}_figure_masks.npz',**panels)
dump(f'{key}_scores.json',result);print('SCORED',key,elapsed(),flush=True)
