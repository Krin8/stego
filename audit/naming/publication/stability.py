"""Label-free seed comparison using sealed names and unchanged prediction functions."""
import os,sys,itertools
from pathlib import Path
os.environ['NAMING_STAGE']='A'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from scipy.optimize import linear_sum_assignment
m=sealed_manifest();rr=json.loads((OUT/'image_records.json').read_text());feat=np.load(OLD/'B2_448_features.npy',mmap_mode='r');pairs=list(itertools.combinations(range(3),2));answer={}
def calc(v):
 inter,a,b,d,n,identical,slices=map(int,v);union=a+b-inter
 return dict(intersection=inter,positive_pixels_a=a,positive_pixels_b=b,different_pixels=d,total_pixels=n,disagreement_fraction=d/max(1,n),binary_dice=2*inter/(a+b) if a+b else 1.,binary_iou=inter/union if union else 1.,identical_slices=identical,slices=slices)
for fit in ['inductive','transductive']:
 ids=[i for i,r in enumerate(rr) if fit=='transductive' or r['split']=='val'];fitids=[i for i,r in enumerate(rr) if fit=='transductive' or r['split']=='train'];pats=sorted({rr[i]['patient'] for i in ids},key=int)
 names=[json.loads((OUT/f'{fit}_s{s}_naming.json').read_text()) for s in range(3)];cent=[np.load(OUT/f'{fit}_s{s}_centers.npy') for s in range(3)];grid=[np.load(OUT/f'{fit}_s{s}_grids.npy',mmap_mode='r') for s in range(3)];out=[]
 for a,b in pairs:
  partition={}
  for which,indices in [('fit_images',fitids),('evaluated_images',ids)]:
   conf=np.bincount((grid[a][indices].astype(np.int64)*8+grid[b][indices]).ravel(),minlength=64).reshape(8,8);ra,cb=linear_sum_assignment(-conf);mapping=np.empty(8,int);mapping[ra]=cb
   partition[which]=dict(a_to_b=mapping.tolist(),confusion=conf.tolist(),aligned_agreement=float(conf[ra,cb].sum()/conf.sum()),selected_ids_correspond=bool(mapping[names[a]['N1']['selected_cluster']]==names[b]['N1']['selected_cluster']))
  out.append(dict(seeds=[a,b],partition=partition,predictions={var:{str(res):{pat:np.zeros(7,np.int64) for pat in pats} for res in [224,512]} for var in ['hard','F1']}))
 for n,i in enumerate(ids):
  im=raw_image(rr[i]);pred=[predict_features(feat[i],im,rr[i],cent[s],dict(liver_clusters=[names[s]['N1']['selected_cluster']],temperature=names[s]['temperature'])) for s in range(3)]
  for res in [224,512]:
   for v,var in enumerate(['hard','F1']):
    ps=[p[v] if res==512 else resize224(p[v]) for p in pred]
    for k,(a,b) in enumerate(pairs):
     pa,pb=ps[a].astype(bool),ps[b].astype(bool);diff=np.count_nonzero(pa!=pb)
     out[k]['predictions'][var][str(res)][rr[i]['patient']]+=np.array([np.count_nonzero(pa&pb),pa.sum(),pb.sum(),diff,pa.size,int(diff==0),1],np.int64)
  if n%400==0:print(fit,n,len(ids),elapsed(),flush=True)
 for row in out:
  for var in ['hard','F1']:
   for res in ['224','512']:
    v=row['predictions'][var][res];row['predictions'][var][res]=dict(pooled=calc(sum(v.values())),patients={p:calc(x) for p,x in v.items()})
 answer[fit]=dict(selected_clusters=[n['N1']['selected_cluster'] for n in names],patients=pats,slices=len(ids),pairs=out)
dump('publication/stability.json',dict(producer='publication/stability.py',labels_read=False,blocked_attempts=BLOCKED,read_paths=sorted(READS),manifest_sha256=sha(OUT/'stageA_manifest.json'),results=answer,seconds=elapsed(),empty_mask_pair_convention='Two empty predictions have inter-seed Dice/IoU1; this is mask-agreement analysis, not the inherited ground-truth scoring convention.'))
print('STABILITY DONE',elapsed(),flush=True)
