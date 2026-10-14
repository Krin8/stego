"""Locked K8/K2 all-cohort label-free fitting followed by per-patient LOPO mapping."""
from common import *
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import argparse
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['D2','D3'],required=True);p.add_argument('--k',type=int,choices=[2,8],required=True);p.add_argument('--seed',type=int,choices=[0,1,2],required=True);a=p.parse_args()
tag=f'{a.arm}_k{a.k}_s{a.seed}';assert not (OUT/f'{tag}_scores.json').exists();rr=records()
os.environ['LABEL_FREE']='1';f=np.load(OUT/f'{a.arm}_features.npy',mmap_mode='r');work=OUT/f'{tag}_work.npy'
x=np.lib.format.open_memmap(work,mode='w+',dtype=np.float32,shape=(2341*784,768))
for start in range(0,2341,16):x[start*784:min(start+16,2341)*784]=f[start:start+16].reshape(-1,768)
x.flush();km=KMeans(n_clusters=a.k,random_state=a.seed,n_init=3,max_iter=100,tol=1e-4,algorithm='lloyd',copy_x=False)
dump(f'{tag}_parameters.json',dict(producer='fit_score.py',parameters=km.get_params(),labels_used=False,protocol_sha256=sha(OUT/'PROTOCOL.json')))
print('FIT',tag,elapsed(),flush=True);km.fit(x)
grid=np.empty((2341,28,28),np.uint8)
for start in range(0,2341,16):grid[start:start+16]=km.predict(np.asarray(f[start:start+16]).reshape(-1,768)).reshape(-1,28,28)
np.save(OUT/f'{tag}_centers.npy',km.cluster_centers_);np.save(OUT/f'{tag}_grids.npy',grid)
dump(f'{tag}_fit.json',dict(producer='fit_score.py',labels_used=False,iterations=int(km.n_iter_),inertia=float(km.inertia_),seconds=elapsed()))
del x;work.unlink();os.environ['LABEL_FREE']='0'
labels={res:np.load(OLD/f'scoring_masks_{"native" if res==512 else 224}.npy',mmap_mode='r') for res in [224,512]}
pats=sorted({r['patient'] for r in rr},key=int);conf={res:{pat:np.zeros((a.k+1,2),np.int64) for pat in pats} for res in [224,512]}
for i,r in enumerate(rr):
 native=inverse_grid(grid[i],r,outside=a.k).astype(np.uint8)
 for res,pr in [(512,native),(224,resize224(native))]:conf[res][r['patient']]+=confusion(pr,labels[res][i],a.k+1)
results={}
for res,cs in conf.items():
 total=sum(cs.values());folds={}
 for pat,c in cs.items():
  train=total-c
  if a.k==8:
   # Exact original 8-cluster exhaustive scoring includes outside-crop false negatives,
   # while never allowing the outside-crop row to become a liver cluster.
   candidates=[scores(train,[i for i in range(8) if mask>>i&1]) for mask in range(256)]
   subsets={'majority':np.flatnonzero(train[:8,1]>train[:8,0]).tolist(),'optimal':max(candidates,key=lambda s:s['iou'])['clusters']}
  else:
   row,col=linear_sum_assignment(-train[:2]);subset=row[col==1].tolist();subsets={'Hungarian':subset}
  folds[pat]={name:scores(c,ids) for name,ids in subsets.items()};folds[pat]['mapping_patients']=[q for q in pats if q!=pat]
 results[str(res)]=dict(patient_confusions={pat:c.tolist() for pat,c in cs.items()},folds=folds)
dump(f'{tag}_scores.json',dict(producer='fit_score.py',arm=a.arm,K=a.k,seed=a.seed,labels_used='Other19 patient labels map; heldout patient labels score; no labels fit; independently at224/native512',results=results,seconds=elapsed()))
print('DONE',tag,elapsed(),flush=True)
