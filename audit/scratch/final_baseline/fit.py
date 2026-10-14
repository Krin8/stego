"""Label-free fit on every CT image; no annotation access permitted."""
import os
os.environ['FINAL_BASELINE_LABEL_FREE']='1'
from common import *
from sklearn.cluster import KMeans
import argparse,warnings
p=argparse.ArgumentParser();p.add_argument('--k',type=int,choices=[8,16],required=True)
p.add_argument('--seed',type=int,choices=[0,1,2],required=True);a=p.parse_args()
tag=f'k{a.k}_s{a.seed}'
assert not (OUT/(tag+'_fit.json')).exists(),'Refusing to overwrite a completed fit'
verification=json.loads((OUT/'verification.json').read_text())
base=ROOT/'audit/scratch/baseline'
train=np.load(base/'train_features.npy',mmap_mode='r')
val=np.load(FORE/'val_features.npy',mmap_mode='r')
x=np.empty((2341,196,768),dtype=np.float32)
x[:len(train)]=train;x[len(train):]=val.transpose(0,2,3,1).reshape(472,196,768)
x=x.reshape(-1,768)
assert np.isfinite(x).all()
x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
km=KMeans(n_clusters=a.k,random_state=a.seed,n_init=3,max_iter=100,tol=1e-4,algorithm='lloyd')
print('Fitting',tag,x.shape,flush=True)
with warnings.catch_warnings(record=True) as ws:km.fit(x)
grids=km.predict(x).reshape(2341,14,14).astype(np.uint8)
np.save(OUT/(tag+'_grids.npy'),grids)
np.save(OUT/(tag+'_centers.npy'),km.cluster_centers_)
r=dict(script=__file__,k=a.k,seed=a.seed,labels_used=False,fit_slices=2341,fit_patients=20,
 patch_vectors=len(x),dimensions=768,centering='none',normalization='L2 per patch',
 parameters=km.get_params(),iterations=int(km.n_iter_),inertia=float(km.inertia_),
 ordered_ids_sha256=verification['ordered_ids_sha256'],warnings=[str(w.message) for w in ws],
 grid_shape=list(grids.shape),elapsed_seconds=elapsed())
dump(tag+'_fit.json',r);note('Completed label-free fit: '+json.dumps(r));print(json.dumps(r),flush=True)
