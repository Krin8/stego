"""Exact BASELINE KMeans settings on all patches; disk-backed working array."""
import os
os.environ['IMPROVE_LABEL_FREE']='1'
from pipeline import *
from sklearn.cluster import KMeans
import argparse,warnings
p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True,choices=[0,1,2]);a=p.parse_args()
tag=f'{a.variant}_s{a.seed}';assert not (OUT/f'{tag}_fit.json').exists()
parts=list(features(a.variant));dim=parts[0].shape[-1];cells=parts[0].shape[1];grid=int(np.sqrt(cells))
work=OUT/f'{tag}_work.npy'
x=np.lib.format.open_memmap(work,mode='w+',dtype='float32',shape=(2341*cells,dim))
offset=0
for part in parts:
 for start in range(0,len(part),16):
  v=np.array(part[start:start+16]).reshape(-1,dim)
  if a.variant=='B0':v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
  x[offset:offset+len(v)]=v;offset+=len(v)
assert offset==len(x);x.flush();del parts
km=KMeans(n_clusters=8,random_state=a.seed,n_init=3,max_iter=100,tol=1e-4,algorithm='lloyd',copy_x=False)
print('fit',tag,x.shape,elapsed(),flush=True)
with warnings.catch_warnings(record=True) as ws:km.fit(x)
assign=np.empty(len(x),np.uint8);gaps=np.empty(len(x),np.float32)
for start in range(0,len(x),32768):
 v=x[start:start+32768];d=km.transform(v)**2
 assign[start:start+len(v)]=km.predict(v)
 nearest=np.partition(d,1,axis=1)[:,:2];gaps[start:start+len(v)]=nearest[:,1]-nearest[:,0]
temperature=max(1e-6,float(np.median(gaps))/float(np.log(4)))
np.save(OUT/f'{tag}_grids.npy',assign.reshape(2341,grid,grid));np.save(OUT/f'{tag}_centers.npy',km.cluster_centers_)
dump(f'{tag}_fit.json',dict(script=__file__,variant=a.variant,seed=a.seed,labels_used=False,
 grid=grid,dim=dim,patches=len(x),parameters=km.get_params(),inertia=float(km.inertia_),iterations=int(km.n_iter_),
 temperature=temperature,warnings=[str(w.message) for w in ws],elapsed_seconds=elapsed(),
 memory_note='copy_x=False on disposable scratch working matrix; all algorithm and initialization parameters unchanged'))
del x;work.unlink()
print('DONE',tag,elapsed(),flush=True)
