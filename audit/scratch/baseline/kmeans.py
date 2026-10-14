"""One bounded all-train KMeans fit; annotations are opened only after fit/predict."""
from common import *
from sklearn.cluster import KMeans
import argparse,warnings
p=argparse.ArgumentParser();p.add_argument('--k',type=int,choices=[8,16],required=True);p.add_argument('--center',type=int,choices=[0,1],required=True);p.add_argument('--seed',type=int,choices=[0,1,2],required=True);args=p.parse_args()
name=f'k{args.k}_c{args.center}_s{args.seed}'
train=np.array(np.load(OUT/'train_features.npy',mmap_mode='r')).reshape(-1,768)
val=np.array(np.load(FORE/'val_features.npy',mmap_mode='r')).transpose(0,2,3,1).reshape(-1,768)
assert len(train)==1869*196 and len(val)==472*196
if args.center:
 mean=np.load(OUT/'train_mean.npy');train-=mean;val-=mean
train/=np.maximum(np.linalg.norm(train,axis=1,keepdims=True),1e-12)
val/=np.maximum(np.linalg.norm(val,axis=1,keepdims=True),1e-12)
km=KMeans(n_clusters=args.k,random_state=args.seed,n_init=3,max_iter=100,tol=1e-4,algorithm='lloyd')
print('fit starts',name,elapsed(),flush=True)
with warnings.catch_warnings(record=True) as ws:km.fit(train)
fit_elapsed=elapsed();print('fit done',name,fit_elapsed,'iterations',km.n_iter_,flush=True)
grids=km.predict(val).reshape(472,1,14,14)
pred=F.interpolate(torch.tensor(grids,dtype=torch.float32),(224,224),mode='nearest')[:,0].numpy().astype(np.uint8)
np.save(OUT/f'{name}_assignments.npy',pred);np.save(OUT/f'{name}_centers.npy',km.cluster_centers_)
result=evaluate(pred,args.k)
result.update(method='Frozen K-means'+(' + train mean subtraction' if args.center else ''),K=args.k,seed=args.seed,centered=bool(args.center),script=__file__,command=f'python3 -B audit/scratch/baseline/kmeans.py --k {args.k} --center {args.center} --seed {args.seed}',train_slices=1869,fit_patches=len(train),val_slices=472,n_init=3,max_iter=100,iterations=int(km.n_iter_),inertia=float(km.inertia_),tolerance=1e-4,algorithm='lloyd',mapping_search='exhaustive all 2^K subsets; FORENSICS scores and definitions',labels='No labels for fit/mean/assignment. Global mappings fit+score all472 val labels; LOPO mapping fit three val patients, score held-out patient.',warnings=[str(w.message) for w in ws],fit_elapsed_seconds=fit_elapsed,elapsed_seconds=elapsed())
dump(f'{name}.json',result);note(f'{name}: majority IoU={result["majority"]["iou"]}, optimal IoU={result["optimal"]["iou"]}; LOPO mean IoU={result["lopo_summary"]["iou"]["mean"]}; producer `{result["command"]}`, `{name}.json`, runtime {elapsed():.2f}s.')
print(name,'majority',result['majority']['iou'],'optimal',result['optimal']['iou'],'LOPO',result['lopo_summary'],'seconds',elapsed(),flush=True)
