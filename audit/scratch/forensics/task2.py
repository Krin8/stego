from common import *
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,mutual_info_score
import itertools

def agreement(cm):
 cm=np.asarray(cm,dtype=np.int64);n=int(cm.sum());a=cm.sum(1);b=cm.sum(0)
 pairs=lambda v:sum(int(x)*(int(x)-1)//2 for x in np.asarray(v).ravel())
 s=pairs(cm);sa=pairs(a);sb=pairs(b);den=n*(n-1)/2;expected=sa*sb/den;top=(sa+sb)/2
 ari=(s-expected)/(top-expected) if top!=expected else 1.
 mi=mutual_info_score(None,None,contingency=cm)
 entropy=lambda v:-sum((x/n)*np.log(x/n) for x in v if x)
 h=(entropy(a)+entropy(b))/2;nmi=mi/h if h else 1.
 return dict(ARI=float(ari),NMI=float(nmi),contingency=cm.tolist())
def compare(x,y):
 cm=np.zeros((8,8),dtype=np.int64)
 for start in range(0,len(x),16):cm+=np.bincount((np.asarray(x[start:start+16],dtype=np.int64)*8+np.asarray(y[start:start+16],dtype=np.int64)).ravel(),minlength=64).reshape(8,8)
 return agreement(cm)
a0=np.array([0,1,2,0,1,1,1,2]);b0=np.array([0,2,1,1,1,1,2,2]);cm=np.bincount(a0*8+b0,minlength=64).reshape(8,8);t=agreement(cm)
assert abs(t['ARI']-adjusted_rand_score(a0,b0))<1e-12 and abs(t['NMI']-normalized_mutual_info_score(a0,b0))<1e-12
main=json.loads((OUT/'task1.json').read_text());pred={k:np.load(OUT/f'{k}_assignments.npy',mmap_mode='r') for k in main}
result=dict(script=__file__,protocol='Pixelwise label-free ARI/NMI on all472x224x224 assignments; NMI arithmetic entropy mean; tested against sklearn. No GT used to fit KMeans or compute agreement.',pairwise={f'{x} vs {y}':compare(pred[x],pred[y]) for x,y in itertools.combinations(pred,2)},active={k:v['active'] for k,v in main.items()},kmeans={})
features=np.load(OUT/'val_features.npy',mmap_mode='r');plan=json.loads((ROOT/'audit/scratch/diag/plan.json').read_text());order=json.loads((OUT/'val_order.json').read_text());oldids=[order.index(r['sample_id']) for r in plan['val']]
oldf=np.load(ROOT/'audit/scratch/diag/val_block12.npy',mmap_mode='r');fresh=np.asarray(features[oldids]).transpose(0,2,3,1).reshape(100,196,768)
result['DIAG_feature_max_abs_difference']=float(np.max(np.abs(fresh-oldf)))
for centered in [0,1]:
 train=np.array(np.load(ROOT/'audit/scratch/diag/train_block12.npy',mmap_mode='r')).reshape(-1,768)
 val=np.array(features).transpose(0,2,3,1).reshape(-1,768);mean=train.mean(axis=0,dtype=np.float64).astype(np.float32)
 if centered:train-=mean;val-=mean
 train/=np.maximum(np.linalg.norm(train,axis=1,keepdims=True),1e-12);val/=np.maximum(np.linalg.norm(val,axis=1,keepdims=True),1e-12)
 km=KMeans(n_clusters=8,random_state=SEED,n_init=3,max_iter=100,tol=1e-4,algorithm='lloyd').fit(train)
 grids=km.predict(val).reshape(472,1,14,14)
 full=F.interpolate(torch.tensor(grids,dtype=torch.float32),(224,224),mode='nearest')[:,0].numpy().astype(np.uint8)
 np.save(OUT/f'frozen_kmeans_center{centered}.npy',full)
 y=np.load(OUT/'val_labels.npy',mmap_mode='r');conf=np.bincount((full[oldids].astype(np.int64)*2+y[oldids]).ravel(),minlength=16).reshape(8,2)
 old=json.loads((ROOT/f'audit/scratch/diag/ceiling_b12_c{centered}.json').read_text())['kmeans']['8']
 result['kmeans'][str(centered)]=dict(train_patches=len(train),train_only_mean_subtracted=bool(centered),iterations=km.n_iter_,inertia=km.inertia_,DIAG100_reproduced_majority_iou=mapping_report(conf)['majority']['iou'],DIAG100_previous_majority_iou=old['majority_oracle']['liver_iou'],agreement={k:compare(p,full) for k,p in pred.items()})
 dump('task2.json',result);print('KMeans centered',centered,'done',elapsed(),flush=True)
result['elapsed_seconds']=elapsed();dump('task2.json',result);note('Task2 finished: label-free pairwise ARI/NMI and both original DIAG K8 mean-subtraction settings. Producer`task2.py`; detail`task2.json`. No validation labels used for fitting or partition comparisons.');print('DONE',elapsed())
