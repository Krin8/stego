"""All-train code k-means; annotation-free fitting and matched probe inference."""
from common import *
from sklearn.cluster import KMeans
import argparse,warnings
p=argparse.ArgumentParser();p.add_argument('--name',choices=['seed0','seed1','seed2','best600','final'],required=True);args=p.parse_args()
inv=json.loads((FORE/'inventory.json').read_text());path=inv[args.name]['path']
if not Path(path).is_file():dump(f'codes_{args.name}.json',dict(script=__file__,status='UNVERIFIED',reason='Checkpoint missing',path=path));raise SystemExit()
model,ck=load(path);ref=torch.load(inv['seed0']['path'],map_location='cpu',mmap=True,weights_only=False)
keys=[k for k in ck['state_dict'] if k.startswith('net.model.')];assert keys and all(torch.equal(ck['state_dict'][k],ref['state_dict'][k]) for k in keys)
train_features=np.load(BASE/'train_features.npy',mmap_mode='r');val_features=np.load(FORE/'val_features.npy',mmap_mode='r');meta=json.loads((BASE/'samples.json').read_text())
original=model.net.model.get_intermediate_layers;provider={}
def from_diagnostic_features(img,*a,**kw):
 assert kw==dict(n=1,reshape=True,norm=True) and len(img)==len(provider['features'])
 return (provider['features'],)
def cached_forward(f):
 provider['features']=f
 return model.net(torch.zeros(len(f),3,224,224))
errors={};codes={}
with torch.inference_mode():
 for split,source in [('train',train_features),('val',val_features)]:
  ds=dataset(model.cfg,split);assert [r['sample_id'] for r in ds.dataset.records]==meta[split+'_ids']
  ds.dataset=ImagesOnlyDataset(ds.dataset) # no annotation needed even for forward verification
  x=torch.stack([ds[i]['img'] for i in range(16)]);f_live,c_live=model.net(x)
  def block(start):
   f=torch.from_numpy(np.array(source[start:start+16]))
   return f.reshape(-1,14,14,768).permute(0,3,1,2).contiguous() if split=='train' else f
  error=float((f_live-block(0)).abs().max());assert error==0
  model.net.model.get_intermediate_layers=from_diagnostic_features
  _,c=cached_forward(block(0));code_error=float((c-c_live).abs().max());assert code_error==0
  errors[split]=dict(feature_max_abs_difference=error,code_max_abs_difference=code_error)
  out=np.empty((len(ds),70,14,14),dtype=np.float32)
  for start in range(0,len(ds),16):
   _,c=cached_forward(block(start));out[start:start+len(c)]=c.numpy()
  codes[split]=out;model.net.model.get_intermediate_layers=original
  print(args.name,split,'codes ready',elapsed(),flush=True)
train=F.normalize(torch.from_numpy(codes['train']),dim=1).permute(0,2,3,1).reshape(-1,70).numpy()
val=F.normalize(torch.from_numpy(codes['val']),dim=1).permute(0,2,3,1).reshape(-1,70).numpy()
assert train.shape==(1869*196,70) and val.shape==(472*196,70)
fits={};results={}
# Finish ALL unsupervised fits/predictions before loading validation masks.
for seed in [0,1,2]:
 km=KMeans(n_clusters=8,random_state=seed,n_init=3,max_iter=100,tol=1e-4,algorithm='lloyd')
 with warnings.catch_warnings(record=True) as ws:km.fit(train)
 fits[seed]=km;np.save(OUT/f'{args.name}_kmeans_seed{seed}_centers.npy',km.cluster_centers_)
 grids=km.predict(val).reshape(472,1,14,14)
 pred=F.interpolate(torch.tensor(grids,dtype=torch.float32),(224,224),mode='nearest')[:,0].numpy().astype(np.uint8)
 results[seed]=dict(nearest_predictions=pred,iterations=int(km.n_iter_),inertia=float(km.inertia_),warnings=[str(w.message) for w in ws])
 print(args.name,'fit seed',seed,'seconds',elapsed(),flush=True)
bilinear={seed:np.empty((472,224,224),dtype=np.uint8) for seed in fits};probe=np.empty((472,224,224),dtype=np.uint8)
with torch.inference_mode():
 for start in range(0,472,16):
  up=F.interpolate(torch.from_numpy(codes['val'][start:start+16]),(224,224),mode='bilinear',align_corners=False)
  _,probs=model.cluster_probe(up,None);probe[start:start+len(up)]=probs.argmax(1).numpy()
  unit=F.normalize(up,dim=1).permute(0,2,3,1).reshape(-1,70).numpy()
  for seed,km in fits.items():bilinear[seed][start:start+len(up)]=km.predict(unit).reshape(len(up),224,224).astype(np.uint8)
old=np.load(FORE/f'{args.name}_assignments.npy',mmap_mode='r');assert np.array_equal(probe,old)
own=evaluate(probe,8);old_report=json.loads((FORE/'task1.json').read_text())[args.name]
assert own['majority']==old_report['majority'] and own['optimal']==old_report['optimal']
for seed,r in results.items():
 r['nearest']=evaluate(r.pop('nearest_predictions'),8);r['matched_bilinear']=evaluate(bilinear[seed],8)
result=dict(script=__file__,command=f'python3 -B audit/scratch/pairs/codes.py --name {args.name}',checkpoint=path,name=args.name,backbone_all_parameters_equal=True,live_forward_errors=errors,own_probe_assignments_equal_forensics=True,train_slices=1869,train_patches=len(train),val_slices=472,code_dim=70,normalization='Per-vector L2; no train mean subtraction. For matched inference: raw code bilinear-upsampled then L2, exactly as own probe normalization order.',primary_kmeans_seed=0,kmeans_seeds=[0,1,2],n_init=3,max_iter=100,tol=1e-4,own_probe=own,kmeans=results,frozen_reference={str(seed):json.loads((BASE/f'k8_c0_s{seed}.json').read_text()) for seed in fits},labels='Only all472 validation masks for mapping/scoring after fitting. All1869 train images/codes used without masks. Nearest-grid inference compares with frozen-feature BASELINE; matched bilinear inference isolates assignment from interpolation differences with the probe.',elapsed_seconds=elapsed())
dump(f'codes_{args.name}.json',result);note(f'Code diagnostic {args.name}: `codes.py --name {args.name}` → `codes_{args.name}.json`; runtime {elapsed():.2f}s; source/probe exact reproduction verified; no training labels.')
print(json.dumps(dict(name=args.name,own_probe=own['optimal']['iou'],kmeans={seed:{key:100*r[key]['optimal']['iou'] for key in ['nearest','matched_bilinear']} for seed,r in results.items()},elapsed_seconds=elapsed()),indent=2))
