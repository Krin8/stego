"""Instrument actual correspondence forward: full top-seven val image neighbours."""
from common import *
import modules
records=json.loads((OUT/'records.json').read_text());setup=json.loads((OUT/'setup.json').read_text())
model,ck=load(setup['checkpoints']['seed0']['path']);cfg=model.cfg
assert cfg.pointwise and not cfg.stabalize and not cfg.use_salience and cfg.feature_samples==11
lossfn=model.contrastive_corr_loss_fn;features=torch.from_numpy(np.array(np.load(FORE/'val_features.npy',mmap_mode='r')))
labels=torch.from_numpy(np.array(np.load(FORE/'val_labels.npy',mmap_mode='r'),dtype=np.float32))[:,None]
occupancy=F.avg_pool2d(labels,16);patch_labels=(occupancy>.5).float();assert patch_labels.shape==(472,1,14,14)
nns=np.load(OUT/'val_image_neighbors.npy');k=int(cfg.num_neighbors);assert k==7
pid=np.array([r['patient_id'] for r in records]);series=np.array([r['series_uid'] for r in records]);rank=np.empty(472,dtype=int)
for key in set(zip(pid,series)):
 indices=[i for i in range(472) if (pid[i],series[i])==key]
 assert len({records[i]['physical_position'] for i in indices})==len(indices)
 indices.sort(key=lambda i:records[i]['physical_position']);rank[indices]=np.arange(len(indices))
def relations(a,b):
 samepatient=pid[a]==pid[b];sameseries=samepatient&(series[a]==series[b]);dist=np.abs(rank[a]-rank[b])
 return dict(all=np.ones(len(a),dtype=bool),same_patient=samepatient,cross_patient=~samepatient,same_slice=a==b,adjacent_slice=sameseries&(dist==1),same_or_adjacent=sameseries&(dist<=1),within_3_slices=sameseries&(dist<=3),same_patient_beyond_3=samepatient&~(sameseries&(dist<=3)))
counts={};image_draws={};ctx={};original_helper=lossfn.helper;original_sample=modules.sample
def spy_sample(t,coords):
 if t.shape[1]==768:
  if ctx['feature_samples_seen']<2:ctx['coords'+str(ctx['feature_samples_seen']+1)]=coords
  ctx['feature_samples_seen']+=1
 return original_sample(t,coords)
def hard_labels(indices,coords):
 return F.grid_sample(patch_labels[torch.as_tensor(indices)],coords.permute(0,2,1,3),mode='nearest',padding_mode='reflection',align_corners=True)[:,0].reshape(len(indices),-1).numpy().astype(np.int64)
def capture(f1,f2,c1,c2,shift):
 loss,cd=original_helper(f1,f2,c1,c2,shift)
 assert float((cd-1).abs().max())<1e-6
 # Positive constant scalar codes have cosine one; invert the real returned loss.
 affinity=(float(shift)-loss/cd).detach().numpy().reshape(len(f1),121,121)
 a=np.rint(c1[:,0,0,0].numpy()).astype(int)-1;b=np.rint(c2[:,0,0,0].numpy()).astype(int)-1
 assert np.array_equal(a,ctx['anchors'])
 call=ctx['helper_calls'];typ='intra' if call==0 else 'knn' if call==1 else 'negative'
 if typ=='knn':assert np.array_equal(b,ctx['positives'])
 if typ=='negative':assert np.all(a!=b)
 ya=hard_labels(a,ctx['coords1']);yb=ya if typ=='intra' else hard_labels(b,ctx['coords2'])
 category=2*ya[:,:,None]+yb[:,None,:];attract=affinity>float(shift)
 combined=category+4*attract+8*np.arange(len(a))[:,None,None]
 per_edge=np.bincount(combined.ravel(),minlength=len(a)*8).reshape(len(a),2,4)
 for name,mask in relations(a,b).items():
  key=typ+'/'+name
  if key not in counts:counts[key]=np.zeros((2,4),dtype=np.int64);image_draws[key]=0
  counts[key]+=per_edge[mask].sum(0);image_draws[key]+=int(mask.sum())
 ctx['helper_calls']+=1
 return loss,cd
lossfn.helper=capture;modules.sample=spy_sample
rng=np.random.default_rng(SEED)
with torch.inference_mode():
 for neighbour_rank in range(1,k+1):
  order=rng.permutation(472)
  for j,start in enumerate(range(0,472,16)):
   a=order[start:start+16];b=nns[a,neighbour_rank];n=len(a)
   ctx.clear();ctx.update(anchors=a,positives=b,helper_calls=0,feature_samples_seen=0)
   ca=torch.from_numpy(a.astype(np.float32)+1)[:,None,None,None].expand(n,1,14,14)
   cb=torch.from_numpy(b.astype(np.float32)+1)[:,None,None,None].expand(n,1,14,14)
   torch.manual_seed(SEED+neighbour_rank*100+j)
   lossfn(features[a],features[b],None,None,ca,cb,features[a],ca)
   assert ctx['helper_calls']==2+min(int(cfg.neg_samples),n)
  print('rank',neighbour_rank,'seconds',elapsed(),flush=True)
modules.sample=original_sample;lossfn.helper=original_helper
def label_summary(c):
 total=int(c.sum());ll=int(c[3]);mixed=int(c[1]+c[2]);anchor_liver=int(c[2]+c[3])
 return dict(pairs=total,same_label=(int(c[0])+ll)/total if total else None,liver_liver=ll/total if total else None,mixed=mixed/total if total else None,background_background=int(c[0])/total if total else None,target_liver_given_anchor_liver=ll/anchor_liver if anchor_liver else None,anchor_liver_pairs=anchor_liver,counts_BB_BL_LB_LL=c.tolist())
summary={}
for key,matrix in counts.items():
 allpairs=matrix.sum(0);selected=matrix[1];n=int(allpairs.sum())
 summary[key]=dict(image_pair_draws=image_draws[key],attraction_rate=int(selected.sum())/n if n else None,random_unfiltered=label_summary(allpairs),attracted=label_summary(selected),attraction_given_label_pair={label:int(selected[i])/int(allpairs[i]) if allpairs[i] else None for i,label in enumerate(['BB','BL','LB','LL'])})
# Image-cache locality is exact; it has no patch-level neighbour IDs.
a=np.repeat(np.arange(472),k);b=nns[:,1:k+1].ravel();rel=relations(a,b)
liver=patch_labels[:,0].sum((1,2)).numpy().astype(np.int64);weights=liver[a]
locality=dict(image_edges=len(a),unweighted={key:float(mask.mean()) for key,mask in rel.items()},liver_anchor_weighted={key:float(weights[mask].sum()/weights.sum()) for key,mask in rel.items()},liver_patch_anchor_image_edges=int(weights.sum()),liver_patch_count=int(liver.sum()),total_patch_count=int(patch_labels.numel()),neighbour_image_contains_any_liver=float(weights[liver[b]>0].sum()/weights.sum()),target_liver_patch_fraction_uniform_all_patches=float((weights*liver[b]).sum()/(weights.sum()*196)),all_patch_pairs_LL=int((weights*liver[b]).sum()),all_liver_anchor_patch_pairs=int(weights.sum()*196))
result=dict(script=__file__,seed=SEED,protocol='Seven passes enumerate every top7 non-self validation image-cache edge; each pass shuffles all472 images in batches16 (last8). Actual loss.forward samples11x11 coordinates and min(20,batch) nonzero rolls. Baseline is all unfiltered uniformly sampled patch pairs of identical image-pair type. Positive scalar ID codes expose the exact helper affinity; no learned-code gradient gate is assessed.',labels='Only all472 validation masks; 16x16 pixel block majority labels, ties background, nearest sampled at actual feature coordinates. No labels choose images, neighbours, shifts or sampled coordinates.',shifts=setup['shifts'],patch_label_occupancy=dict(liver_fraction=float(patch_labels.mean()),mixed_occupancy_patch_fraction=float(((occupancy>0)&(occupancy<1)).float().mean())),summary=summary,cache_locality=locality,elapsed_seconds=elapsed())
dump('purity.json',result);note('Pair purity completed: `audit/scratch/pairs/purity.py` → `purity.json`, runtime '+str(elapsed())+'s. Only validation masks used for diagnostic labels.');print(json.dumps(dict(locality=locality,overall={key:r for key,r in summary.items() if key.endswith('/all')},elapsed_seconds=elapsed()),indent=2))
