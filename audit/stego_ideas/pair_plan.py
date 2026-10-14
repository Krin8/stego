"""Label-free 448 affinity/coordinate plans, fixed before scoring annotations."""
from common import *
SEED=20261003
reuse(ROOT/'audit/scratch/pairs/purity.py','hard_labels',globals())
# Existing PAIRS nearest sampler receives token IDs here, not annotations.
patch_labels=torch.arange(784,dtype=torch.float32).reshape(1,1,28,28).expand(472,-1,-1,-1)
features=feature_batch(VAL)
meta=read(ROOT/'audit/scratch/pairs/records.json');lookup={r['sample_id']:r for r in meta}
records=[lookup[RR[i]['sample_id']] for i in VAL]
groups={}
for i,r in enumerate(records):groups.setdefault((r['patient_id'],r['series_uid']),[]).append(i)
for ids in groups.values():
 ids.sort(key=lambda i:records[i]['physical_position']);assert len({records[i]['physical_position'] for i in ids})==len(ids)
# Execute the sealed 448 cache's exact pooled-feature/cosine/ranking statements,
# against validation images only; save diagnostic indices only in this output directory.
src=(ROOT/'experiments/stego_biomedclip/preflight.py').read_text()
fragment=src[src.index('pooled=np.stack(pooled)'):src.index('assert np.all(neighbors')]
ns=dict(np=np,pooled=[np.array(FEATURES[i]).mean(0) for i in VAL]);exec(compile(fragment,'sealed preflight 448 neighbor recipe','exec'),ns)
neighbors=ns['neighbors'];assert neighbors.shape==(472,30) and np.all(neighbors!=np.arange(472)[:,None])
np.save(OUT/'validation_neighbors448.npy',neighbors)
crit=ContrastiveCorrelationLoss(SimpleNamespace(**CFG));positions=np.random.default_rng(SEED+77).choice(121*121,2048,replace=False)
pools={};edge_info={}
def make_pool(name,edges,colocated=False):
 data={k:[] for k in ['a','b','ta','tb','affinity']};random_data={k:[] for k in data}
 rng=np.random.default_rng(SEED);order=rng.permutation(len(edges));edges=np.array(edges,dtype=int)[order]
 for start in range(0,len(edges),16):
  a,b=edges[start:start+16].T;n=len(a);torch.manual_seed(SEED+start)
  coords1=torch.rand(n,11,11,2)*2-1;coords2=coords1 if colocated or name=='same-image' else torch.rand(n,11,11,2)*2-1
  fa=sample(features[a],coords1);fb=sample(features[b],coords2)
  one=torch.ones(n,1,11,11);loss,cd=crit.helper(fa,fb,one,one,0.)
  assert torch.max((cd-1).abs())==0
  aff=(-loss).numpy().reshape(n,121,121);ta=hard_labels(a,coords1);tb=hard_labels(b,coords2)
  if colocated:
   score=aff[:,np.arange(121),np.arange(121)];ia,ib=ta,tb
  else:score=aff.reshape(n,-1)[:,positions];ia,ib=ta[:,positions//121],tb[:,positions%121]
  for k,v in [('a',a),('b',b),('ta',ia),('tb',ib),('affinity',score)]:data[k].append(v)
  if colocated:
   for k,v in [('a',a),('b',b),('ta',ta[:,positions//121]),('tb',tb[:,positions%121]),('affinity',aff.reshape(n,-1)[:,positions])]:random_data[k].append(v)
  if start%400==0:print(name,start,len(edges),round(time.monotonic()-START,1),flush=True)
 for suffix,d in [('',data)]+([('_random',random_data)] if colocated else []):
  key=name+suffix
  for k,v in d.items():np.save(OUT/f'{key}_{k}.npy',np.concatenate(v))
  x=np.concatenate(d['affinity']).ravel();cuts=[float(np.quantile(x,1-r)) for r in [.05,.1,.2]]
  pools[key]=dict(edges=len(edges),pairs=len(x),thresholds=cuts,attraction=[float((x>t).mean()) for t in cuts])
 if colocated:
  mm=np.array([abs(records[int(a)]['physical_position']-records[int(b)]['physical_position']) for a,b in edges])
  edge_info[name]=dict(directed_edges=len(edges),anchor_slices=len(set(edges[:,0])),distance_mm=dict(min=float(mm.min()),median=float(np.median(mm)),max=float(mm.max())),patient_counts={p:sum(records[int(a)]['patient_id']==p for a,b in edges) for p in ['5','6','8','30']})
make_pool('same-image',[(i,i) for i in range(472)])
make_pool('knn',[(i,int(j)) for i in range(472) for j in neighbors[i,:7]])
for offset in [1,2,3,5]:
 edges=[]
 for ids in groups.values():
  for j,i in enumerate(ids):
   for sign in [-1,1]:
    if 0<=j+sign*offset<len(ids):edges.append((i,ids[j+sign*offset]))
 make_pool(f'offset{offset}',edges,True)
dump('plan.json',dict(seed=SEED,labels_read=False,rates=[.05,.1,.2],pools=pools,offset_info=edge_info,
 validation_global_indices=VAL.tolist(),patient_counts={p:sum(RR[i]['patient']==p for i in VAL) for p in ['5','6','8','30']},
 alignment='same normalized (x,y) coordinate in per-slice saved square body crop; no registration; physical-position order within patient/series',
 knn='sealed 448 recipe: L2(mean normalized patch tokens), cosine, stable sort, exclude self, top 7 of 30; diagnostic-only indices',
 sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'experiments/stego_biomedclip/reference_loss.py',ROOT/'audit/scratch/pairs/purity.py',ROOT/'audit/scratch/selectivity/affinities.py']},seconds=time.monotonic()-START))
ledger('pair_plan_access.json');print('PLAN COMPLETE',flush=True)
