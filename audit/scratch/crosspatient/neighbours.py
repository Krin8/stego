"""Reuse project image pooling and exact sorting AST; no pipeline cache writes."""
from common import *
from collections import Counter
from knn_cache import expected_identity,cache_path
from tqdm import tqdm
source=ast.parse((ROOT/'src/precompute_knns.py').read_text())
node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='get_feats')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/'src/precompute_knns.py'),'exec'),globals())
app=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='my_app')
rank_loop=next(n for n in ast.walk(app) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='i')
prefix=rank_loop.body[:4];sortnode=rank_loop.body[4]
assert ast.unparse(prefix[0]).startswith('batch_feats =') and ast.unparse(sortnode).startswith('neighbors =')
prefix_code=compile(ast.Module(body=prefix,type_ignores=[]),str(ROOT/'src/precompute_knns.py'),'exec');sort_code=compile(ast.Module(body=[sortnode],type_ignores=[]),str(ROOT/'src/precompute_knns.py'),'exec')
def ranked(normed_feats,pids,cross_only=False):
 result=[];device=torch.device('cpu');identity={'k':31};step=max(1,len(normed_feats)//16)
 for i in range(0,len(normed_feats),step):
  scope=dict(normed_feats=normed_feats,device=device,identity=identity,step=step,i=i,torch=torch)
  exec(prefix_code,scope)
  if cross_only:
   same=pids[i:i+len(scope['batch_feats']),None]==pids[None,:]
   scope['pairwise_sims'][torch.from_numpy(same)]=-torch.inf
  exec(sort_code,scope);result.append(scope['neighbors'])
 return torch.cat(result).numpy()
inv=json.loads((FORE/'inventory.json').read_text());model,ck=load(inv['seed0']['path']);results={}
class ActualFeatureWrapper(torch.nn.Module):
 def __init__(self,net,reference,split):super().__init__();self.net=net;self.reference=reference;self.split=split;self.offset=0;self.error=0.
 def forward(self,img):
  f,c=self.net(img);expected=torch.from_numpy(np.array(self.reference[self.offset:self.offset+len(img)]))
  if self.split=='train':expected=expected.reshape(-1,14,14,768).permute(0,3,1,2)
  self.error=max(self.error,float((f-expected).abs().max()));self.offset+=len(img);return f
for split in ['val','train']:
 ds=dataset(model.cfg,split,True);records=ds.dataset.records
 assert [r['sample_id'] for r in records]==json.loads((BASE/'samples.json').read_text())[split+'_ids']
 identity=expected_identity(ds.dataset,model.cfg);path=cache_path(model.cfg.pytorch_data_dir,identity);cached=ds.nns.numpy();assert cached.shape==(len(ds),30)
 reference=np.load(FORE/'val_features.npy' if split=='val' else BASE/'train_features.npy',mmap_mode='r')
 ds.dataset=ImagesOnlyDataset(ds.dataset);wrapper=ActualFeatureWrapper(model.net,reference,split).eval()
 # Match the actual cache writer's feature-extraction batch size.
 with torch.inference_mode():pooled=get_feats(wrapper,DataLoader(ds,batch_size=256,shuffle=False,num_workers=0),torch.device('cpu'))
 assert wrapper.offset==len(ds)
 pids=np.array([r['patient_id'] for r in records]);current=ranked(pooled,pids);cross=ranked(pooled,pids,True)
 cache_equal=bool(np.array_equal(current[:,:29],cached[:,1:]))
 disagreements=np.any(current[:,:29]!=cached[:,1:],axis=1)
 overlap=[len(set(a)&set(b))/29 for a,b in zip(current[:,:29],cached[:,1:])]
 rank_comparison=dict(exact=cache_equal,rows_with_any_rank_difference=int(disagreements.sum()),different_rank_entries=int((current[:,:29]!=cached[:,1:]).sum()),mean_set_overlap=float(np.mean(overlap)),min_set_overlap=float(np.min(overlap)))
 assert np.all(pids[cross]!=pids[:,None]) and np.all(current!=np.arange(len(ds))[:,None])
 anchors=np.arange(len(ds)) if split=='val' else np.random.default_rng(SEED).choice(len(ds),200,replace=False)
 assert split=='val' or len(set(pids[anchors]))>=12
 np.save(OUT/f'{split}_current30.npy',current);np.save(OUT/f'{split}_cross30.npy',cross);np.save(OUT/f'{split}_cached29.npy',cached[:,1:]);np.save(OUT/f'{split}_image_features.npy',pooled.numpy())
 dump(f'{split}_records.json',records);dump(f'{split}_anchors.json',anchors.tolist())
 series=np.array([r['series_uid'] for r in records]);rank=np.empty(len(ds),int);relative=np.empty(len(ds),float)
 for patient,serie in set(zip(pids,series)):
  indices=[i for i in range(len(ds)) if (pids[i],series[i])==(patient,serie)];indices.sort(key=lambda i:records[i]['physical_position']);rank[indices]=np.arange(len(indices));relative[indices]=np.arange(len(indices))/max(1,len(indices)-1)
 def distribution(v):
  v=np.asarray(v);return None if not len(v) else dict(n=len(v),mean=float(v.mean()),quantiles={str(q):float(np.percentile(v,q)) for q in [0,5,25,50,75,95,100]})
 subsets={}
 for k in [7,29,30]:
  for name,nn in [('current',current),('cross',cross)]:
   a=np.repeat(anchors,k);b=nn[anchors,:k].ravel();same=pids[a]==pids[b];same_series=same&(series[a]==series[b]);sims=(pooled[a]*pooled[b]).sum(1).numpy()
   subsets[f'{name}_k{k}']=dict(edges=len(a),same_patient=float(same.mean()),same_slice=float((a==b).mean()),adjacent=float((same_series&(np.abs(rank[a]-rank[b])==1)).mean()),within3=float((same_series&(np.abs(rank[a]-rank[b])<=3)).mean()),same_series_native_rank_difference=distribution(np.abs(rank[a[same_series]]-rank[b[same_series]])),normalized_stack_position_difference_all=distribution(np.abs(relative[a]-relative[b])),normalized_stack_position_difference_cross_patient=distribution(np.abs(relative[a[~same]]-relative[b[~same]])),image_cosine_similarity=distribution(sims))
 results[split]=dict(script=__file__,split=split,total_slices=len(ds),anchor_count=len(anchors),anchor_patient_counts=dict(Counter(pids[anchors])),candidate_patient_count=len(set(pids)),cache_path=str(path),cache_rank_comparison=rank_comparison,max_live_feature_error_vs_batch16=wrapper.error,feature_extraction_batch_size=256,neighbour_sets=subsets,labels_used=False,elapsed_seconds=elapsed())
 dump('neighbours.json',results);note(f'{split} image-neighbour construction: `neighbours.py` → `neighbours.json`; rank comparison {rank_comparison}. No labels used. Runtime {elapsed():.2f}s.');print(split,results[split]['anchor_patient_counts'],rank_comparison,elapsed(),flush=True)
