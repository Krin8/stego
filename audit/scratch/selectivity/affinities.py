"""Label-free paired scores from actual loss helpers; no cloned affinity formula."""
from common import *
import modules
from omegaconf import OmegaConf
from numpy.lib.format import open_memmap
def no_labels(event,args):
 if event=='open' and not isinstance(args[0],int) and ('/labels/' in str(args[0]) or str(args[0]).endswith('labels.npy')):raise PermissionError('No annotations during score or threshold construction')
sys.addaudithook(no_labels)
# Reuse original PAIRS/CROSSPATIENT definitions, with only their global paths rebound.
for path,names in [(ROOT/'audit/scratch/crosspatient/engine.py',['batches','image_id_codes','reuse_pair_function'])]:
 tree=ast.parse(path.read_text())
 for name in names:
  node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),globals())
inv=json.loads((FORE/'inventory.json').read_text());model,ck=load(inv['seed0']['path']);cfg=model.cfg
assert cfg.pointwise and not cfg.stabalize and not cfg.use_salience and cfg.feature_samples==11
features=torch.from_numpy(np.array(np.load(FORE/'val_features.npy',mmap_mode='r')))
other={f'block{b}':torch.from_numpy(np.array(np.load(OUT/f'block{b}.npy',mmap_mode='r'))) for b in [9,11]}
mean=torch.from_numpy(np.load(BASE/'train_mean.npy'))[None,:,None,None]
other['mean']=F.normalize(features-mean,dim=1,eps=1e-10)
lossfn=model.contrastive_corr_loss_fn;original_helper=lossfn.helper;original_sample=modules.sample
uncfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True));uncfg.pointwise=False
uncentered=modules.ContrastiveCorrelationLoss(uncfg)
ctx={};reuse_pair_function('spy_sample',globals());reuse_pair_function('hard_labels',globals())
# Use PAIRS' nearest sampler on token indices, not annotation data.
patch_labels=torch.arange(196,dtype=torch.float32).reshape(1,1,14,14).expand(472,-1,-1,-1)
clusters=np.load(OUT/'kmeans_tokens.npy').reshape(472,196)
positions=np.random.default_rng(SEED+77).choice(121*121,2048,replace=False)
np.save(OUT/'pair_positions.npy',positions)
pools=['intra','knn','random'];variants=['baseline','uncentered','mean','mean_loss','block9','block11','kmeans8'];rows=472*7
arrays={v:{p:open_memmap(OUT/f'scores_{v}_{p}.npy',mode='w+',dtype='float32',shape=(rows,len(positions))) for p in pools} for v in variants}
plan={p:{name:open_memmap(OUT/f'{p}_{name}.npy',mode='w+',dtype=dtype,shape=shape) for name,dtype,shape in [('a','int16',(rows,)),('b','int16',(rows,)),('ta','uint8',(rows,len(positions))),('tb','uint8',(rows,len(positions)))]} for p in pools}
counts={p:0 for p in pools};neighbours=np.load(OUT/'neighbors.npy')
def record_affinity(v,p,loss,cd,shift,start,n):
 fd=(float(shift)-loss/cd).numpy().reshape(n,-1)
 assert np.isfinite(fd).all() and float((cd-1).abs().max())<1e-6
 arrays[v][p][start:start+n]=fd[:,positions]
def capture(f1,f2,c1,c2,shift):
 loss,cd=original_helper(f1,f2,c1,c2,shift);call=ctx['helper_calls'];ctx['helper_calls']+=1
 if call>2:return loss,cd
 pool=pools[call];a=np.rint(c1[:,0,0,0].numpy()).astype(int)-1;b=np.rint(c2[:,0,0,0].numpy()).astype(int)-1
 assert np.array_equal(a,ctx['anchors'])
 if pool=='knn':assert np.array_equal(b,ctx['positives'])
 if pool=='random':assert np.all(a!=b)
 start=counts[pool];n=len(a);counts[pool]+=n
 coords_a=ctx['coords1'];coords_b=coords_a if pool=='intra' else ctx['coords2']
 ta=hard_labels(a,coords_a)[:,positions//121];tb=hard_labels(b,coords_b)[:,positions%121]
 for name,value in [('a',a),('b',b),('ta',ta),('tb',tb)]:plan[pool][name][start:start+n]=value
 record_affinity('baseline',pool,loss,cd,shift,start,n)
 l,d=uncentered.helper(f1,f2,c1,c2,0);record_affinity('uncentered',pool,l,d,0,start,n)
 for name,grid in other.items():
  fa=original_sample(grid[a],coords_a);fb=fa if pool=='intra' else original_sample(grid[b],coords_b)
  helper=uncentered.helper if name=='mean' else original_helper
  l,d=helper(fa,fb,c1,c2,0);record_affinity(name,pool,l,d,0,start,n)
  if name=='mean':
   l,d=original_helper(fa,fb,c1,c2,0);record_affinity('mean_loss',pool,l,d,0,start,n)
 arrays['kmeans8'][pool][start:start+n]=(clusters[a[:,None],ta]==clusters[b[:,None],tb]).astype(np.float32)
 return loss,cd
modules.sample=spy_sample;lossfn.helper=capture
with torch.inference_mode():
 for nr,a,seed in batches(np.arange(472),7):
  b=neighbours[a,nr];ctx.clear();ctx.update(helper_calls=0,feature_samples_seen=0,anchors=a,positives=b)
  torch.manual_seed(seed);lossfn(features[a],features[b],None,None,image_id_codes(a),image_id_codes(b),features[a],image_id_codes(a))
  assert ctx['helper_calls']==2+min(int(cfg.neg_samples),len(a))
  if seed==SEED+(nr+1)*100+29:print('rank',nr+1,'seconds',elapsed(),flush=True)
modules.sample=original_sample;lossfn.helper=original_helper
assert all(n==rows for n in counts.values())
for group in list(arrays.values())+list(plan.values()):
 for arr in group.values():arr.flush()
budgets=[.02,.05,.10,.20,.30,.50,.75];thresholds={}
for v in variants:
 thresholds[v]={}
 for p in pools:
  x=np.asarray(arrays[v][p]).reshape(-1)
  cuts=[.5] if v=='kmeans8' else np.quantile(x,1-np.array(budgets)).tolist()
  thresholds[v][p]=[dict(budget=None if v=='kmeans8' else budgets[i],threshold=float(t),attraction=float(np.mean(x>t))) for i,t in enumerate(cuts)]
baseline_shifts=dict(zip(pools,[float(cfg.pos_intra_shift),float(cfg.pos_inter_shift),float(cfg.neg_inter_shift)]))
result=dict(script=__file__,seed=SEED,labels_used=False,rows_per_pool=rows,pairs_per_image_edge=len(positions),pairs_per_pool=rows*len(positions),all_val_slices=472,neighbors=7,variants=variants,pools=pools,thresholds=thresholds,actual_shifts=baseline_shifts,baseline_attraction_at_C_shift={p:float(np.mean(arrays['baseline'][p]>t)) for p,t in baseline_shifts.items()},intra_coordinate_diagonal_sample_fraction=float(np.mean(positions//121==positions%121)),elapsed_seconds=elapsed())
dump('affinities.json',result);note('Label-free scores/thresholds completed: `audit/scratch/selectivity/affinities.py` → `affinities.json`, paired score arrays and token-index plans. Thresholds fixed before labels are opened.');print(json.dumps({k:v for k,v in result.items() if k!='thresholds'},indent=2))
