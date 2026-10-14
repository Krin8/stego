"""Actual-loss per-batch calibration, no labels or parameter training."""
from common import *
SEED=20261003
SAMPLE=np.array(read(ROOT/'audit/preflight/sample.json')['local_indices']);plan=read(ROOT/'audit/preflight/pair_plan.json')
pos=np.array(plan['positive_local_indices']);centers=torch.from_numpy(np.load(ROOT/'audit/final_pipeline/inductive_seed0_centers.npy'))
# Reuse the preflight's actual candidate code construction, including random draw order.
tree=ast.parse((ROOT/'audit/preflight/diagnose.py').read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='codes')
exec(compile(ast.Module(body=[node],type_ignores=[]),'preflight codes','exec'),globals())
fixed=read(ROOT/'audit/calibrated/objective.json')['groups']['D448']['shifts'];rows=[]
class Capture(ContrastiveCorrelationLoss):
 def helper(self,f1,f2,c1,c2,shift):
  one1=torch.ones_like(c1[:,:1]);one2=torch.ones_like(c2[:,:1]);loss,_=super().helper(f1,f2,one1,one2,0.)
  self.aff[min(self.calls,2)].append((-loss).numpy().ravel());self.calls+=1
  return super().helper(f1,f2,c1,c2,shift)
for start in range(0,200,16):
 ids=SAMPLE[start:start+16];f=feature_batch(TRAIN[ids]);fp=feature_batch(TRAIN[pos[start:start+16]])
 torch.manual_seed(SEED+start+10);pairs={k:(codes(f,k),codes(fp,k)) for k in ['constant','random','cluster']}
 crit=Capture(SimpleNamespace(**CFG));crit.aff=[[],[],[]];crit.calls=0;torch.manual_seed(SEED+start)
 objective(crit,crit.cfg,f,fp,*pairs['constant']);aa=[np.concatenate(v) for v in crit.aff]
 shifts=[float(np.quantile(a,q)) for a,q in zip(aa,[.8,.8,.9])]
 result={}
 for name,ss in [('adaptive',shifts),('fixed',fixed)]:
  cfg=SimpleNamespace(**{**CFG,**dict(zip(['pos_intra_shift','pos_inter_shift','neg_inter_shift'],ss))});fn=ContrastiveCorrelationLoss(cfg);losses={}
  for k,(z,zp) in pairs.items():
   torch.manual_seed(SEED+start);loss,parts=objective(fn,cfg,f,fp,z,zp)
   losses[k]=dict(total=float(loss),**{k:float(v) for k,v in parts.items()})
  result[name]=dict(shifts=ss,attraction=[float((a>s).mean()) for a,s in zip(aa,ss)],losses=losses,
   admissible=losses['constant']['total']>losses['random']['total'] and losses['cluster']['total']<losses['random']['total'])
 rows.append(dict(batch=len(rows)+1,slices=len(ids),sample_start=start,**result));print('BATCH',len(rows),flush=True)
v=np.array([r['adaptive']['shifts'] for r in rows]);dump('batch_shifts.json',dict(seed=SEED,labels_used=False,batches=rows,fixed=fixed,
 mean=v.mean(0).tolist(),sd=v.std(0,ddof=1).tolist(),minimum=v.min(0).tolist(),maximum=v.max(0).tolist(),
 passes={k:sum(r[k]['admissible'] for r in rows) for k in ['adaptive','fixed']},seconds=time.monotonic()-START))
ledger('batch_access.json')
