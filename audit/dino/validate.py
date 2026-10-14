"""Check artifact dimensions, cohort isolation, original scoring, and complete declared coverage."""
from common import *
from scipy.optimize import linear_sum_assignment
import subprocess
ex=json.loads((OUT/'D1_extract.json').read_text());assert ex['missing_keys']==ex['unexpected_keys']==[] and ex['deterministic_exact'] and ex['all_requires_grad_false'] and ex['model_eval']
f=np.load(OUT/'D1_features.npy',mmap_mode='r');normerr=0.
for start in range(0,2341,32):
 x=np.asarray(f[start:start+32]);assert np.isfinite(x).all();normerr=max(normerr,float(np.abs(np.linalg.norm(x,axis=-1)-1).max()))
assert normerr<1e-5
for k in [8,2]:
 for s in [0,1,2]:
  r=json.loads((OUT/f'D1_k{k}_s{s}_scores.json').read_text());g=np.load(OUT/f'D1_k{k}_s{s}_grids.npy',mmap_mode='r');assert g.shape==(2341,28,28) and g.min()>=0 and g.max()<k
  for res in ['224','512']:
   folds=r['results'][res]['folds'];assert len(folds)==20
   for patient,fold in folds.items():
    assert patient not in fold['mapping_patients'] and len(fold['mapping_patients'])==19
    for rule in (['majority','optimal'] if k==8 else ['Hungarian']):assert all(c<k for c in fold[rule]['clusters'])
# Forced-background outside-crop pixels stay false negatives. Hungarian swaps IDs correctly.
c=np.array([[0,10],[80,0],[0,10]],np.int64);row,col=linear_sum_assignment(-c[:2]);r=scores(c,row[col==1]);assert r['iou']==.5 and r['clusters']==[0]
sel=json.loads((OUT/'pair_selectivity.json').read_text())
for name,r in sel['results'].items():
 assert .0<=r['auc']<=1 and r['pairs']==472*7*2048
 assert [s['budget'] for s in r['sweep']]==[.05,.1,.2]
 assert all(abs(s['attraction']-s['budget'])<.001 for s in r['sweep'])
source=ROOT/'src/modules.py'
state=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),git_diff=subprocess.check_output(['git','diff'],cwd=ROOT,text=True),git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
assert state['git_diff']==''
dump('validation.json',dict(producer='validate.py',max_L2_error=normerr,all_declared_available_runs_complete=True,lopo_heldout_excluded=True,synthetic_hungarian_and_outside_crop_pass=True,score_source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [source,ROOT/'src/dino/vision_transformer.py',SELECT/'score.py',SELECT/'affinities.py']},state=state))
print('VALIDATED',normerr)
