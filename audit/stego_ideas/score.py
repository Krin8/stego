"""Validation-label scoring only after the label-free pair plan is complete."""
from common import *
import common
from PIL import Image
plan=read(OUT/'plan.json');assert not plan['labels_read'];assert plan['validation_global_indices']==VAL.tolist()
common.ALLOW_LABELS=True
reuse(ROOT/'audit/scratch/pairs/purity.py','label_summary',globals())
reuse(ROOT/'audit/scratch/selectivity/score.py','basic',globals())
reuse(ROOT/'audit/improve/pipeline.py','crop_image',globals())
patches=[];paths=[]
for i in VAL:
 r=RR[i];p=Path(r['image']).parent.parent/'labels'/Path(r['image']).name
 assert r['split']=='val' and '/val/labels/' in str(p)
 with Image.open(p) as im:y=(np.array(im)>0).astype(np.float32)
 cropped=crop_image(y,r['box']);t=F.interpolate(torch.from_numpy(cropped)[None,None],(448,448),mode='nearest')
 # PAIRS' exact majority-patch convention, now in 448 body-crop coordinates.
 occupancy=F.avg_pool2d(t,16);patches.append((occupancy>.5).numpy().reshape(784).astype(np.uint8));paths.append(str(p))
patches=np.stack(patches)
def summaries(category,selected):
 c=np.bincount(category[selected],minlength=4);allc=np.bincount(category,minlength=4)
 result=label_summary(c);result['liver_target_share_all']=float((c[1]+c[3])/c.sum()) if c.sum() else None
 liver_c=c.copy();liver_c[:2]=0;result['liver_anchors']=label_summary(liver_c)
 result['attraction']=float(np.mean(selected));result['attraction_liver_anchors']=float(liver_c.sum()/max(1,allc[2]+allc[3]))
 return result
rows=[];baselines=[]
for name,meta in plan['pools'].items():
 a=np.load(OUT/f'{name}_a.npy');b=np.load(OUT/f'{name}_b.npy');ta=np.load(OUT/f'{name}_ta.npy');tb=np.load(OUT/f'{name}_tb.npy')
 cat=(2*patches[a[:,None],ta]+patches[b[:,None],tb]).ravel();x=np.load(OUT/f'{name}_affinity.npy',mmap_mode='r').reshape(-1)
 base=summaries(cat,np.ones(len(cat),bool));baselines.append(dict(pool=name,**base))
 if name.endswith('_random'):continue
 for rate,threshold in zip(plan['rates'],meta['thresholds']):
  selected=x>threshold;r=summaries(cat,selected)
  # Independent existing SELECTIVITY scorer must agree with existing PAIRS summary.
  check=basic(cat,x,threshold)
  assert check['same_label']==r['same_label'] and check['liver_target_precision']==r['target_liver_given_anchor_liver']
  rows.append(dict(pool=name,source='affinity' if name in ['same-image','knn'] else 'colocated+affinity',rate=rate,threshold=threshold,**r))
  if name.startswith('offset'):
   # Fixed-seed independent thinning is the only label-free way to budget unranked co-location.
   rng=np.random.default_rng(plan['seed']+int(name[6:]));chosen=np.zeros(len(cat),bool);chosen[rng.permutation(len(cat))[:round(rate*len(cat))]]=True
   rows.append(dict(pool=name,source='colocated-only',rate=rate,threshold=None,**summaries(cat,chosen)))
 print('SCORED',name,flush=True)
dump('purity.json',dict(script=__file__,labels_used='472 validation PNG masks only; scoring, patch majority labels, purity/base rates. No training labels; pair construction and thresholds sealed before scoring.',
 patch_liver_fraction=float(patches.mean()),rows=rows,baselines=baselines,label_paths=paths,plan_sha256=sha(OUT/'plan.json'),seconds=time.monotonic()-START))
ledger('score_access.json')
