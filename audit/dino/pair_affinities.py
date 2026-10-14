"""Replay SELECTIVITY sampling with actual loss.forward/helper; scores fixed before labels."""
from common import *
os.environ['LABEL_FREE']='1'
import modules
from omegaconf import OmegaConf
PAIR=ROOT/'audit/scratch/pairs';reuse(ROOT/'audit/scratch/crosspatient/engine.py',['batches','image_id_codes','reuse_pair_function'],globals())
rr=records();vi=[i for i,r in enumerate(rr) if r['split']=='val'];assert len(vi)==472
original_records=json.loads((SELECT/'records.json').read_text());assert [r['sample_id'] for r in original_records]==[rr[i]['sample_id'] for i in vi]
sources={name:torch.from_numpy(np.array(np.load(path,mmap_mode='r')[vi])).permute(0,2,1).reshape(472,768,28,28) for name,path in [('D1',OUT/'D1_features.npy'),('BiomedCLIP_B2_448',OLD/'B2_448_features.npy')]}
cfg=OmegaConf.create(dict(pointwise=True,stabalize=False,zero_clamp=True,use_salience=False,feature_samples=11,neg_samples=20,pos_intra_shift=0.,pos_inter_shift=0.,neg_inter_shift=0.))
lossfn=modules.ContrastiveCorrelationLoss(cfg);original_helper=lossfn.helper;original_sample=modules.sample
ctx={};reuse_pair_function('spy_sample',globals());reuse_pair_function('hard_labels',globals())
# Token IDs, not annotations: support28x28 indices with uint16, avoiding uint8 overflow.
patch_labels=torch.arange(784,dtype=torch.float32).reshape(1,1,28,28).expand(472,-1,-1,-1)
pos=np.load(SELECT/'pair_positions.npy');assert len(pos)==2048
rows=472*7;arrays={name:np.lib.format.open_memmap(OUT/f'pair_scores_{name}.npy',mode='w+',dtype=np.float32,shape=(rows,2048)) for name in sources}
plan={name:np.lib.format.open_memmap(OUT/f'pair_{name}.npy',mode='w+',dtype=dtype,shape=shape) for name,dtype,shape in [('a','int16',(rows,)),('ta','uint16',(rows,2048)),('tb','uint16',(rows,2048))]}
count=0
def capture(f1,f2,c1,c2,shift):
 global count
 loss,cd=original_helper(f1,f2,c1,c2,shift);call=ctx['calls'];ctx['calls']+=1
 if call:return loss,cd
 a=ctx['anchors'];coords=ctx['coords1'];start=count;n=len(a);count+=n
 token=hard_labels(a,coords)
 plan['a'][start:start+n]=a;plan['ta'][start:start+n]=token[:,pos//121];plan['tb'][start:start+n]=token[:,pos%121]
 for name,features in sources.items():
  sample=original_sample(features[a],coords);l,d=original_helper(sample,sample,c1,c1,0)
  assert (d-1).abs().max()<1e-6
  arrays[name][start:start+n]=(-l/d).numpy().reshape(n,-1)[:,pos]
 return loss,cd
modules.sample=spy_sample;lossfn.helper=capture
with torch.inference_mode():
 for nr,a,seed in batches(np.arange(472),7):
  ctx.clear();ctx.update(feature_samples_seen=0,calls=0,anchors=a);torch.manual_seed(seed)
  features=sources['D1'][a];codes=image_id_codes(a)
  lossfn(features,features,None,None,codes,codes,features,codes)
  if seed==SEED+(nr+1)*100+29:print('PAIRS',nr+1,elapsed(),flush=True)
modules.sample=original_sample;lossfn.helper=original_helper
assert count==rows;assert np.array_equal(plan['a'],np.load(SELECT/'intra_a.npy'))
for v in list(arrays.values())+list(plan.values()):v.flush()
budgets=[.05,.1,.2];thresholds={name:[dict(budget=b,threshold=float(t),attraction=float(np.mean(v>t))) for b,t in zip(budgets,np.quantile(v,1-np.array(budgets)))] for name,v in arrays.items()}
dump('pair_affinities.json',dict(producer='pair_affinities.py',labels_used=False,rows=count,pairs_per_row=2048,pairs_per_source=count*2048,sources=list(sources),thresholds=thresholds,selectivity_anchor_order_exact=True,seconds=elapsed()))
print('PAIRS DONE',elapsed(),flush=True)
