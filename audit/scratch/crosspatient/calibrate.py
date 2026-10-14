"""Label-free quantiles; labels are blocked throughout calibration."""
from engine import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','train'],required=True);args=p.parse_args()
def forbid_annotations(event,a):
 if event=='open' and not isinstance(a[0],int) and ('/labels/' in str(a[0]) or str(a[0]).endswith('labels.npy')):raise PermissionError('No annotations in shift calibration')
sys.addaudithook(forbid_annotations)
model,cfg,records,anchors,features,current,cross,k,pid,series,rank=initialize(args.split)
lossfn=model.contrastive_corr_loss_fn;original_helper=lossfn.helper;original_sample=modules.sample;ctx={};reuse_pair_function('spy_sample',globals());modules.sample=spy_sample
positions=np.random.default_rng(SEED+77).choice(121*121,512,replace=False)
values={key:[] for key in ['intra','positive_current','positive_cross','negative']};prefix={}
def capture(f1,f2,c1,c2,shift):
 loss,cd=original_helper(f1,f2,c1,c2,shift);call=ctx['helper_calls'];assert float((cd-1).abs().max())<1e-6
 if call<=2:
  fd=(float(shift)-loss/cd).numpy().reshape(len(f1),-1)
  key=['intra','positive_current','negative'][call];values[key].append(fd[:,positions].reshape(-1).copy())
  if call==1:
   b=ctx['cross_positive'];fcross=original_sample(features[b],ctx['coords2']);ccross=original_sample(image_id_codes(b),ctx['coords2'])
   other,other_cd=original_helper(f1,fcross,c1,ccross,0.0)
   values['positive_cross'].append((-other/other_cd).numpy().reshape(len(f1),-1)[:,positions].reshape(-1).copy())
 ctx['helper_calls']+=1;return loss,cd
lossfn.helper=capture
def quantiles():
 result={}
 for key,parts in values.items():
  v=np.concatenate(parts);q=.85 if key=='negative' else .25
  result[key]=dict(shift=float(np.quantile(v,q)),quantile=q,affinity_samples=len(v),sample_attraction=float(np.mean(v>np.quantile(v,q))))
 return result
with torch.inference_mode():
 for nr,a,seed in batches(anchors,k,100000):
  b=current[a,nr];ctx.clear();ctx.update(helper_calls=0,feature_samples_seen=0,cross_positive=cross[a,nr])
  torch.manual_seed(seed);lossfn(features[a],features[b],None,None,image_id_codes(a),image_id_codes(b),features[a],image_id_codes(a))
  if seed==SEED+100000+(nr+1)*100+(len(anchors)-1)//16:
   print(args.split,'rank',nr+1,elapsed(),flush=True)
   if nr+1 in [7,29,30]:prefix[str(nr+1)]=quantiles()
modules.sample=original_sample;lossfn.helper=original_helper
result=dict(script=__file__,split=args.split,k_nonself=k,anchor_count=len(anchors),anchor_patients=len(set(pid[anchors])),labels_used=False,sampling='Actual PAIRS/project forward with common anchors/batches/coordinates/random rolls across variants. Retain512 fixed uniformly selected entries of each121x121 affinity matrix; negative calibration retains the first nonzero roll per batch. Positive cross-only uses actual sample/helper on the SAME coordinates. Calibration coordinate seeds offset100000 from purity evaluation.',calibration=quantiles(),prefix_calibration=prefix,original_C_shifts={key:float(getattr(cfg,key)) for key in ['pos_intra_shift','pos_inter_shift','neg_inter_shift']},rule='C heuristic, not Paper A: positive and intra25th percentile; common random-negative85th percentile. No labels. Exact quantiles of the retained Monte Carlo sample, not exhaustive affinity quantiles.',elapsed_seconds=elapsed())
dump(f'calibration_{args.split}.json',result);note(f'Label-free {args.split} calibration: `calibrate.py --split {args.split}` → `calibration_{args.split}.json`, runtime {elapsed():.2f}s. No annotation files opened.');print(json.dumps(result,indent=2))
