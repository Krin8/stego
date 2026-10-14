"""PAIRS scoring unchanged; compare positives with shared randomness and calibration."""
from engine import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','train'],required=True);args=p.parse_args()
calibration_path=OUT/f'calibration_{args.split}.json';cal=json.loads(calibration_path.read_text());calibration_hash=hashlib.sha256(calibration_path.read_bytes()).hexdigest()
model,cfg,records,anchors,features,current,cross,k,pid,series,rank=initialize(args.split)
q=cal['calibration'];cfg.pos_intra_shift=q['intra']['shift'];cfg.pos_inter_shift=q['positive_current']['shift'];cfg.neg_inter_shift=q['negative']['shift']
needed=np.unique(np.concatenate([anchors,current[anchors].ravel(),cross[anchors].ravel()]));patch_labels=torch.zeros((len(records),1,14,14));mask_paths=[]
if args.split=='train':os.environ['ALLOW_TRAIN_LABEL_DIAGNOSTIC']='1'
ds=dataset(cfg,args.split)
assert [r['sample_id'] for r in ds.dataset.records]==[r['sample_id'] for r in records]
mixed=0;positive=0
for i in needed:
 y=ds.dataset[int(i)][1];assert set(torch.unique(y).tolist())<={0,1}
 occupancy=F.avg_pool2d(y[None,None].float(),16);patch_labels[i]=occupancy[0]>.5
 mixed+=int(((occupancy>0)&(occupancy<1)).sum());positive+=int((occupancy>.5).sum());mask_paths.append(ds.dataset.samples[int(i)][1])
os.environ.pop('ALLOW_TRAIN_LABEL_DIAGNOSTIC',None)
ctx={};counts={};image_draws={};cross_counts={};cross_draws={};prefix={}
lossfn=model.contrastive_corr_loss_fn;original_helper=lossfn.helper;original_sample=modules.sample
for name in ['relations','spy_sample','hard_labels','capture','label_summary']:reuse_pair_function(name,globals())
pairs_capture=capture
def capture(f1,f2,c1,c2,shift):
 global counts,image_draws
 call=ctx['helper_calls'];current_result=pairs_capture(f1,f2,c1,c2,shift)
 if call==1:
  current_counts,current_draws=counts,image_draws;original_positive=ctx['positives'];counts,image_draws=cross_counts,cross_draws
  b=ctx['cross_positive'];ctx['helper_calls']=1;ctx['positives']=b
  fcross=original_sample(features[b],ctx['coords2']);ccross=original_sample(image_id_codes(b),ctx['coords2'])
  pairs_capture(f1,fcross,c1,ccross,q['positive_cross']['shift'])
  counts,image_draws=current_counts,current_draws;ctx['positives']=original_positive
 return current_result
lossfn.helper=capture;modules.sample=spy_sample
def summarize(acc,draws):
 out={}
 for key,m in acc.items():
  allpairs=m.sum(0);a=m[1];total=int(allpairs.sum())
  out[key]=dict(image_pair_draws=draws[key],attraction_rate=int(a.sum())/total if total else None,random_unfiltered=label_summary(allpairs),attracted=label_summary(a),attraction_given_label_pair={label:int(a[i])/int(allpairs[i]) if allpairs[i] else None for i,label in enumerate(['BB','BL','LB','LL'])})
 return out
with torch.inference_mode():
 for nr,a,seed in batches(anchors,k):
  b=current[a,nr];ctx.clear();ctx.update(anchors=a,positives=b,cross_positive=cross[a,nr],helper_calls=0,feature_samples_seen=0)
  torch.manual_seed(seed);lossfn(features[a],features[b],None,None,image_id_codes(a),image_id_codes(b),features[a],image_id_codes(a))
  assert ctx['helper_calls']==2+min(int(cfg.neg_samples),len(a))
  if seed==SEED+(nr+1)*100+(len(anchors)-1)//16:
   print(args.split,'rank',nr+1,elapsed(),flush=True)
   if nr+1 in [7,29,30]:prefix[str(nr+1)]=dict(current=summarize(counts,image_draws),cross=summarize(cross_counts,cross_draws))
modules.sample=original_sample;lossfn.helper=original_helper
assert hashlib.sha256(calibration_path.read_bytes()).hexdigest()==calibration_hash
result=dict(script=__file__,split=args.split,k_nonself=k,anchor_count=len(anchors),anchor_patient_count=len(set(pid[anchors])),calibration_sha256=calibration_hash,shifts={key:r['shift'] for key,r in q.items()},current=summarize(counts,image_draws),cross=summarize(cross_counts,cross_draws),prefix_sensitivity=prefix,labels_used=('TRAIN masks strictly diagnostic, explicitly authorized Task3 exception' if args.split=='train' else 'Validation masks only'),mask_count=len(needed),anchor_mask_count=len(anchors),neighbour_only_mask_count=len(set(needed)-set(anchors)),mask_indices=needed.tolist(),mask_paths=mask_paths,mask_patient_count=len(set(pid[needed])),mixed_footprint_fraction=mixed/(len(needed)*196),liver_patch_fraction=positive/(len(needed)*196),protocol='Unmodified PAIRS capture/strata/patch-label sampling imported by AST; actual loss.forward/helper. Current/cross positive comparisons share anchor batch, coordinates and label rule. Cross positive receives its separate label-free shift; intra/negative statistics are identical shared comparisons, computed once. Calibration uses distinct coordinate seeds and no labels. Only required anchor/neighbour masks opened after calibration file finalized.',elapsed_seconds=elapsed())
dump(f'purity_{args.split}.json',result);note(f'{args.split} purity: `purity.py --split {args.split}` → `purity_{args.split}.json`, runtime {elapsed():.2f}s. {result["labels_used"]}; {len(needed)} masks for {len(anchors)} anchors plus neighbour targets.');print(json.dumps(dict(split=args.split,current=result['current']['knn/all'],cross=result['cross']['knn/all'],mask_count=len(needed),elapsed_seconds=elapsed()),indent=2))
