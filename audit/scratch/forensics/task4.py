from common import *
from torch.utils.data._utils.collate import default_collate
from unlabeled_training import ImagesOnlyDataset
import random
inv=inventory();results={};pairs0=None;model=None
class TraceImages(ImagesOnlyDataset):
 def __init__(self,ds):super().__init__(ds);self.indices=[]
 def __getitem__(self,index):self.indices.append(int(index));return super().__getitem__(index)
def train_label_guard(event,args):
 if event=='open' and not isinstance(args[0],int) and '/preprocessed_v2/train/labels/' in str(Path(os.fsdecode(args[0])).resolve()):raise PermissionError('No train annotations in objective diagnostic')
sys.addaudithook(train_label_guard)
for name in ['best600','final']:
 if not inv[name]['exists']:continue
 model,ck=load(inv[name]['path'],model);cfg=model.cfg;ds=dataset(cfg,'train',True);ds.dataset=TraceImages(ds.dataset)
 order=np.random.default_rng(SEED).permutation(len(ds))[:256].reshape(16,16);np.random.seed(SEED);random.seed(SEED);torch.manual_seed(SEED)
 batchrows=[];lossfn=model.contrastive_corr_loss_fn;original=lossfn.helper
 with torch.inference_mode():
  for j,indices in enumerate(order):
   batch=default_collate([ds[int(i)] for i in indices]);f,c=model.net(batch['img']);fp,cp=model.net(batch['img_pos'])
   captured=[]
   def capture(f1,f2,c1,c2,shift):
    loss,cd=original(f1,f2,c1,c2,shift)
    affinity=-original(f1,f2,torch.ones_like(c1[:,:1]),torch.ones_like(c2[:,:1]),0)[0]
    captured.append(dict(raw_affinity_weighted_similarity=float((affinity*cd).mean()),clamped_affinity_weighted_similarity=float((affinity*cd.clamp_min(0)).mean()),mean_code_similarity=float(cd.mean()),mean_affinity=float(affinity.mean()),mean_shifted_weighted_similarity=float(((affinity-shift)*cd.clamp_min(0)).mean())))
    return loss,cd
   lossfn.helper=capture;torch.manual_seed(SEED+j)
   terms=lossfn(f,fp,None,None,c,cp,f,c);lossfn.helper=original
   intra=float(terms[0]);inter=float(terms[2]);neg=float(terms[4].mean());weighted=cfg.correspondence_weight*(cfg.pos_intra_weight*intra+cfg.pos_inter_weight*inter+cfg.neg_inter_weight*neg)
   assert len(captured)==18
   pairmeans={label:{k:float(np.mean([v[k] for v in group])) for k in group[0]} for label,group in [('knn',[captured[1]]),('random',captured[2:])]}
   batchrows.append(dict(batch=j,anchor_indices=indices.tolist(),loss_seed=SEED+j,pos_intra=intra,pos_inter=inter,neg_inter=neg,weighted_correspondence=weighted,pairs=pairmeans))
   print(name,j,'correspondence',weighted,'elapsed',elapsed(),flush=True)
 pairs=np.asarray(ds.dataset.indices).reshape(16,16,2).tolist()
 if pairs0 is None:pairs0=pairs
 else:assert pairs==pairs0,'Positive pairs drifted'
 result=dict(path=inv[name]['path'],batches=batchrows,anchor_positive_indices=pairs,summary={k:float(np.mean([b[k] for b in batchrows])) for k in ['pos_intra','pos_inter','neg_inter','weighted_correspondence']},pair_means={p:{k:float(np.mean([b['pairs'][p][k] for b in batchrows])) for k in batchrows[0]['pairs'][p]} for p in ['knn','random']},elapsed_seconds=elapsed())
 results[name]=result;dump('task4.json',dict(script=__file__,seed=SEED,protocol='16 fixed batches of16 train images; same top7 verified-cache kNN draws; real correspondence forward/helper with restored global mean; dropout off/eval; all16 negatives use nonzero batch shifts; no training masks opened; raw mean(fd*cd) and clamped mean(fd*max(cd,0)) reported.',results=results));del ck
note('Task4 completed: actual correspondence forward on16 identical train batches for best600/final with dropout off, fixed patch/negative seeds and identical positive pairs. Producer`task4.py`; details`task4.json`.');print('DONE',elapsed())
