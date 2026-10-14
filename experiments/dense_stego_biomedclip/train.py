"""Label-free training. Requires an approved, hash-locked config; never inspects val."""
import os
os.environ['DENSE_STAGE']='training'
from runtime import *
from model import CorrespondenceModel
import correspondence as corr
import argparse
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['A','B'],required=True);p.add_argument('--seed',type=int,choices=[0,1,2],required=True);a=p.parse_args()
c=read(HERE/'configs/locked.json');assert c['status']=='LOCKED';assert sha(HERE/'configs/locked.json')==(HERE/'configs/locked.sha256').read_text().strip();assert not (HERE/'predictions/stageA_manifest.json').exists()
seed_all(a.seed);rr=records();tr=[r for r in rr if r['split']=='train'];patients=sorted({r['patient'] for r in tr},key=int);by={pat:[r for r in tr if r['patient']==pat] for pat in patients};directory=HERE/f'checkpoints/{a.arm}_seed{a.seed}';directory.mkdir(exist_ok=False)
net=frozen_backbone();m=CorrespondenceModel(a.arm,c['ct_branch'] and a.arm=='B');opt=m.optimizer(c['lr']);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=c['scheduler_T_max']);assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in m.parameters()};assert not {id(p) for p in net.parameters()}&{id(p) for g in opt.param_groups for p in g['params']}
params=dict(backbone_trainable=sum(p.numel() for p in net.parameters() if p.requires_grad),new_trainable=sum(p.numel() for p in m.parameters() if p.requires_grad));assert params['backbone_trainable']==0 and params['new_trainable']>0
rng=np.random.default_rng(a.seed);g=torch.Generator().manual_seed(a.seed);monitor_rng=np.random.default_rng(c['monitor_seed']);monitor=[by[pat][monitor_rng.integers(len(by[pat]))] for pat in patients];fixed=[]
def image_batch(rs):
 xs=[]
 for r in rs:
  raw=torch.from_numpy(crop_image(load_image(r),r['box']))[None,None];xs.append(F.interpolate(raw,(448,448),mode='bilinear',align_corners=False,antialias=True)[0])
 return torch.stack(xs)
mg=torch.Generator().manual_seed(c['monitor_seed'])
for j in range(0,len(monitor),c['batch_size']):
 raw=image_batch(monitor[j:j+c['batch_size']]);va,ia,aa=corr.views(raw,mg);vb,ib,ab=corr.views(raw,mg);fixed.append((tokens(net,va),tokens(net,vb),ia,ib,aa,ab))
log=HERE/f'logs/{a.arm}_seed{a.seed}_train.jsonl';best=float('inf');beststep=None;t0=time.monotonic();source_hash={str(x.relative_to(HERE)):sha(x) for x in HERE.glob('*.py')};history=[]
for step in range(1,c['batches']+1):
 # Balanced patient cycle; all16 patients participate regardless of500-batch sampling.
 rs=[by[patients[((step-1)*c['batch_size']+j)%len(patients)]][rng.integers(len(by[patients[((step-1)*c['batch_size']+j)%len(patients)]]))] for j in range(c['batch_size'])]
 raw=image_batch(rs);va,ia,aa=corr.views(raw,g);vb,ib,ab=corr.views(raw,g);fa,fb=tokens(net,va),tokens(net,vb);m.train();za,zb=m(fa,ia),m(fb,ib)
 loss,details=corr.loss(za,zb,aa,ab,g,c['temperature'],c['correspondences'],c['negative_distance']);assert torch.isfinite(loss)
 opt.zero_grad(set_to_none=True);loss.backward();assert all(x.grad is None for x in net.parameters());opt.step();scheduler.step();row=dict(step=step,train_loss=float(loss.detach()),lr=scheduler.get_last_lr()[0],seconds=time.monotonic()-t0,**details)
 if step%c['monitor_every']==0 or step==c['batches']:
  m.eval();values=[];stds=[];cosines=[];mg=torch.Generator().manual_seed(c['monitor_seed']+1)
  with torch.no_grad():
   for fa,fb,ia,ib,aa,ab in fixed:
    za,zb=m(fa,ia),m(fb,ib);v,_=corr.loss(za,zb,aa,ab,mg,c['temperature'],c['correspondences'],c['negative_distance']);values.append(float(v));stds.append(float(za.flatten(2).std(2).mean()));pooled=F.normalize(za.mean((2,3)),dim=1);cosines.append(float((pooled[0]*pooled[1]).sum()))
  criterion=float(np.mean(values));row.update(monitor_loss=criterion,code_std=float(np.mean(stds)),cross_image_cosine=float(np.mean(cosines)))
  state=dict(state_dict=m.state_dict(),arm=a.arm,seed=a.seed,ct_branch=m.ct_branch,step=step,criterion=criterion,config=c,config_sha256=sha(HERE/'configs/locked.json'),source_sha256=source_hash,backbone_sha256=sha(ROOT/'src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin'),train_patient_ids=patients,monitor_sample_ids=[r['sample_id'] for r in monitor],parameter_counts=params)
  if criterion<best:best=criterion;beststep=step;torch.save(state,directory/'best.pt')
  torch.save(state,directory/'last.pt');print(json.dumps(row),flush=True)
 with log.open('a') as f:f.write(json.dumps(row)+'\n')
 history.append(row)
dump(f'logs/{a.arm}_seed{a.seed}_training.json',dict(arm=a.arm,seed=a.seed,batches=c['batches'],best_step=beststep,best_loss=best,checkpoint_selection=c['checkpoint_selection'],parameters=params,train_patients=patients,validation_images_used=0,mask_files_read=0,history=history));ledger(f'logs/{a.arm}_seed{a.seed}_training_access.json');baseline_seal()
