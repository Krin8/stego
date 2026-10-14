"""FIXES3 protocol, only shifts differ; new scheduled checkpoints, no label fitting."""
from common import *
import argparse,json,time,shutil,hashlib,fcntl,subprocess
from omegaconf import OmegaConf
from pytorch_lightning import Trainer,Callback
import train_segmentation as training
from process_check import check
p=argparse.ArgumentParser();p.add_argument('--candidate',choices=['D','E'],required=True);p.add_argument('--seed',type=int,choices=[0,1,2],required=True);args=p.parse_args()
name=f'fixes4_{args.candidate}_seed{args.seed}';runroot=OUT/'experiments'/name
if runroot.exists():raise FileExistsError('New experiment required; preserving '+str(runroot))
lock=(OUT/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
check(OUT/f'{name}_process_check.json')
runroot.mkdir(parents=True)
def no_train_labels(event,a):
 if event=='open' and not isinstance(a[0],int) and '/preprocessed_v2/train/labels/' in str(Path(os.fsdecode(a[0])).resolve()):raise PermissionError('Training annotations forbidden')
sys.addaudithook(no_train_labels)
cfg=OmegaConf.load(ROOT/'src/configs/train_config.yaml')
cfg.output_root=str(runroot);cfg.pytorch_data_dir=str(ROOT/'src/pytorch_data_dir');cfg.seed=args.seed;cfg.experiment_name=name
cfg.max_steps=5000;cfg.extra_clusters=6;cfg.collapse_early_stop=True
cfg.num_workers=0;cfg.val_freq=50;cfg.batch_size=16
cal=json.loads((OUT/'calibration.json').read_text())
for key in ['pos_intra_shift','pos_inter_shift','neg_inter_shift']:setattr(cfg,key,cal['candidates'][args.candidate][key])
assert not cfg.train_linear_probe and not cfg.use_true_labels and not cfg.use_salience and not cfg.use_text_prompts
assert cfg.rec_weight==cfg.crf_weight==cfg.aug_alignment_weight==cfg.text_align_weight==0
assert (cfg.pos_intra_weight,cfg.pos_inter_weight,cfg.neg_inter_weight)==(.67,.25,.63)
reference=OmegaConf.load(ROOT/'audit/scratch/fixes3/experiments/fixes3_C_seed1/resolved_config.yaml')
allowed={'output_root','experiment_name','seed','pos_intra_shift','pos_inter_shift','neg_inter_shift'}
actual=OmegaConf.to_container(cfg,resolve=True);old=OmegaConf.to_container(reference,resolve=True)
differences={k:dict(old=old.get(k),new=actual.get(k)) for k in set(old)|set(actual) if old.get(k)!=actual.get(k)}
assert set(differences)<=allowed,differences
OmegaConf.save(cfg,runroot/'resolved_config.yaml')
(runroot/'provenance.json').write_text(json.dumps(dict(command=sys.argv,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),reference_config_differences=differences,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),calibration_sha256=hashlib.sha256((OUT/'calibration.json').read_bytes()).hexdigest()),indent=2)+'\n')
start=time.monotonic();captured={}
class BoundedRun(Callback):
 def __init__(self):self.batches=0;self.saved={}
 def on_train_start(self,trainer,model):
  assert trainer.lr_scheduler_configs[0].scheduler.T_max==5000
  self.linear_start={k:v.detach().cpu().clone() for k,v in model.linear_probe.state_dict().items()}
  assert all(not p.requires_grad for p in model.net.full_model.parameters())
 def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
  self.batches+=1;assert self.batches<=500
  assert trainer.global_step==3*self.batches
  if self.batches in (100,250,500):
   path=runroot/'checkpoints'/f'batch_{self.batches:06d}.ckpt';assert not path.exists()
   trainer.save_checkpoint(path);self.saved[str(self.batches)]=str(path)
   (runroot/'checkpoint_index.json').write_text(json.dumps(self.saved,indent=2)+'\n')
  if self.batches%25==0:print('PROGRESS',name,self.batches,'seconds',round(time.monotonic()-start,1),flush=True)
  if self.batches==500:trainer.should_stop=True
 def on_validation_end(self,trainer,model):
  if not trainer.sanity_checking and model.validation_history:
   r=model.validation_history[-1]
   print('VALIDATION',json.dumps(dict(run=name,batches=self.batches,code_std=r['code_std'],cosine=r['cross_image_cosine'],usage=r['cluster_usage'],entropy=r['cluster_entropy'],bad_checks=r['consecutive_bad_checks'],early_stop=r['early_stopped'])),flush=True)
cb=BoundedRun()
def bounded_trainer(**kw):
 assert kw['max_steps']==15000
 kw.update(max_steps=1500,accelerator='cpu',devices=1,callbacks=[cb],enable_checkpointing=False,logger=[kw['logger'][0]],enable_progress_bar=False,enable_model_summary=False)
 trainer=Trainer(**kw);captured['trainer']=trainer;return trainer
training.Trainer=bounded_trainer
print('PREDECLARED',name,'500 batch cap; scheduler5000; train labels forbidden',flush=True)
os.chdir(ROOT/'src');training.my_app.__wrapped__(cfg)
trainer=captured['trainer'];model=trainer.lightning_module
assert 0<cb.batches<=500 and trainer.global_step==3*cb.batches
assert all(torch.equal(cb.linear_start[k],v.detach().cpu()) for k,v in model.linear_probe.state_dict().items())
assert all(p.grad is None for p in model.net.full_model.parameters())
if model.validation_history[-1]['batches']!=cb.batches:trainer.validate(model,dataloaders=trainer.val_dataloaders,verbose=False)
last=model.validation_history[-1]
if cb.batches<500:
 path=runroot/'checkpoints'/f'early_stop_batch_{cb.batches:06d}.ckpt';assert not path.exists();trainer.save_checkpoint(path)
else:assert set(cb.saved)=={'100','250','500'}
overlay=runroot/f'fixed_slices_batch_{cb.batches:06d}.png';shutil.copyfile(last['fixed_overlay_path'],overlay)
summary=dict(candidate=args.candidate,seed=args.seed,name=name,batches=cb.batches,optimizer_steps=trainer.global_step,early_stopped=last['early_stopped'],monitors={k:last[k] for k in ['code_std','cross_image_cosine','cluster_usage','cluster_entropy','consecutive_bad_checks']},checkpoints=cb.saved,overlay=str(overlay),fixed_samples=last['fixed_samples'],evaluated_images=last['images'],linear_parameters_unchanged=True,train_mask_reads=0,shifts={k:float(getattr(cfg,k)) for k in ['pos_intra_shift','pos_inter_shift','neg_inter_shift']},scheduler_T_max=trainer.lr_scheduler_configs[0].scheduler.T_max,elapsed_seconds=time.monotonic()-start)
(runroot/'result.json').write_text(json.dumps(summary,indent=2)+'\n');record(f'Task2 {name}: {cb.batches} batches; early_stop={last["early_stopped"]}; {summary["elapsed_seconds"]:.2f}s; result `{runroot}/result.json`.')
print('RESULT',json.dumps(summary),flush=True)
