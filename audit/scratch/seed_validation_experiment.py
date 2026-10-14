"""Actual project training entrypoint, CPU, independent hard cap500 batches, no train labels."""
from seed_validation_common import *
import argparse,json,time,shutil
from omegaconf import OmegaConf
from pytorch_lightning import Trainer,Callback
from pytorch_lightning.loggers import TensorBoardLogger
import train_segmentation as training
p=argparse.ArgumentParser();p.add_argument('--run',choices=['A','B','C'],required=True);p.add_argument('--name',required=True);p.add_argument('--seed',type=int,choices=[1,2],required=True);p.add_argument('data_override');args=p.parse_args()
assert args.data_override.startswith('pytorch_data_dir=')
runroot=OUT/'experiments'/args.name
if runroot.exists():raise FileExistsError('New experiment name required; will not overwrite '+str(runroot))
runroot.mkdir(parents=True)
def no_train_labels(event,a):
    if event=='open' and not isinstance(a[0],int):
        if '/preprocessed_v2/train/labels/' in str(Path(os.fsdecode(a[0])).resolve()):raise PermissionError('Training annotations are forbidden')
sys.addaudithook(no_train_labels)
cfg=OmegaConf.load(ROOT/'src/configs/train_config.yaml')
cfg.output_root=str(runroot);cfg.pytorch_data_dir=args.data_override.split('=',1)[1];cfg.seed=args.seed;cfg.experiment_name=args.name
cfg.max_steps=5000  # Preserve the existing baseline LR schedule; Trainer below independently caps500 batches.
cfg.extra_clusters=6;cfg.collapse_early_stop=True
cfg.num_workers=0;cfg.val_freq=50;cfg.batch_size=16
cal=json.loads((OUT/'calibration.json').read_text())
if args.run!='A':
 for k in ['pos_intra_shift','pos_inter_shift','neg_inter_shift']:setattr(cfg,k,cal['candidates'][args.run][k])
assert not cfg.use_true_labels and not cfg.use_salience and not cfg.use_text_prompts
assert cfg.rec_weight==cfg.crf_weight==cfg.aug_alignment_weight==cfg.text_align_weight==0
assert (cfg.pos_intra_weight,cfg.pos_inter_weight,cfg.neg_inter_weight)==(.67,.25,.63)
OmegaConf.save(cfg,runroot/'resolved_config.yaml')
start=time.monotonic();captured={}
class BoundedRun(Callback):
    def __init__(self):self.batches=0
    def on_train_start(self,trainer,model):
        assert trainer.lr_scheduler_configs[0].scheduler.T_max==5000
        self.linear_start={k:v.detach().cpu().clone() for k,v in model.linear_probe.state_dict().items()}
        assert all(not p.requires_grad for p in model.net.full_model.parameters())
    def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
        self.batches+=1
        assert self.batches<=500,'Exceeded authorized budget'
        if self.batches%25==0:print('PROGRESS',args.run,self.batches,'seconds',round(time.monotonic()-start,1),flush=True)
        if self.batches==500:trainer.should_stop=True
    def on_validation_end(self,trainer,model):
        if not trainer.sanity_checking and model.validation_history:
            r=model.validation_history[-1]
            print('VALIDATION',json.dumps(dict(run=args.run,batches=self.batches,code_std=r['code_std'],cosine=r['cross_image_cosine'],usage=r['cluster_usage'],early_stop=r['early_stopped'])),flush=True)
cb=BoundedRun()
def bounded_trainer(**kw):
    assert kw['max_steps']==15000
    kw.update(max_steps=1500,accelerator='cpu',devices=1,callbacks=[cb],enable_checkpointing=False,
              logger=[kw['logger'][0]],enable_progress_bar=False,enable_model_summary=False)
    trainer=Trainer(**kw);captured['trainer']=trainer;return trainer
training.Trainer=bounded_trainer
print('AUTHORIZED LIMIT500 BATCHES; CPU; labels only in evaluation; no supervised fitting',flush=True)
os.chdir(ROOT/'src');training.my_app.__wrapped__(cfg)
trainer=captured['trainer'];model=trainer.lightning_module
assert 0<cb.batches<=500 and trainer.global_step==3*cb.batches
assert all(torch.equal(cb.linear_start[k],v.detach().cpu()) for k,v in model.linear_probe.state_dict().items())
assert all(p.grad is None for p in model.net.full_model.parameters())
if model.validation_history[-1]['batches']!=cb.batches:
    trainer.validate(model,dataloaders=trainer.val_dataloaders,verbose=False)
last=model.validation_history[-1]
checkpoint=runroot/'checkpoints'/'last.ckpt'
assert not checkpoint.exists();trainer.save_checkpoint(checkpoint)
# Preserve a compact, tracked final panel in addition to the new experiment's log directory.
copy=OUT/f'{args.name}_final.png';assert not copy.exists();shutil.copyfile(last['fixed_overlay_path'],copy)
ref=json.loads((ROOT/'audit/scratch/diag/ceiling_b12_c0.json').read_text())['linear']['liver_iou']*100
summary=dict(run=args.run,name=args.name,seed=args.seed,batch_budget=500,batches=cb.batches,optimizer_steps=trainer.global_step,
    early_stopped=last['early_stopped'],monitors={k:last[k] for k in ['code_std','cross_image_cosine','cluster_usage','cluster_entropy','consecutive_bad_checks']},
    cluster_metrics={k:v for k,v in last['metrics'].items() if 'cluster' in k},cluster_to_class=last['cluster_to_class'],
    shifts={k:float(getattr(cfg,k)) for k in ['pos_intra_shift','pos_inter_shift','neg_inter_shift']},
    supervised_reference=dict(liver_IoU=ref,scope='Existing DIAG frozen final-block probe on100 val slices; identical reference, NOT a fitted per-run code probe'),
    checkpoint=str(checkpoint),overlay=str(copy),logged_overlay=last['fixed_overlay_path'],fixed_samples=last['fixed_samples'],
    evaluated_images=last['images'],linear_parameters_unchanged=True,train_mask_reads=0,elapsed_seconds=time.monotonic()-start)
(OUT/f'{args.name}_result.json').write_text(json.dumps(summary,indent=2)+'\n')
print('RESULT',json.dumps(summary),flush=True)
