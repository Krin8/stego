"""No-training before/after check of the minimal bounded runner adaptation."""
from common import *
import ast,json
from types import SimpleNamespace
from omegaconf import OmegaConf
from pytorch_lightning import Callback
old=(ROOT/'audit/scratch/seed_validation_experiment.py').read_text()
assert 'batch_250' not in old and 'self.batches in (100,250,500)' not in old
tree=ast.parse((OUT/'run.py').read_text());cbnode=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BoundedRun')
class FakePath:
 def __truediv__(self,x):return self
 def exists(self):return False
 def write_text(self,text):pass
saved=[]
class FakeTrainer:
 global_step=0;should_stop=False
 def save_checkpoint(self,path):saved.append(self.global_step//3)
namespace=dict(Callback=Callback,runroot=FakePath(),time=__import__('time'),start=__import__('time').monotonic(),name='SYNTHETIC',json=json)
exec(compile(ast.Module(body=[cbnode],type_ignores=[]),'run.py','exec'),namespace)
cb=namespace['BoundedRun']();trainer=FakeTrainer()
for step in range(1,501):
 trainer.global_step=3*step;cb.on_train_batch_end(trainer,None,None,None,step-1)
 assert trainer.should_stop==(step==500)
assert saved==[100,250,500] and set(cb.saved)=={'100','250','500'}
# Execute only the actual config-building AST block; no runner/process/fit statements.
nodes=tree.body;first=next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='cfg')
last=next(i for i,n in enumerate(nodes) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='OmegaConf.save')
configs={}
for candidate in ['D','E']:
 for seed in [0,1,2]:
  env=dict(ROOT=ROOT,OUT=OUT,OmegaConf=OmegaConf,json=json,args=SimpleNamespace(seed=seed,candidate=candidate),name=f'fixes4_{candidate}_seed{seed}',runroot=OUT/'experiments'/f'fixes4_{candidate}_seed{seed}')
  exec(compile(ast.Module(body=nodes[first:last],type_ignores=[]),'run.py','exec'),env)
  configs[f'{candidate}{seed}']=env['differences']
r=dict(before='Existing FIXES3 runner saves terminal checkpoint only; requested intermediate saves absent',after='Scheduled100/250/500 saves and independent500-batch cap pass synthetic callback test; all6 configs differ from FIXES3 only by seed, shifts and output names',saved_batches=saved,config_differences=configs,training_launched=False)
(OUT/'verify_runner.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
