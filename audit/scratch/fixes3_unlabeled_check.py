from fixes3_common import *
import argparse,ast
p=argparse.ArgumentParser();p.add_argument('--fixed',action='store_true');args=p.parse_args()
s=(ROOT/'src/train_segmentation.py').read_text()
if not args.fixed:
 assert 'train_linear_probe' not in s
 print('REPRODUCED: default training always fits supervised CE and reads train masks');raise SystemExit
from unlabeled_training import ImagesOnlyDataset
from data import CHAOS
from chaos_preprocessing import get_chaos_transform
from omegaconf import OmegaConf
from modules import ContrastiveCorrelationLoss,ClusterLookup,norm,sample
from torch import nn
import torch.nn.functional as F
def deny(event,a):
 if event=='open' and not isinstance(a[0],int) and '/train/labels/' in str(Path(a[0]).resolve()):raise PermissionError('train mask read')
sys.addaudithook(deny)
d=CHAOS(str(ROOT/'src/pytorch_data_dir'),'CT','train',get_chaos_transform(),get_chaos_transform(is_label=True),2,True)
x,y,m=ImagesOnlyDataset(d)[0];assert (y==-1).all()
# Execute the actual training method on tiny tensors; optimizer spies perform no updates.
tree=ast.parse(s);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LitUnsupervisedSegmenter');method=next(n for n in cls.body if getattr(n,'name',None)=='training_step')
ns=dict(torch=torch,F=F,norm=norm,sample=sample);exec(compile(ast.Module(body=[method],type_ignores=[]),'training_step','exec'),ns)
class Opt:
 def __init__(self,p):self.p=list(p)
 def zero_grad(self):
  for p in self.p:p.grad=None
 def step(self):pass
class Net(nn.Module):
 def __init__(self):super().__init__();self.head=nn.Conv2d(3,70,1)
 def forward(self,x):return x.detach(),self.head(x)
class Harness(nn.Module):
 training_step=ns['training_step']
 def __init__(self):
  super().__init__();self.cfg=OmegaConf.load(ROOT/'src/configs/train_config.yaml');self.n_classes=2;self.global_step=0;self.net=Net();self.cluster_probe=ClusterLookup(70,8);self.linear_probe=nn.Conv2d(70,2,1);self.contrastive_corr_loss_fn=ContrastiveCorrelationLoss(self.cfg);self.logs={}
 def optimizers(self):return [Opt(m.parameters()) for m in [self.net,self.linear_probe,self.cluster_probe]]
 def log(self,k,v,**kw):self.logs[k]=float(v.detach())
 def manual_backward(self,l):l.backward()
 def lr_schedulers(self):return None
def run(label):
 torch.manual_seed(2);m=Harness();x=torch.randn(2,3,4,4);y=torch.full((2,4,4),label)
 m.training_step({'main':dict(ind=torch.arange(2),img=x,img_aug=x,coord_aug=torch.zeros(2,4,4,2),img_pos=x.flip(0),label=y,label_pos=y)},0)
 assert m.logs['loss/linear_probe']==0 and torch.count_nonzero(m.linear_probe.weight.grad)==0
 return m.logs['loss/total'],m.net.head.weight.grad
la,ga=run(0);lb,gb=run(1);assert la==lb and torch.equal(ga,gb)
print('PASS no train mask reads; zero linear gradients; identical objective/head gradients under label changes')
