"""CPU component smoke from raw approved DICOM, without an optimizer update."""
import fixes2_guard as g
import sys,torch,numpy as np
from torch import nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,str(g.ROOT/'src'));sys.modules['wandb']=None
from data import CHAOS
from chaos_preprocessing import get_chaos_transform
torch.set_num_threads(4);torch.manual_seed(11)
cfg=OmegaConf.load(g.ROOT/'src/configs/train_config.yaml')
assert cfg.dim==70 and cfg.extra_clusters==0 and cfg.chaos_n_classes==2 and not cfg.use_text_prompts
print('PASS config: 70-D, CT2, extra0, text disabled',flush=True)
dataset=CHAOS(str(g.ROOT/'src/pytorch_data_dir'),'CT','train',get_chaos_transform(),get_chaos_transform(is_label=True),2,False)
index=next(i for i,r in enumerate(dataset.records) if r['patient_id']=='1' and r['instance']==50)
x,label,_=dataset[index];assert x.shape==(3,224,224)
ns=g.definitions('src/modules.py',['BiomedCLIPFeaturizer','norm','tensor_correlation','sample','ContrastiveCorrelationLoss'],dict(torch=torch,nn=nn,F=F,__file__=str(g.ROOT/'src/modules.py')))
net=ns['BiomedCLIPFeaturizer'](cfg.dim,cfg).cpu().eval()
assert not net.full_model.training and all(not p.requires_grad for p in net.full_model.parameters())
features,code=net(x[None])
assert features.shape==(1,768,14,14) and code.shape==(1,70,14,14)
assert not features.requires_grad and code.requires_grad and torch.isfinite(code).all()
print('PASS real raw DICOM -> normalized (1,3,224,224) -> features(1,768,14,14) -> code(1,70,14,14)',dataset.records[index]['dicom'],flush=True)
# Actual pretrained model/head: diagnostic CE must not send any gradient into net.
probe=nn.Conv2d(70,2,1)
small_label=F.interpolate(label[None,None].float(),(14,14),mode='nearest').squeeze(1).long()
F.cross_entropy(probe(code.detach()),small_label).backward()
assert probe.weight.grad is not None and all(p.grad is None for p in net.parameters())
print('PASS actual pretrained backbone/head receives no diagnostic CE gradient',flush=True)
f=torch.cat([features,features.flip(-1)]);c=torch.cat([code,code.flip(-1)])
results=ns['ContrastiveCorrelationLoss'](cfg)(f,f.flip(0),None,None,c,c.flip(0),f,c)
loss=cfg.correspondence_weight*(cfg.pos_intra_weight*results[0]+cfg.pos_inter_weight*results[2]+cfg.neg_inter_weight*results[4].mean())
assert torch.isfinite(loss);loss.backward()
assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.cluster1.parameters())
assert all(p.grad is None for p in net.full_model.parameters())
print('PASS configured correspondence loss/backward:',float(loss.detach()),'; frozen backbone; no optimizer step',flush=True)
class Metric(nn.Module):
    def __init__(self,**kwargs):super().__init__()
    def add_state(self,name,default,**kwargs):self.register_buffer(name,default)
metric_type=g.definitions('src/utils.py',['UnsupervisedMetrics'],dict(torch=torch,np=np,Metric=Metric,linear_sum_assignment=linear_sum_assignment))['UnsupervisedMetrics']
metric=metric_type('',2,0,True);metric.update(torch.tensor([1,1,0,0]),torch.tensor([0,0,1,1]))
scores=metric.compute();assert scores=={'mIoU':100.,'Accuracy':100.}
assert metric.map_clusters(torch.tensor([1,0])).tolist()==[0,1]
print('PASS synthetic Hungarian metric:',scores,flush=True)
print('No training, derivative/cache generation, checkpoint save or dataset mutation. Full Lightning lifecycle not executed.',flush=True)
