"""Final bounded CPU component smoke: config, real image/model, loss, metrics."""
import guard, sys, signal
signal.alarm(110)
sys.modules['wandb'] = None
import torch, numpy as np
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment
torch.set_num_threads(4); torch.manual_seed(11)
cfg=OmegaConf.load(guard.ROOT/'src/configs/train_config.yaml')
assert cfg.dim == 70 and cfg.extra_clusters == 0 and not cfg.use_text_prompts
print('PASS config: dim70, extra_clusters0, text disabled', flush=True)
ns=guard.definitions('src/modules.py', ['BiomedCLIPFeaturizer','norm','tensor_correlation','sample','ContrastiveCorrelationLoss'], dict(torch=torch,nn=nn,F=F,__file__=str(guard.ROOT/'src/modules.py')))
tr=guard.definitions('src/utils.py', ['normalize','ToTargetTensor','identity_transform','get_transform'], dict(torch=torch,np=np,T=T,Image=Image))
net=ns['BiomedCLIPFeaturizer'](cfg.dim,cfg).cpu().eval()
path=guard.ROOT/'src/pytorch_data_dir/preprocessed/train/images/CT_28_0110.png'
x=tr['get_transform'](224,False,'center')(Image.open(path).convert('RGB'))[None]
with torch.no_grad(): features,code=net(x)
assert features.shape==(1,768,14,14) and code.shape==(1,70,14,14)
assert torch.isfinite(features).all() and torch.isfinite(code).all()
print('PASS real image forward: (1,3,224,224) -> (1,768,14,14) -> (1,70,14,14)', flush=True)
# One real image's feature map and its reflection form a synthetic loss batch.
f=torch.cat([features,features.flip(-1)]); c=torch.cat([code,code.flip(-1)])
r=ns['ContrastiveCorrelationLoss'](cfg)(f,f.flip(0),None,None,c,c.flip(0),f,c)
loss=cfg.correspondence_weight*(cfg.pos_intra_weight*r[0]+cfg.pos_inter_weight*r[2]+cfg.neg_inter_weight*r[4].mean())
assert torch.isfinite(loss)
print('PASS configured correspondence loss:',float(loss),flush=True)
class Metric(nn.Module):
    def __init__(self,**kwargs):super().__init__()
    def add_state(self,name,default,**kwargs):setattr(self,name,default)
metric_cls=guard.definitions('src/utils.py',['UnsupervisedMetrics'],dict(torch=torch,np=np,Metric=Metric,linear_sum_assignment=linear_sum_assignment))['UnsupervisedMetrics']
metric=metric_cls('',2,0,True)
metric.update(torch.tensor([1,1,0,0]),torch.tensor([0,0,1,1]))
scores=metric.compute(); assert scores=={'mIoU':100.,'Accuracy':100.}
assert metric.map_clusters(torch.tensor([1,0])).tolist()==[0,1]
print('PASS synthetic metric:',scores,flush=True)
print('No optimizer step, cache write, dataset change or checkpoint save/load of a trained head. Full dataset startup remains blocked by explicit M4/M5 guards.',flush=True)
