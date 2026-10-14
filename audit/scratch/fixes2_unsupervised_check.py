"""Execute unchanged training_step on tiny tensors, with optimizer steps disabled."""
import fixes2_guard as g
import ast,copy,json,torch,numpy as np
from torch import nn
import torch.nn.functional as F
from omegaconf import OmegaConf
torch.set_num_threads(2)
cfg=OmegaConf.load(g.ROOT/'src/configs/train_config.yaml')
assert not cfg.use_true_labels and not cfg.use_salience and not cfg.use_text_prompts
assert cfg.text_align_weight==cfg.aug_alignment_weight==cfg.rec_weight==cfg.crf_weight==0
ns=g.definitions('src/modules.py',['norm','tensor_correlation','sample','sample_nonzero_locations','ContrastiveCorrelationLoss','ClusterLookup'],dict(torch=torch,nn=nn,F=F))
source=(g.ROOT/'src/train_segmentation.py').read_text();tree=ast.parse(source)
klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LitUnsupervisedSegmenter')
method=next(n for n in klass.body if getattr(n,'name',None)=='training_step')
exec(compile(ast.Module(body=[method],type_ignores=[]),str(g.ROOT/'src/train_segmentation.py'),'exec'),ns)
class NoUpdate:
    def __init__(self,params):self.params=list(params)
    def zero_grad(self):
        for p in self.params:p.grad=None
    def step(self):pass
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=nn.Conv2d(3,6,1).requires_grad_(False);self.head=nn.Conv2d(6,70,1)
    def forward(self,x):
        with torch.no_grad():f=self.backbone(x)
        return f,self.head(f)
class Harness(nn.Module):
    training_step=ns['training_step']
    def __init__(self,config):
        super().__init__();self.cfg=config;self.n_classes=2;self.global_step=0
        self.net=TinyNet();self.linear_probe=nn.Conv2d(70,2,1);self.cluster_probe=ns['ClusterLookup'](70,2)
        self.linear_probe_loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([1.,5.]))
        self.contrastive_corr_loss_fn=ns['ContrastiveCorrelationLoss'](config);self.text_embeddings=None;self.logged={}
    def optimizers(self):return [NoUpdate(m.parameters()) for m in [self.net,self.linear_probe,self.cluster_probe]]
    def log(self,name,value,**kw):self.logged[name]=float(value.detach())
    def manual_backward(self,loss):loss.backward()
    def lr_schedulers(self):return None
def run(label,config=cfg):
    torch.manual_seed(42);m=Harness(config)
    img=torch.randn(2,3,4,4);batch=dict(ind=torch.arange(2),img=img,img_aug=img,coord_aug=torch.zeros(2,4,4,2),img_pos=img.flip(0),label=torch.full((2,4,4),label),label_pos=torch.full((2,4,4),1-label),mask=torch.full((2,4,4),float(label)),mask_pos=torch.full((2,4,4),float(1-label)))
    original=[p.detach().clone() for p in m.parameters()]
    m.training_step({'main':batch},0)
    assert all(torch.equal(a,p.detach()) for a,p in zip(original,m.parameters()))
    assert all(p.grad is None for p in m.net.backbone.parameters())
    grads={name:p.grad.clone() if p.grad is not None else None for name,p in m.named_parameters()}
    return grads,m.logged
a,la=run(0);b,lb=run(1)
for name in a:
    if name.startswith(('net.head','cluster_probe')):
        assert a[name] is not None and torch.equal(a[name],b[name]),name
assert not torch.equal(a['linear_probe.weight'],b['linear_probe.weight'])
linear_only=copy.deepcopy(cfg);linear_only.correspondence_weight=0
c,lc=run(0,linear_only);assert all(v is None for k,v in c.items() if k.startswith('net.'))
fixed = '--expect-fixed' in __import__('sys').argv
if fixed:
    assert la['loss/total']==lb['loss/total']
    assert la['loss/linear_probe']!=lb['loss/linear_probe']
    assert 'loss/linear' not in la
    ignored,li=run(-1)
    assert li['loss/linear_probe']==0 and li['loss/total']==la['loss/total']
    for name in a:
        if name.startswith(('net.head','cluster_probe')):assert torch.equal(a[name],ignored[name]),name
else:
    assert la['loss/total']!=lb['loss/total']  # Before the approved headline-loss change.
assert 'monitor="test/cluster/mIoU"' in source
print(json.dumps(dict(segmentation_gradient_max_label_change=max(float((a[k]-b[k]).abs().max()) for k in a if k.startswith('net.head')),cluster_gradient_max_label_change=max(float((a[k]-b[k]).abs().max()) for k in a if k.startswith('cluster_probe')),linear_gradient_changes=True,backbone_gradient=None,linear_only_head_gradient=None,total_loss_for_labels_0_1=[la['loss/total'],lb['loss/total']],optimizer_updates=0)))
print('PASS gradient isolation; headline excludes diagnostic CE' if fixed else 'BEFORE headline includes diagnostic CE')
