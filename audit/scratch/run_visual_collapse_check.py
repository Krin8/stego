"""Read-only CPU diagnosis of the saved w44g1zvt run; no optimizer or file outputs."""
import fixes2_guard as g
import sys,json,torch,numpy as np
from torch import nn
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
sys.path.insert(0,str(g.ROOT/'src'));sys.modules['wandb']=None
torch.set_num_threads(4);torch.manual_seed(11)
from chaos_preprocessing import get_chaos_transform
cfg=OmegaConf.load(g.ROOT/'logs/chaos/chaos_chaos_v2_ct_date_Sep30_15-52-07/lightning_logs/version_0/hparams.yaml').cfg
ns=g.definitions('src/modules.py',['BiomedCLIPFeaturizer','norm','tensor_correlation','sample','ContrastiveCorrelationLoss','ClusterLookup'],dict(torch=torch,nn=nn,F=F,__file__=str(g.ROOT/'src/modules.py')))
root=g.ROOT/'src/pytorch_data_dir/preprocessed_v2'
records=json.loads((root/'manifest.json').read_text())['samples']
selected=[]
for patient in ['5','6','8','30']:
    rs=[r for r in records if r['modality']=='CT' and r['split']=='val' and r['patient_id']==patient]
    selected.append(rs[len(rs)//2])
xs=[];ys=[]
for r in selected:
    with Image.open(root/r['images']) as im:xs.append(get_chaos_transform()(im))
    with Image.open(root/r['labels']) as im:ys.append((get_chaos_transform(is_label=True)(im)[0]>0).long())
x=torch.stack(xs);y=torch.stack(ys)
net=ns['BiomedCLIPFeaturizer'](70,cfg).cpu().eval()
ckptpath=g.ROOT/'checkpoints/chaos/chaos_chaos_v2_ct_date_Sep30_15-52-07/checkpoint-epoch=10-step=3600-mIoU=test/cluster/mIoU=46.1792.ckpt'
ckpt=torch.load(ckptpath,map_location='cpu',mmap=True,weights_only=False)
sd=ckpt['state_dict']
for name in ['cluster1','cluster2']:
    getattr(net,name).load_state_dict({k[len('net.'+name+'.'):]:v for k,v in sd.items() if k.startswith('net.'+name+'.')},strict=True)
cluster=ns['ClusterLookup'](70,2);cluster.load_state_dict({k[len('cluster_probe.'):]:v for k,v in sd.items() if k.startswith('cluster_probe.')},strict=True)
linear=nn.Conv2d(70,2,1);linear.load_state_dict({k[len('linear_probe.'):]:v for k,v in sd.items() if k.startswith('linear_probe.')},strict=True)
with torch.no_grad():
    feats,code=net(x)
    up=F.interpolate(code,(224,224),mode='bilinear',align_corners=False)
    _,cp=cluster(up,None);lp=linear(up).argmax(1)
    fn=F.normalize(feats.flatten(2).transpose(1,2),dim=-1)
    cn=F.normalize(code.flatten(2).transpose(1,2),dim=-1)
    report=dict(checkpoint=str(ckptpath.relative_to(g.ROOT)),step=ckpt['global_step'],samples=[r['stem'] for r in selected],shapes=[list(feats.shape),list(code.shape)],liver_pixels=[int(z.sum()) for z in y],raw_cluster_counts=torch.bincount(cp.argmax(1).flatten(),minlength=2).tolist(),linear_counts=torch.bincount(lp.flatten(),minlength=2).tolist(),feature_cos_cross_image_mean=float((fn@fn.roll(1,0).transpose(1,2)).mean()),code_cos_cross_image_mean=float((cn@cn.roll(1,0).transpose(1,2)).mean()),code_cos_within_image_mean=float((cn@cn.transpose(1,2)).mean()),code_norm_mean=float(code.norm(dim=1).mean()),backbone_finite=bool(torch.isfinite(feats).all()),code_finite=bool(torch.isfinite(code).all()))
    coords=torch.rand((4,11,11,2))*2-1
    sampled=ns['sample'](feats,coords)
    for name,a,b,shift in [('intra',sampled,sampled,cfg.pos_intra_shift),('random_cross_patient',sampled,sampled.roll(1,0),cfg.neg_inter_shift)]:
        fd=ns['tensor_correlation'](ns['norm'](a),ns['norm'](b));old=fd.mean();fd=fd-fd.mean([3,4],keepdim=True);fd=fd-fd.mean()+old
        report[name]=dict(shift=float(shift),restored_affinity_mean=float(fd.mean()),attraction_fraction=float((fd>shift).float().mean()),affinity_percentiles=torch.quantile(fd.flatten(),torch.tensor([.01,.1,.5,.9,.99])).tolist())
    # Identical coordinates/negatives compare loss preference, not training outcomes.
    def objective(c):
        torch.manual_seed(19)
        o=ns['ContrastiveCorrelationLoss'](cfg)(feats,feats.roll(1,0),None,None,c,c.roll(1,0),feats,c)
        return float(cfg.pos_intra_weight*o[0]+cfg.pos_inter_weight*o[2]+cfg.neg_inter_weight*o[4].mean())
    report['synthetic_cross_patient_pair_objective']={'constant_code':objective(torch.ones_like(code)),'random_code':objective(torch.randn_like(code)),'trained_code':objective(code)}
print(json.dumps(report,indent=2))
