"""Bidirectional geometry-matched InfoNCE. No labels, nearest-image cache or text loss."""
import math
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from runtime import MEAN,STD
# All geometry uses normalized pixel-centre coordinates and align_corners=False.
def affine_batch(batch,generator):
 scale=.80+.20*torch.rand(batch,generator=generator);angle=(torch.rand(batch,generator=generator)*10-5)*math.pi/180
 translation=(torch.rand(batch,2,generator=generator)*2-1)*.08
 a=torch.zeros(batch,2,3);a[:,0,0]=scale*angle.cos();a[:,0,1]=-scale*angle.sin();a[:,1,0]=scale*angle.sin();a[:,1,1]=scale*angle.cos();a[:,:,2]=translation;return a

def views(image,generator):
 # image Bx1x448x448, CT windowed0..1. Transformation maps view->canonical crop.
 a=affine_batch(len(image),generator);grid=F.affine_grid(a,image.shape,align_corners=False);raw=F.grid_sample(image,grid,mode='bilinear',padding_mode='border',align_corners=False)
 gain=.95+.10*torch.rand(len(image),1,1,1,generator=generator);offset=(torch.rand(len(image),1,1,1,generator=generator)*2-1)*.02;raw=(raw*gain+offset).clamp(0,1)
 rgb=raw.repeat(1,3,1,1);return TF.normalize(rgb,MEAN,STD),raw,a

def transform(points,a):return points@a[:,:2].T+a[:,2]
def invert(points,a):return (points-a[:,2])@torch.linalg.inv(a[:,:2]).T

def matched_locations(a,b,generator,n=96):
 # Uniform canonical overlap sampling; no tissue/organ/mask selection.
 pa=[];pb=[];canonical=[]
 for aa,bb in zip(a,b):
  candidates=torch.rand(8192,2,generator=generator)*2-1;qa=invert(candidates,aa);qb=invert(candidates,bb)
  valid=(qa.abs()<.90).all(1)&(qb.abs()<.90).all(1);assert valid.sum()>=n
  p=candidates[valid][:n];canonical.append(p);pa.append(invert(p,aa));pb.append(invert(p,bb))
 return torch.stack(pa),torch.stack(pb),torch.stack(canonical)
def sample(z,xy):return F.normalize(F.grid_sample(z,xy[:,None],mode='bilinear',align_corners=False)[:,:,0].transpose(1,2),dim=-1)
def loss(za,zb,aa,ab,generator,temperature=.10,n=96,negative_distance=.15):
 qa,qb,p=matched_locations(aa,ab,generator,n);ea,eb=sample(za,qa),sample(zb,qb)
 logits=ea@eb.transpose(1,2)/temperature;dist=torch.cdist(p,p);allow=dist>=negative_distance;eye=torch.eye(n,dtype=torch.bool)[None];allow=allow|eye;assert (allow.sum(-1)>1).all()
 logits=logits.masked_fill(~allow,-torch.inf);target=torch.arange(n)[None].expand(len(za),-1)
 value=.5*(F.cross_entropy(logits.reshape(-1,n),target.reshape(-1))+F.cross_entropy(logits.transpose(1,2).reshape(-1,n),target.reshape(-1)))
 return value,dict(positive_cosine=(ea*eb).sum(-1).mean().detach().item(),negative_count=(allow.sum(-1)-1).float().mean().item(),sampled_locations=n)
