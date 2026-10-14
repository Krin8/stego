"""Two predeclared arms. No categorical-mask resizing in the dense decoder."""
import torch
from torch import nn
from torch.nn import functional as F
class Refine(nn.Module):
 def __init__(self,c):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(c,c,3,padding=1,groups=c),nn.GroupNorm(4,c),nn.GELU(),nn.Conv2d(c,c,1))
 def forward(self,x):return x+self.net(x)
class UpsampleBlock(nn.Module):
 def __init__(self,c=16):
  super().__init__();self.up=nn.Sequential(nn.Conv2d(c,4*c,1),nn.PixelShuffle(2),nn.GroupNorm(4,c),nn.GELU());self.refine=Refine(c)
 def forward(self,x):return self.refine(self.up(x))
class CorrespondenceModel(nn.Module):
 def __init__(self,arm,ct_branch=False):
  super().__init__();assert arm in ['A','B'];assert not(ct_branch and arm=='A');self.arm=arm;self.ct_branch=ct_branch
  self.project=nn.Sequential(nn.Conv2d(768,128,1),nn.GELU(),nn.Conv2d(128,32,1));self.refine=Refine(32)
  if arm=='B':
   self.decode_in=nn.Conv2d(32,16,1);self.up=nn.ModuleList([UpsampleBlock() for _ in range(4)]);self.out=nn.Conv2d(16,32,1)
   if ct_branch:self.image_branch=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.GELU(),Refine(16))
 def forward(self,tokens,image=None,return_trace=False):
  x=self.refine(self.project(tokens));trace={'projection':list(x.shape)}
  if self.arm=='B':
   x=self.decode_in(x)
   for block in self.up:x=block(x);trace[str(x.shape[-1])]=list(x.shape)
   if self.ct_branch:
    assert image is not None and image.shape[1:]==(1,448,448);x=x+self.image_branch(image)
   x=self.out(x)
  z=F.normalize(x,dim=1);trace['output']=list(z.shape)
  return (z,trace) if return_trace else z
 def optimizer(self,lr):return torch.optim.Adam(self.parameters(),lr=lr)
