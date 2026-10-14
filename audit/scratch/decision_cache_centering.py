"""Cache identity observations and a tiny centering comparison; no model runs."""
import decision_guard as g
import hashlib,json,ast
from types import SimpleNamespace
import numpy as np,torch
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(2);torch.manual_seed(11)
weight=g.ROOT/'src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin'
h=hashlib.sha256()
with weight.open('rb') as f:
    for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
observations=[]
for split in ['train','val']:
    names=sorted(p.name for p in (g.ROOT/'src/pytorch_data_dir/preprocessed'/split/'images').glob('CT_*.png'))
    digest=hashlib.sha256(json.dumps(names,separators=(',',':')).encode()).hexdigest()
    path=g.ROOT/'src/pytorch_data_dir/nns'/f'nns_vit_base_chaos_{split}_five_224.npz'
    with np.load(path,allow_pickle=False) as cache:
        n=cache['nns'];r=dict(split=split,current_filename_count=len(names),first=names[:3],last=names[-3:],current_names_sha256=digest,cache=str(path.relative_to(g.ROOT)),keys=list(cache),shape=list(n.shape),dtype=str(n.dtype),index_range=[int(n.min()),int(n.max())],first_column_self_fraction=float(np.mean(n[:,0]==np.arange(len(n)))))
    observations.append(r)
g.append('## M4 — Current cache identity observations\n\nCommand: `python3 -B audit/scratch/decision_cache_centering.py`. Only existing cache contents and weight bytes were read. Filename ordering below is the hypothetical static order before the mixed-family guard; it does not instantiate or bypass that guard for training. Array dimensions/self-neighbors do not prove provenance.\n\nBackbone file SHA256: `'+h.hexdigest()+'` (`'+str(weight)+'`).\n\n```json\n'+json.dumps(observations,indent=2)+'\n```')
tree=ast.parse((g.ROOT/'src/modules.py').read_text());names={'norm','tensor_correlation','ContrastiveCorrelationLoss'}
selected=[n for n in tree.body if getattr(n,'name',None) in names]
ns=dict(torch=torch,nn=nn,F=F);exec(compile(ast.Module(body=selected,type_ignores=[]),str(g.ROOT/'src/modules.py'),'exec'),ns)
cfg=SimpleNamespace(pointwise=True,zero_clamp=True,stabalize=False);fn=ns['ContrastiveCorrelationLoss'](cfg)
f=torch.randn(2,3,2,2);other=torch.randn(2,3,2,2);c=torch.randn(2,4,2,2);d=torch.randn(2,4,2,2)
results=[]
for name,fa,fb,ca,cb,b in [('self',f,f,c,c,.18),('neighbor',f,other,c,d,.12),('negative',f,-other,c,d,.46)]:
    actual,_=fn.helper(fa,fb,ca,cb,b)
    affinity=ns['tensor_correlation'](F.normalize(fa,dim=1),F.normalize(fb,dim=1))
    sim=ns['tensor_correlation'](F.normalize(ca,dim=1),F.normalize(cb,dim=1)).clamp_min(0)
    centered=affinity-affinity.mean((3,4),keepdim=True)
    literal=-((centered-b)*sim).mean();offset=affinity.mean()*sim.mean()
    results.append(dict(term=name,reference_loss=float(actual.mean()),literal_eq3_loss=float(literal),mean_affinity=float(affinity.mean()),equivalence_error=float((actual.mean()-(literal-offset)).abs())))
g.append('## m1 — Centering behavior, without changing it\n\nSource: `src/modules.py:478–481`. Paper contract is the already-recorded Eq(3) ledger in `audit/AUDIT.md:263`; no paper reread. Command: `python3 -B audit/scratch/decision_cache_centering.py`, seed11,2×3×2×2 synthetic features.\n\n```json\n'+json.dumps(results,indent=2)+'\n```\n\nWith `mu=mean(F)` and `Splus=max(S,0)`, current `L_ref = L_literal - mu * mean(Splus)` (floating-point residual below1e-7 here). Equivalently, current code uses an effective literal-centering shift `b - mu`. This is batch-dependent, so changing centering while retaining inherited shifts is not numerically neutral. For constant affinity0.7 and shift0.18, literal centering gives a positive0.18*Splus penalty, while restored-mean centering gives a negative0.52*Splus attraction. This is an algebraic illustration, not a measured training outcome.')
print(json.dumps(dict(backbone_sha256=h.hexdigest(),caches=observations,centering=results)))
