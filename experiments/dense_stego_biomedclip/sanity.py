"""CPU pre-training checks; no optimizer steps, fitted models, labels or val images."""
from runtime import *
from model import CorrespondenceModel
import correspondence as corr
seed_all(0);seal=baseline_seal();rr=records();record=next(r for r in rr if r['split']=='train');im=load_image(record);assert body_box(im)[0]==record['box'];crop=crop_image(im,record['box']);raw=torch.from_numpy(crop)[None,None];raw=F.interpolate(raw,(448,448),mode='bilinear',align_corners=False,antialias=True)
net=frozen_backbone();t0=time.monotonic();g=torch.Generator().manual_seed(991);va,ia,aa=corr.views(raw,g);vb,ib,ab=corr.views(raw,g);fa=tokens(net,va);fb=tokens(net,vb);backbone_seconds=time.monotonic()-t0
assert not net.full_model.training and all(not p.requires_grad for p in net.parameters())
# Actual preprocessing/features must match the saved baseline for this train image.
original=image_tensor(crop,448)[None];feat=tokens(net,original);saved=np.load(OLD/'B2_448_features.npy',mmap_mode='r')[rr.index(record)];error=float(np.max(np.abs(feat[0].permute(1,2,0).reshape(784,768).numpy()-saved)));assert error<1e-5,error
# Geometry recovers canonical coordinates from both views.
qa,qb,p=corr.matched_locations(aa,ab,torch.Generator().manual_seed(3));assert torch.max(torch.abs(corr.transform(qa[0],aa[0])-p[0]))<1e-6;assert torch.max(torch.abs(corr.transform(qb[0],ab[0])-p[0]))<1e-6
# Identity geometry provides a discriminative synthetic positive test.
identity=torch.tensor([[[1.,0,0],[0,1.,0]]]);z=F.normalize(torch.randn(1,32,28,28),dim=1);same,_=corr.loss(z,z,identity,identity,torch.Generator().manual_seed(11));wrong,_=corr.loss(z,torch.flip(z,[-1]),identity,identity,torch.Generator().manual_seed(11));assert same<wrong
checks=[]
for arm in ['A','B']:
 seed_all(0);m=CorrespondenceModel(arm);opt=m.optimizer(5e-4);bbids={id(x) for x in net.parameters()};newids={id(x) for x in m.parameters()};optids={id(x) for group in opt.param_groups for x in group['params']};assert optids==newids and not optids&bbids
 t=time.monotonic();za,trace=m(fa,ia,True);zb=m(fb,ib);objective,details=corr.loss(za,zb,aa,ab,torch.Generator().manual_seed(11));objective.backward();assert torch.isfinite(objective) and all(p.grad is None for p in net.parameters());grad=sum(float(x.grad.abs().sum()) for x in m.parameters() if x.grad is not None);assert grad>0
 with torch.no_grad():assert torch.equal(m(fa,ia),m(fa,ia))
 checks.append(dict(arm=arm,input=[1,3,448,448],tokens=list(fa.shape),trace=trace,loss=float(objective.detach()),gradient_abs_sum=grad,backbone_trainable=sum(p.numel() for p in net.parameters() if p.requires_grad),new_trainable=sum(p.numel() for p in m.parameters() if p.requires_grad),optimizer_only_new=True,backbone_zero_gradients=True,deterministic=True,seconds=time.monotonic()-t,**details));m.zero_grad(set_to_none=True)
dump('logs/sanity.json',dict(producer='sanity.py',baseline_manifest_sha256=seal,baseline_token_max_abs_error=error,view_geometry_max_error=float(torch.max(torch.abs(corr.transform(qa[0],aa[0])-p[0]))),synthetic_aligned_loss=float(same),synthetic_misaligned_loss=float(wrong),two_view_backbone_seconds=backbone_seconds,checks=checks,optimizer_steps=0,label_files_read=0));ledger('logs/sanity_access.json');assert baseline_seal()==seal;print(json.dumps(checks,indent=2),flush=True)
