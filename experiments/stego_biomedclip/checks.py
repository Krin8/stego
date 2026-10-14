"""Synthetic reference parity and frozen-backbone/cache checks; no optimizer steps."""
from runtime import *
from model import StegoHead, objective
from reference_loss import ContrastiveCorrelationLoss, sample, super_perm
from geometry import coordinate_image, alignment
import copy

seed_all(0)
c = SimpleNamespace(**read(HERE/'configs/proposed.json'))
model = StegoHead()
params = sum(p.numel() for p in model.parameters())
assert params == 698252
tree = ast.parse((HERE/'reference/src/modules.py').read_text())
nodes = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
         and n.name in ['norm','tensor_correlation','sample','super_perm',
                        'sample_nonzero_locations','ContrastiveCorrelationLoss']]
for n in nodes:
    if isinstance(n,ast.FunctionDef): n.decorator_list=[]
ns = dict(torch=torch, nn=torch.nn, F=F)
exec(compile(ast.Module(body=nodes,type_ignores=[]), 'pinned reference parity', 'exec'),ns)
f = F.normalize(torch.randn(2,768,5,5),dim=1)
fp = F.normalize(torch.randn_like(f),dim=1)
code = torch.randn(2,70,5,5,requires_grad=True)
cp = torch.randn_like(code,requires_grad=True)
ours = ContrastiveCorrelationLoss(c); ref = ns['ContrastiveCorrelationLoss'](c)
torch.manual_seed(7); out = ours(f,fp,None,None,code,cp)
torch.manual_seed(7); expected = ref(f,fp,None,None,code,cp)
assert all(torch.equal(x,y) for x,y in zip(out,expected))
weights = [c.pos_intra_weight,c.pos_inter_weight,c.neg_inter_weight]
l1 = sum(w*out[k].mean() for w,k in zip(weights,[0,2,4]))
l2 = sum(w*expected[k].mean() for w,k in zip(weights,[0,2,4]))
g1 = torch.autograd.grad(l1,(code,cp),retain_graph=True)
g2 = torch.autograd.grad(l2,(code,cp))
assert all(torch.equal(x,y) for x,y in zip(g1,g2))
for _ in range(20):
    perm = super_perm(16,torch.device('cpu')); assert (perm != torch.arange(16)).all()
# Two independent projection dropout calls and one signal dropout: exact RNG parity.
manual = copy.deepcopy(model)
torch.manual_seed(11); signal,z = model(f)
torch.manual_seed(11)
zz = manual.cluster1(manual.dropout(f))+manual.cluster2(manual.dropout(f))
ff = manual.dropout(f)
assert torch.equal(z,zz) and torch.equal(signal,ff)
code = F.normalize(torch.randn(1,70,5,5),dim=1)
grid = coordinate_image(5,5)[None]
assert abs(float(alignment(code,code,grid))+1) < 1e-6
assert abs(float(alignment(code,code.flip(-1),grid.flip(-1)))+1) < 1e-6
rr = records(); r = next(r for r in rr if r['split']=='train')
net = frozen_backbone()
x = image_tensor(crop_image(load_image(r),r['box']),448)[None]
with torch.no_grad():
    a=tokens(net,x); b=tokens(net,x)
assert torch.equal(a,b)
cached=np.load(OLD/'B2_448_features.npy',mmap_mode='r')
expected=torch.from_numpy(np.array(cached[rr.index(r)])).reshape(28,28,768).permute(2,0,1)[None]
error=float((a-expected).abs().max()); assert error < 2e-5
model.train(); sf,z=model(a); sf2,z2=model(b)
value,_=objective(ours,c,torch.cat([sf,sf2]),torch.cat([sf2,sf]),
                  torch.cat([z,z2]),torch.cat([z2,z]))
value.backward()
assert all(p.grad is None for p in net.parameters())
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
blocked=False
try: open(OLD/'scoring_masks_native.npy','rb')
except PermissionError: blocked=True
assert blocked
result=dict(passed=True,trainable_head=params,backbone_trainable=0,
    reference_loss_and_gradient_exact=True,head_dropout_exact=True,
    optional_geometry_identity_flip=True,feature_shape=list(a.shape),code_shape=list(z.shape),
    deterministic_backbone=True,cache_max_absolute_error=error,backbone_gradients=0,
    optimizer_steps=0,mask_guard_passed=True,labels_opened=False)
dump('logs/checks.json',result); ledger('logs/checks_access.json');print(json.dumps(result),flush=True)
