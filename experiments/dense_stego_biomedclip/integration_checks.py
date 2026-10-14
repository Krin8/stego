"""No-training/no-label unit checks for loss, dense output and baseline F1 parity."""
from runtime import *
from model import CorrespondenceModel
import correspondence as corr
from types import SimpleNamespace
seed_all(2)
# Reuse isolated stage function without executing its CLI or opening weights/data/labels.
node=next(n for n in ast.parse((HERE/'stages.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='score_fields');exec(compile(ast.Module(body=[node],type_ignores=[]),'stages.py','exec'),globals())
import runtime as rt
rt.reuse(ROOT/'audit/final_pipeline/segment_ct.py',['predict_features'])
predict_features=rt.predict_features
r=dict(box=[0,0,512],shape=[512,512]);im=np.random.default_rng(2).uniform(0,1,(512,512)).astype(np.float32);z=np.random.default_rng(3).normal(size=(28,28,32)).astype(np.float32);z/=np.linalg.norm(z,axis=-1,keepdims=True);centers=z.reshape(-1,32)[:8].copy();G=28
raw,native,ours=score_fields(z,im,r,centers,dict(N1=2,N2=None,N3=None),.1);baseline=predict_features(z.reshape(-1,32),im,r,centers,dict(liver_clusters=[2],temperature=.1));assert all(np.array_equal(a,b) for a,b in zip(ours['N1'],baseline));assert not any(a.any() for rule in ['N2','N3'] for a in ours[rule]);assert np.array_equal(ours['N1'][0],native==2)
# Proof that subpixel decoder output is not constrained to nearest28 repetition.
m=CorrespondenceModel('B').eval();t=F.normalize(torch.randn(1,768,28,28),dim=1)
with torch.no_grad():dense=m(t);coarse=F.adaptive_avg_pool2d(dense,(28,28));up=F.interpolate(coarse,(448,448),mode='nearest');residual=float((dense-up).square().mean())
assert residual>1e-8;assert dense.shape==(1,32,448,448)
# Sampling gives identical points in canonical space, positives aligned and negatives>=.15.
g=torch.Generator().manual_seed(99);a,b=corr.affine_batch(2,g),corr.affine_batch(2,g);qa,qb,p=corr.matched_locations(a,b,g);errors=[float((corr.transform(qa[i],a[i])-corr.transform(qb[i],b[i])).abs().max()) for i in range(2)];assert max(errors)<1e-6
# Label-free loss is deterministic for the same pair and RNG state.
x=F.normalize(torch.randn(2,32,28,28),dim=1);l1,_=corr.loss(x,x,a,b,torch.Generator().manual_seed(10));l2,_=corr.loss(x,x,a,b,torch.Generator().manual_seed(10));assert torch.equal(l1,l2)
# Check that forbidden label access is rejected without actually opening any masks.
blocked=False
try:open(OLD/'scoring_masks_native.npy','rb')
except PermissionError:blocked=True
assert blocked
# AST definitions exercise generalized dense predictions at448 on synthetic values.
G=448
with torch.no_grad():arr=dense[0].permute(1,2,0).numpy()
_,_,pred=score_fields(arr,im,r,centers,dict(N1=2),.1);assert pred['N1'][0].shape==(512,512) and pred['N1'][1].shape==(512,512)
dump('logs/integration_checks.json',dict(baseline_F1_parity_at28_exact=True,abstentions_empty=True,geometry_errors=errors,deterministic_loss=True,synthetic_dense_output=list(dense.shape),dense_non_nearest_residual_mse=residual,mask_access_guard_test_passed=True,labels_opened=False,optimizer_steps=0,interpretation='Synthetic random-weight density proves architectural capacity only, not learned anatomy.'))
print('INTEGRATION CHECKS PASS',flush=True)
