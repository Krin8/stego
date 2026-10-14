"""F1 is inherited verbatim except the channel dimension; no data labels read."""
from runtime import *
from model import StegoHead
import runtime as rt

node=next(n for n in ast.parse((HERE/'stages.py').read_text()).body
          if isinstance(n,ast.FunctionDef) and n.name=='score_fields')
exec(compile(ast.Module(body=[node],type_ignores=[]),'stages score_fields','exec'),globals())
rt.reuse(ROOT/'audit/final_pipeline/segment_ct.py',['predict_features'])
G=28
r=dict(box=[-20,30,480],shape=[512,512])
rng=np.random.default_rng(42)
im=rng.uniform(0,1,(512,512)).astype(np.float32)
z=rng.normal(size=(28,28,70)).astype(np.float32);z/=np.linalg.norm(z,axis=-1,keepdims=True)
centers=z.reshape(-1,70)[:8].copy()
_,raw,ours=score_fields(z,im,r,centers,dict(N1=2,N2=None,N3=None),.1)
expected=rt.predict_features(z.reshape(-1,70),im,r,centers,
                             dict(liver_clusters=[2],temperature=.1))
assert all(np.array_equal(a,b) for a,b in zip(ours['N1'],expected))
assert np.array_equal(ours['N1'][0],raw==2)
assert not any(a.any() for rule in ['N2','N3'] for a in ours[rule])
assert set(np.unique(raw)).issubset(set(range(9))) #8 is outside-crop sentinel.
rr=records();train={r['patient'] for r in rr if r['split']=='train'}
val={r['patient'] for r in rr if r['split']=='val'}
assert len(train)==16 and len(val)==4 and not train&val
for file in HERE.glob('*.py'):compile(file.read_text(),str(file),'exec')
dump('logs/integration_checks.json',dict(passed=True,F1_exact=True,
     outside_crop_geometry_preserved=True,abstention_empty=True,
     train_patients=sorted(train,key=int),val_patients=sorted(val,key=int),
     labels_read=False,optimizer_steps=0))
print('INTEGRATION CHECKS PASS',flush=True)
