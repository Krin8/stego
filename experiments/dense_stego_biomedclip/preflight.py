"""Final no-update preflight, serialization parity and current baseline integrity."""
from runtime import *
from model import CorrespondenceModel
from sklearn.cluster import KMeans
import subprocess
seed_all(0);digest=baseline_seal();rr=records();train=[r for r in rr if r['split']=='train'];val=[r for r in rr if r['split']=='val'];assert len(train)==1869 and len(val)==472
for p in HERE.glob('*.py'):ast.parse(p.read_text())
a=CorrespondenceModel('A').eval();x=F.normalize(torch.randn(1,768,28,28),dim=1)
with torch.no_grad():before=a(x)
p=HERE/'logs/synthetic_checkpoint.pt';torch.save(dict(state_dict=a.state_dict(),arm='A'),p);s=torch.load(p,map_location='cpu',weights_only=True);b=CorrespondenceModel('A').eval();result=b.load_state_dict(s['state_dict'],strict=True);assert not result.missing_keys and not result.unexpected_keys
with torch.no_grad():assert torch.equal(before,b(x))
# Exercise installed clustering and generalized C32 distance path, no dataset fit.
sample=np.random.default_rng(123).normal(size=(128,32)).astype(np.float32);km=KMeans(n_clusters=8,n_init=3,max_iter=100,tol=1e-4,random_state=0,algorithm='lloyd').fit(sample);assert np.array_equal(km.predict(sample),distances(sample,km.cluster_centers_).argmin(1))
assert not(HERE/'configs/locked.json').exists() and not(HERE/'predictions/stageA_manifest.json').exists()
status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True);diff=subprocess.check_output(['git','diff'],cwd=ROOT,text=True);assert diff==''
dump('logs/preflight.json',dict(train_patients=len({r['patient'] for r in train}),val_patients=len({r['patient'] for r in val}),train_slices=len(train),val_slices=len(val),all_sources_parse=True,synthetic_checkpoint_replay_exact=True,generalized_distance_matches_sklearn=True,mask_reads=0,training_updates=0,baseline_seal_sha256=digest,baseline_unchanged=True,git_status=status,source_sha256={p.name:sha(p) for p in HERE.glob('*.py')},pending=['Training budget','Token-only vs approved image branch','Suspended training process resolution']));ledger('logs/preflight_access.json');print('PREFLIGHT PASS; NO TRAINING LAUNCHED',flush=True)
