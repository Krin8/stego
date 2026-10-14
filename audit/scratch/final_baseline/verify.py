"""Verify reusable features against the actual frozen extractor without labels."""
import os
os.environ['FINAL_BASELINE_LABEL_FREE']='1'
from common import *
from collections import Counter
BASE=ROOT/'audit/scratch/baseline'
meta=json.loads((BASE/'samples.json').read_text())
inv=json.loads((FORE/'inventory.json').read_text())
model,ck=load(inv['seed0']['path'])
assert not model.training and all(not p.requires_grad for p in model.parameters())
checks={}; all_ids=[]; all_patients=[]
for split in ['train','val']:
 ds=dataset(model.cfg,split)
 ids=[r['sample_id'] for r in ds.dataset.records]
 pats=[str(r['patient_id']) for r in ds.dataset.records]
 assert ids==meta[split+'_ids'] and pats==meta[split+'_patients']
 assert all(s[2]=='CT' for s in ds.dataset.samples)
 ds.dataset=ImagesOnlyDataset(ds.dataset)
 batch=next(iter(DataLoader(ds,batch_size=16,shuffle=False,num_workers=0)))
 with torch.inference_mode():features,_=model.net(batch['img'])
 if split=='train':
  stored=np.load(BASE/'train_features.npy',mmap_mode='r')
  actual=features.permute(0,2,3,1).reshape(16,196,768).numpy()
 else:
  stored=np.load(FORE/'val_features.npy',mmap_mode='r');actual=features.numpy()
 difference=float(np.max(np.abs(actual-stored[:16])))
 assert difference<1e-5,(split,difference)
 assert len(stored)==len(ds)
 checks[split]=dict(slices=len(ds),patients=dict(Counter(pats)),verified_live_images=16,
                   feature_max_absolute_difference=difference,stored_shape=list(stored.shape))
 all_ids+=ids;all_patients+=pats
assert len(all_ids)==2341 and len(set(all_ids))==2341 and len(set(all_patients))==20
assert not set(meta['train_patients'])&set(meta['val_patients'])
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
inputs=[BASE/'train_features.npy',FORE/'val_features.npy',BASE/'samples.json',
        ROOT/'src/modules.py',ROOT/'src/chaos_preprocessing.py',ROOT/'src/data.py',
        BASE/'kmeans.py',BASE/'common.py',FORE/'common.py',FORE/'task1.py']
result=dict(script=__file__,labels_used=False,checks=checks,ordered_sample_ids=all_ids,
 patients=all_patients,patient_counts=dict(Counter(all_patients)),
 ordered_ids_sha256=hashlib.sha256(json.dumps(all_ids,separators=(',',':')).encode()).hexdigest(),
 input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
 source_checkpoint=inv['seed0']['path'],elapsed_seconds=elapsed())
dump('verification.json',result)
note('Feature verification: `audit/scratch/final_baseline/verify.py` → `verification.json`. '
     'Both live 16-image batches match saved features (tolerance 1e-5); exact sample/patient order '
     'verified for all 2,341 slices. Annotation reads were prohibited by an audit hook. '
     'Checks: '+json.dumps(checks))
print(json.dumps(dict(checks=checks,elapsed_seconds=elapsed())),flush=True)
