"""Check completed artifacts, LOPO isolation, and unchanged feature/source inputs."""
from common import *
v=json.loads((OUT/'verification.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
assert all(sha(ROOT/p)==value for p,value in v['input_sha256'].items())
assert len(v['ordered_sample_ids'])==2341==sum(v['patient_counts'].values())
base_conf=None;checks=[]
for k in [8,16]:
 for seed in [0,1,2]:
  fit=json.loads((OUT/f'k{k}_s{seed}_fit.json').read_text())
  r=json.loads((OUT/f'k{k}_s{seed}_scores.json').read_text())
  assert fit['ordered_ids_sha256']==v['ordered_ids_sha256'] and not fit['labels_used']
  assert fit['centering']=='none' and fit['normalization']=='L2 per patch'
  assert fit['parameters']['n_init']==3 and fit['parameters']['max_iter']==100
  grids=np.load(OUT/f'k{k}_s{seed}_grids.npy',mmap_mode='r')
  assert grids.shape==(2341,14,14) and grids.min()>=0 and grids.max()<k
  label_totals={p:np.asarray(c).sum(0).tolist() for p,c in r['patient_confusions'].items()}
  if base_conf is None:base_conf=label_totals
  assert base_conf==label_totals
  for patient,q in r['lopo'].items():
   assert patient not in q['mapping_fit_patients']
   assert set(q['mapping_fit_patients'])==set(v['patient_counts'])-{patient}
   assert q['heldout_pixels']==v['patient_counts'][patient]*224*224
   assert q['mapping_fit_pixels']+q['heldout_pixels']==2341*224*224
   conf=np.asarray(r['patient_confusions'][patient])
   for mapping in ['optimal','majority']:
    assert scores(conf,q[mapping]['clusters'])==q[mapping]
  for mapping in ['optimal','majority']:
   for metric in ['iou','dice']:
    values=np.array([q[mapping][metric] for q in r['lopo'].values()])
    expected=dict(mean=values.mean(),std=values.std(ddof=1),min=values.min(),max=values.max(),median=np.median(values))
    for name,x in expected.items():assert abs(float(x)-r['summary'][mapping][metric][name])<1e-14
  checks.append(dict(k=k,seed=seed,folds=len(r['lopo']),pass_checks=True))
# Verify generalized exhaustive mapping against original K=8 and a padded K=16.
c=np.zeros((8,2),dtype=np.int64);c[0]=[80,0];c[1]=[0,20]
assert mapping_any_k(c)==mapping_report(c)
assert mapping_any_k(c)['optimal']['clusters']==[1]
assert mapping_any_k(np.pad(c,((0,8),(0,0))))['optimal']['clusters']==[1]
dump('validation.json',dict(script=__file__,checks=checks,
 source_and_feature_sha256_unchanged=True,patient_label_counts_consistent=True,
 heldout_labels_excluded_from_mapping=True,exhaustive_mapping_dimension_test=True,
 elapsed_seconds=elapsed()))
print('PASS: six fits, 120 folds, 240 scored mappings, unchanged feature/source hashes, exhaustive mapping dimension check.',flush=True)
