from common import *
rs=[json.loads((OUT/f'seed{s}.json').read_text()) for s in [0,1,2]]
# Exact Euclidean distance check and empty-mask conventions.
x=np.zeros((12,12),bool);y=x.copy();x[1,1]=True;y[4,5]=True
d,case=boundary_distances(x,y);assert np.array_equal(d,[5,5]) and case=='both_present'
assert boundary_summary([d])['hd95']==5 and boundary_summary([d])['mean_surface_distance']==5
d,case=boundary_distances(x,np.zeros_like(x));assert case=='one_empty' and np.isinf(d).all()
assert boundary_summary([d])['mean_surface_distance'] is None
assert boundary_summary([np.array([0,1,2,3,4,5,6,7,8,9,np.inf])])['hd95'] is None
meta=metadata();patients=set(meta['patient_counts']);gt=np.load(OLD/'scoring_masks_native.npy',mmap_mode='r')
assert len(gt)==2341 and sum(meta['patient_counts'].values())==2341
for seed,r in enumerate(rs):
 assert r['forward_passes']==0 and r['kmeans_fits']==0
 for v in ['B2_hard','B5_2D','F1','F2']:
  rr=r['variants'][v]
  if v in ['B2_hard','B5_2D']:assert rr['reference_max_metric_difference']==0
  for res,ev in rr['results'].items():
   assert set(ev['patients'])==patients
   for metric in ['iou','dice']:assert patient_stats([s[metric] for s in ev['patients'].values()])==ev['summary'][metric]
  assert len(rr['slice_scores'])==2341
 f1=np.load(OUT/f'F1_s{seed}_pred.npy',mmap_mode='r');f2=np.load(OUT/f'F2_s{seed}_pred.npy',mmap_mode='r')
 for p in patients:
  ids=sorted([i for i,x in enumerate(meta['records']) if x['patient']==p],key=lambda i:meta['records'][i]['physical_position'])
  a=np.asarray(f1[ids],dtype=bool)
  # Independent Boolean majority-of-three, with within-patient endpoint replication.
  prev=np.concatenate([a[:1],a[:-1]]);nxt=np.concatenate([a[1:],a[-1:]])
  expected=(prev&a)|(prev&nxt)|(a&nxt)
  assert np.array_equal(expected,f2[ids])
inputs=[OLD/f'B2_448_s{s}_scores.json' for s in range(3)]+[OLD/f'B4_s{s}_weights.npy' for s in range(3)]+[OLD/'postprocess.py',OLD/'pipeline.py',OLD/'scoring_masks_native.npy',OLD/'scoring_masks_224.npy']
hashes={}
for p in inputs:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 hashes[str(p)]=h.hexdigest()
dump('validation.json',dict(script=__file__,reference_reproduction_exact=True,all_patient_statistics_verified=True,
 temporal_filter_independent_check=True,surface_distance_synthetic_check=True,forward_passes=0,kmeans_fits=0,
 input_sha256=hashes,elapsed_seconds=elapsed()))
print('PASS references, patient statistics, temporal endpoints, surface units, no inference/refitting',flush=True)
