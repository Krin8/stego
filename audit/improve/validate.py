"""Completed-run integrity checks; no fitting, no parameter changes."""
from pipeline import *
summary=json.loads((OUT/'summary.json').read_text());assert not summary['pending']
meta=metadata();patients=set(meta['patient_counts']);assert len(patients)==20
assert len(meta['records'])==2341 and len({r['sample_id'] for r in meta['records']})==2341
assert len(json.loads((OUT/'figure_selection.json').read_text())['eight_indices'])==8
for row in summary['rows']:
 key=f"{row['variant']}_{row['resolution']}_{row['mapping']}";values=summary['per_patient'][key]
 assert set(values)==patients
 for metric in ['iou','dice']:
  stat=patient_stats([v[metric] for v in values.values()])
  assert stat==row[metric]
for variant in ['B0','B1','B2_336','B2_448','B3','B4_soft','B4_guided','B5_2D','B5_3D','B6']:
 for seed in [0,1,2]:
  result=json.loads((OUT/f'{variant}_s{seed}_scores.json').read_text())
  for resolution,rr in result['results'].items():
   for patient,fold in rr['folds'].items():
    assert set(fold['mapping_patients'])==patients-{patient}
    for m in ['majority','optimal']:
     s=fold[m];assert 0<=s['iou']<=1 and 0<=s['dice']<=1
     assert abs(s['iou']-s['tp']/max(1,s['tp']+s['fp']+s['fn']))<1e-12
     assert abs(s['dice']-2*s['tp']/max(1,2*s['tp']+s['fp']+s['fn']))<1e-12
  if variant=='B0':assert result['baseline_max_patient_metric_difference']==0
  if variant in ['B0','B1','B2_336','B2_448','B3','B6']:
   fit=json.loads((OUT/f'{variant}_s{seed}_fit.json').read_text());params=fit['parameters']
   assert params['n_clusters']==8 and params['n_init']==3 and params['max_iter']==100
   assert params['random_state']==seed and params['tol']==1e-4 and params['algorithm']=='lloyd'
   assert not fit['labels_used']
   for res in ['224','512']:
    assert sum(np.asarray(c).sum() for c in result['results'][res]['patient_confusions'].values())==2341*int(res)**2
    old=json.loads((OUT/f'B0_s0_scores.json').read_text())['results'][res]['patient_confusions']
    for patient,c in result['results'][res]['patient_confusions'].items():
     assert np.array_equal(np.asarray(c).sum(0),np.asarray(old[patient]).sum(0))
source=json.loads((FINAL/'verification.json').read_text())['input_sha256'];unchanged={}
for path,value in source.items():
 h=hashlib.sha256()
 with (ROOT/path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 unchanged[path]=(h.hexdigest()==value)
assert all(unchanged.values()),unchanged
dump('validation.json',dict(script=__file__,all_checks_pass=True,summary_rows=len(summary['rows']),
 heldout_labels_excluded_from_mapping=True,all_pixels_scored=True,
 original_source_features_scoring_helpers_sha256_unchanged=unchanged,elapsed_seconds=elapsed()))
print('PASS',len(summary['rows']),'summary rows; all patient folds and original feature/source hashes checked',flush=True)
