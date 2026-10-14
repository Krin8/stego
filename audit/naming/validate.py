from common import *
import subprocess
m=sealed_manifest();ledger=json.loads((OUT/'stageA_ledger.json').read_text());assert ledger['mask_or_label_files_read']==0 and ledger['prior_label_mapping_or_score_files_read']==0 and not ledger['blocked_attempts']
emb=np.load(OUT/'crop_embeddings.npy');text=np.load(OUT/'text_embeddings.npy');assert np.isfinite(emb).all() and np.isfinite(text).all()
keys=json.loads((OUT/'crop_keys.json').read_text());lookup={tuple(k):i for i,k in enumerate(keys)};rr=json.loads((OUT/'image_records.json').read_text())
checks=[]
for fit in ['inductive','transductive']:
 for seed in [0,1,2]:
  key=f'{fit}_s{seed}';n=json.loads((OUT/f'{key}_naming.json').read_text());score=json.loads((OUT/f'{key}_scores.json').read_text())
  assert sha(OUT/f'{key}_centers.npy')==n['centers_source_sha256']
  for nn in n['nearest']:
   assert len(nn)==200
   order=[(v['distance'],v['image_index'],v['patch_index']) for v in nn];assert order==sorted(order)
   if fit=='inductive':assert all(rr[v['image_index']]['split']=='train' for v in nn)
  averages=np.stack([emb[[lookup[(v['image_index'],v['patch_index'])] for v in nn]].mean(0) for nn in n['nearest']]);averages/=np.linalg.norm(averages,axis=1,keepdims=True);sim=averages@text.T
  assert np.max(np.abs(sim-np.array(n['N1']['cosine'])))<1e-6
  assert int(sim[:,0].argmax())==n['N1']['selected_cluster']
  possible=[x for x in n['N2']['clusters'] if x['centroid_x']<.5 and x['centroid_y']<.6 and 40<=x['mean_HU']<=180];chosen=max(possible,key=lambda x:x['area'])['cluster'] if possible else None;assert chosen==n['N2']['selected_cluster']
  assert n['N3']['selected_cluster']==(n['N1']['selected_cluster'] if n['N1']['selected_cluster']==chosen else None)
  for rule in ['N1','N2','N3']:
   assert n[rule]['selected_cluster']==m['selections'][key][rule]==score['rules'][rule]['selected_cluster']
   if n[rule]['selected_cluster'] is None:
    for variant in ['hard','F1']:
     for res in ['224','512']:
      for pat,r in score['rules'][rule]['results'][variant][res]['patients'].items():assert r['pooled']['tp']==r['pooled']['fp']==0
  # Label-assisted regression: same pre-existing majority reference must reproduce.
  if fit=='inductive':
   ref=json.loads((FINAL/f'inductive_seed{seed}_results.json').read_text())
   for var in ['hard','F1']:
    for res in ['224','512']:
     for pat,v in ref['results'][var]['patients'].items():
      actual=score['rules']['reference_majority']['results'][var][res]['patients'][pat]['pooled']
      for met in ['iou','dice']:assert actual[met]==v['pooled'][res][met]
  else:
   ref=json.loads((ROOT/f'audit/improve2/seed{seed}.json').read_text())
   for var,oldvar in [('hard','B2_hard'),('F1','F1')]:
    for res in ['224','512']:
     for pat,v in ref['variants'][oldvar]['results'][res]['patients'].items():
      actual=score['rules']['reference_majority']['results'][var][res]['patients'][pat]['pooled']
      for met in ['iou','dice']:assert actual[met]==v[met]
  checks.append(dict(fit=fit,seed=seed,selection_verified=True,reference_exact=True))
state=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),diff=subprocess.check_output(['git','diff'],cwd=ROOT,text=True),status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True));assert state['diff']==''
dump('validation.json',dict(producer='validate.py',manifest_sha256=sha(OUT/'stageA_manifest.json'),all_stageA_file_hashes_unchanged=True,stageA_zero_label_reads=True,all_selections_verified=True,abstentions_empty=True,reference_regression_checks=checks,source_state=state))
print('VALIDATION PASS')
