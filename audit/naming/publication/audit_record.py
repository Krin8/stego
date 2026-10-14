"""Recheck saved Stage A without reading masks or semantic reference files."""
import os,sys
from pathlib import Path
os.environ['NAMING_STAGE']='A';sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from datetime import datetime
m=sealed_manifest();ledger=json.loads((OUT/'stageA_ledger.json').read_text());b=json.loads((OUT/'stageB_started.json').read_text());assert m['final_sealed_utc']<b['utc'];assert b['manifest_sha256']==sha(OUT/'stageA_manifest.json');assert not ledger['blocked_attempts']
emb=np.load(OUT/'crop_embeddings.npy');txt=np.load(OUT/'text_embeddings.npy');lookup={tuple(k):i for i,k in enumerate(json.loads((OUT/'crop_keys.json').read_text()))};records=json.loads((OUT/'image_records.json').read_text());names={}
for fit in ['inductive','transductive']:
 for seed in range(3):
  key=f'{fit}_s{seed}';n=json.loads((OUT/f'{key}_naming.json').read_text());avg=np.stack([emb[[lookup[(x['image_index'],x['patch_index'])] for x in nn]].mean(0) for nn in n['nearest']]);avg/=np.linalg.norm(avg,axis=1,keepdims=True);sim=avg@txt.T
  assert np.max(np.abs(sim-np.asarray(n['N1']['cosine'])))<1e-6
  assert n['N1']['selected_cluster']==int(sim[:,0].argmax())
  for nn in n['nearest']:
   assert len(nn)==200
   assert fit=='transductive' or all(records[x['image_index']]['split']=='train' for x in nn)
  possible=[r for r in n['N2']['clusters'] if r['centroid_x']<.5 and r['centroid_y']<.6 and 40<=r['mean_HU']<=180];n2=max(possible,key=lambda r:r['area'])['cluster'] if possible else None;assert n2==n['N2']['selected_cluster'];n3=n2 if n2==n['N1']['selected_cluster'] else None;assert n3==n['N3']['selected_cluster'];assert {r:n[r]['selected_cluster'] for r in ['N1','N2','N3']}==m['selections'][key]
  names[key]=dict(temperature=n['temperature'],selections=m['selections'][key],liver_z_margin=n['N1']['liver_selection_margin'],body_stats=n['N2']['clusters'][n2] if n2 is not None else None,prompt_ranking=n['N1']['cluster_prompt_ranking'])
initial=(OUT/'stageA_initial.log').read_text();completed=(OUT/'stageA.log').read_text();assert 'AssertionError' in initial and 'SEALED' not in initial;assert all('GEOMETRY '+str(i) in completed for i in range(0,2341,400));assert 'SEALED' in completed
value=dict(producer='publication/audit_record.py',manifest_sha256=sha(OUT/'stageA_manifest.json'),sealed_file_count=len(m['output_sha256']),all_hashes_match=True,initial_seal_utc=m['sealed_utc'],final_seal_utc=m['final_sealed_utc'],stageB_start_utc=b['utc'],seconds_between_final_seal_and_B=(datetime.fromisoformat(b['utc'])-datetime.fromisoformat(m['final_sealed_utc'])).total_seconds(),recorded_A_mask_reads=ledger['mask_or_label_files_read'],recorded_A_prior_semantic_reads=ledger['prior_label_mapping_or_score_files_read'],recorded_A_opened_paths=len(ledger['opened_read_paths']),recorded_A_blocked_attempts=ledger['blocked_attempts'],naming_rederived_without_labels=True,correction_evidence='Failed initial log before seal; successful rerun logs raw geometry0..2000, nearest selection, embedding, six names and seal; frozen script always recomputes body geometry from DICOM. EXECUTION_NOTES records removal of extra hard-assignment restriction.',not_verified=['No historical OS-level file-open trace or independently timestamped append-only audit is available; records are local and self-generated, not cryptographic third-party attestation.','No independent complete read ledger was saved for the failed initial attempt. Absence of scored-result influence is supported by recorded control flow and notes, not independently provable from these files.','No untouched cohort or scanner was tested; prior development used cohort labels.'],names=names)
dump('publication/audit_verification.json',value);print('AUDIT VERIFIED',value['sealed_file_count'],value['seconds_between_final_seal_and_B'])
