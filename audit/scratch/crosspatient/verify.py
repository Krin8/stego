from common import *
import subprocess
nn=json.loads((OUT/'neighbours.json').read_text());summary=json.loads((OUT/'summary.json').read_text());checks={}
for split,k in [('val',30),('train',29)]:
 calpath=OUT/f'calibration_{split}.json';c=json.loads(calpath.read_text());p=json.loads((OUT/f'purity_{split}.json').read_text());records=json.loads((OUT/f'{split}_records.json').read_text());anchors=json.loads((OUT/f'{split}_anchors.json').read_text())
 assert hashlib.sha256(calpath.read_bytes()).hexdigest()==p['calibration_sha256']
 assert not c['labels_used'] and not nn[split]['labels_used']
 assert c['k_nonself']==p['k_nonself']==k
 current=np.load(OUT/f'{split}_current30.npy');cached=np.load(OUT/f'{split}_cached29.npy');cross=np.load(OUT/f'{split}_cross30.npy');pid=np.array([r['patient_id'] for r in records])
 assert all(set(a[:29])==set(b) for a,b in zip(current,cached))
 assert np.all(pid[cross]!=pid[:,None]) and np.all(current!=np.arange(len(current))[:,None])
 assert len(anchors)==(472 if split=='val' else 200) and (split=='val' or len(set(pid[anchors]))>=12)
 if split=='train':assert all('/train/labels/' in x for x in p['mask_paths'])
 else:assert all('/val/labels/' in x for x in p['mask_paths'])
 for variant in ['current','cross']:
  rows=p[variant];r=rows['knn/all'];assert r['image_pair_draws']==len(anchors)*k
  for key,row in rows.items():
   for mode in ['random_unfiltered','attracted']:
    s=row[mode];assert sum(s['counts_BB_BL_LB_LL'])==s['pairs']
    if s['pairs']:assert abs(s['same_label']+s['mixed']-1)<1e-12
  assert rows['knn/same_slice']['image_pair_draws']==0
  if variant=='cross':assert rows['knn/same_patient']['image_pair_draws']==0
  assert .73<r['attraction_rate']<.77
 assert .13<p['current']['negative/all']['attraction_rate']<.17
 assert p['elapsed_seconds']<570 and c['elapsed_seconds']<570
 validation_prefix_difference=None
 if split=='val':
  a=p['prefix_sensitivity']['29']['current']['knn/all']['attracted']['target_liver_given_anchor_liver'];b=p['prefix_sensitivity']['29']['cross']['knn/all']['attracted']['target_liver_given_anchor_liver'];validation_prefix_difference=100*(b-a)
 checks[split]=dict(anchors=len(anchors),patients=len(set(pid[anchors])),mask_count=p['mask_count'],calibration_hash_unchanged=True,current_positive_attraction=p['current']['knn/all']['attraction_rate'],cross_positive_attraction=p['cross']['knn/all']['attraction_rate'],negative_attraction=p['current']['negative/all']['attraction_rate'],prefix29_cross_minus_current_liver_target_pp=validation_prefix_difference,calibration_seconds=c['elapsed_seconds'],purity_seconds=p['elapsed_seconds'])
assert nn['train']['elapsed_seconds']<570
assert not subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=ROOT,text=True).strip()
for path in OUT.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
report=(ROOT/'audit/CROSSPATIENT.md').read_text();assert 'VERDICT_PLACEHOLDER' not in report
result=dict(script=__file__,status='PASS',cohorts=checks,no_tracked_source_changes=True,elapsed_seconds=elapsed());dump('verification.json',result);note('Final checks: `verify.py` → `verification.json`; '+json.dumps(result));print(json.dumps(result,indent=2))
