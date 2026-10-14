"""Verify completed round provenance without any model fitting or label reads."""
from pathlib import Path
import json,hashlib,datetime,subprocess
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
calhash=hashlib.sha256((OUT/'calibration.json').read_bytes()).hexdigest();runhash=hashlib.sha256((OUT/'run.py').read_bytes()).hexdigest()
rows=[];previous_end=0;fixed=None
for candidate in ['D','E']:
 for seed in [0,1,2]:
  name=f'fixes4_{candidate}_seed{seed}';folder=OUT/'experiments'/name
  r=json.loads((folder/'result.json').read_text());p=json.loads((folder/'provenance.json').read_text());overlap=json.loads((OUT/f'{name}_process_check.json').read_text())
  assert overlap['clear'] and not overlap['other_python_processes']
  start=datetime.datetime.fromisoformat(overlap['timestamp_utc']).timestamp()
  assert start>=previous_end,'Runs overlapped according to saved process check/result timestamps'
  previous_end=(folder/'result.json').stat().st_mtime
  assert p['calibration_sha256']==calhash and p['runner_sha256']==runhash
  assert r['batches']<=500 and r['optimizer_steps']==3*r['batches'] and r['scheduler_T_max']==5000
  assert r['linear_parameters_unchanged'] and r['train_mask_reads']==0 and r['evaluated_images']==472
  if r['batches']==500:assert set(r['checkpoints'])=={'100','250','500'}
  assert all(Path(x).is_file() and Path(x).stat().st_size>1000000 for x in r['checkpoints'].values())
  assert Path(r['overlay']).is_file()
  with Image.open(r['overlay']) as image:image.verify()
  if fixed is None:fixed=r['fixed_samples']
  assert fixed==r['fixed_samples']
  rows.append(dict(name=name,batches=r['batches'],early_stopped=r['early_stopped'],checkpoint_count=len(r['checkpoints']),checkpoint_sizes={k:Path(v).stat().st_size for k,v in r['checkpoints'].items()},monitors=r['monitors'],elapsed_seconds=r['elapsed_seconds']))
assert not subprocess.check_output(['git','diff','e3c3ab0e396c194a7314d64546c81e9ba4973b72','--','src'],cwd=ROOT,text=True)
result=dict(script=__file__,round=rows,strictly_sequential_timestamps=True,all_overlap_checks_clear=True,unchanged_runner_and_calibration_hashes=True,project_source_unchanged=True,labels_used=False,fixed_samples=fixed,total_checkpoints=sum(r['checkpoint_count'] for r in rows))
(OUT/'verify_round.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
