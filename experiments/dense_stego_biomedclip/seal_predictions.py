"""Freeze all six experiment outputs before evaluation can read labels."""
import os
os.environ['DENSE_STAGE']='inference'
from runtime import *
from datetime import datetime,timezone
path=HERE/'predictions/stageA_manifest.json';assert not path.exists();c=read(HERE/'configs/locked.json');runs={}
for arm in c['arms']:
 for seed in c['seeds']:
  out=HERE/f'predictions/seed{seed}/{arm}';m=read(out/'prediction_manifest.json');assert m['arm']==arm and m['seed']==seed
  for rel,digest in m['outputs'].items():assert sha(out/rel)==digest
  runs[f'{arm}_seed{seed}']=dict(manifest=str((out/'prediction_manifest.json').relative_to(HERE)),manifest_sha256=sha(out/'prediction_manifest.json'),selected=m['selected'])
outputs={str(p.relative_to(HERE)):sha(p) for p in HERE.glob('*.py')};outputs['configs/locked.json']=sha(HERE/'configs/locked.json')
for p in (HERE/'predictions').rglob('*.json'):outputs[str(p.relative_to(HERE))]=sha(p)
for p in (HERE/'checkpoints').rglob('best.pt'):outputs[str(p.relative_to(HERE))]=sha(p)
dump('predictions/stageA_manifest.json',dict(schema_version=1,utc=datetime.now(timezone.utc).isoformat(),runs=runs,outputs=outputs,config_sha256=sha(HERE/'configs/locked.json'),baseline_sha256=baseline_seal(),labels_read=False));(HERE/'predictions/stageA_manifest.sha256').write_text(sha(path)+'\n');ledger('logs/seal_access.json');print('SEALED',sha(path),flush=True)
