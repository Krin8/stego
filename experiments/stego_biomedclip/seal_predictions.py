"""Require all three independent predictions before allowing label reads."""
from runtime import *
from datetime import datetime,timezone
assert not (HERE/'predictions/stageA_manifest.json').exists()
runs={};outputs={}
for seed in [0,1,2]:
    path=HERE/f'predictions/seed{seed}/prediction_manifest.json'
    m=read(path);assert m['seed']==seed and len(m['indices'])==472 and not m['labels_read']
    for name,digest in m['outputs'].items():
        f=path.parent/name;assert sha(f)==digest
        outputs[str(f.relative_to(HERE))]=digest
    for name in ['naming.json','fit.json','prediction_manifest.json']:
        f=path.parent/name;outputs[str(f.relative_to(HERE))]=sha(f)
    runs[str(seed)]=dict(manifest=str(path.relative_to(HERE)))
dump('predictions/stageA_manifest.json',dict(runs=runs,outputs=outputs,
    completed_utc=datetime.now(timezone.utc).isoformat(),
    config_sha256=sha(HERE/'configs/locked.json'),labels_read=False))
(HERE/'predictions/stageA_manifest.sha256').write_text(sha(HERE/'predictions/stageA_manifest.json')+'\n')
ledger('logs/seal_access.json')
