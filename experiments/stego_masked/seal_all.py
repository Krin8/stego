"""Single global gate: all passing candidates, all three final-state predictions."""
from runtime import *
from datetime import datetime,timezone
runs={};outputs={}
root=HERE
for seed in [0,1,2]:
    manifest=root/f'predictions/seed{seed}/prediction_manifest.json';m=read(manifest)
    assert not m['labels_read'] and len(m['indices'])==472
    complete=read(root/f'logs/seed{seed}_complete.json');assert complete['batches']==500 and not complete['resumed']
    assert m['checkpoint_sha256']==sha(root/f'checkpoints/seed{seed}/final.pt')
    for p in [root/f'checkpoints/seed{seed}/final.pt',root/'configs/locked.json',root/f'logs/seed{seed}_complete.json']:
        outputs[str(p.relative_to(PACKAGE))]=sha(p)
    for name,digest in m['outputs'].items():
        p=manifest.parent/name;assert sha(p)==digest
        outputs[str(p.relative_to(PACKAGE))]=digest
    for name in ['fit.json','naming.json','prediction_manifest.json']:
        p=manifest.parent/name;outputs[str(p.relative_to(PACKAGE))]=sha(p)
    runs[f'residual/seed{seed}']=str(manifest.relative_to(PACKAGE))
assert runs and not (PACKAGE/'all_predictions_sealed.json').exists()
dump('all_predictions_sealed.json',dict(runs=runs,outputs=outputs,labels_read=False,completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256={p.name:sha(p) for p in PACKAGE.glob('*.py')}))
(PACKAGE/'all_predictions_sealed.sha256').write_text(sha(PACKAGE/'all_predictions_sealed.json')+'\n')
ledger('logs/global_seal_access.json');print('All predictions sealed')
