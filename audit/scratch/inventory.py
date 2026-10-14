"""Inventory all pre-existing paths; no content writes except AUDIT.md."""
from pathlib import Path
from collections import Counter
import hashlib
root=Path(__file__).resolve().parents[2]
def role(p):
    s=p.as_posix()
    if s.startswith('.git/'): return 'Git history/metadata; not pipeline input'
    if '__pycache__' in s: return 'generated Python cache; not provenance of current source'
    if s.startswith('Dataset/'): return 'dataset artifact; modality/split/image or label determined by directory'
    if s.startswith('logs/'): return 'historical training logs, configs, summaries or logged figures'
    if s.startswith('checkpoints/'): return 'saved model artifact; producing run must be established'
    if s.startswith('models/'): return 'model/artifact storage; inspect file type before treating as weights'
    if s.startswith(('results/','correspondence_visualization/')): return 'visual result; not executable pipeline'
    if s.startswith('runs/'): return 'TensorBoard event history'
    if s.startswith(('dino/','picie/')): return 'duplicate legacy backbone code'
    if '/tests/' in s: return 'test; not training entrypoint'
    if '/configs/' in s: return 'entrypoint configuration'
    return 'source/support; specific roles in Phase 2 narrative'
lines=['\n<details><summary>Exhaustive path inventory (including hidden and ignored paths; role inherited per group)</summary>\n']
files=[]
for directory in sorted([root]+[p for p in root.rglob('*') if p.is_dir() and 'audit' not in p.relative_to(root).parts]):
    rel=directory.relative_to(root)
    entries=sorted(p for p in directory.iterdir() if p.is_file())
    lines.append(f'\n**`{rel}/` — {role(rel)}**')
    for p in entries:
        r=p.relative_to(root)
        if r.parts[0]=='audit':continue
        files.append(p)
        lines.append(f'- `{p.name}` ({p.stat().st_size} B)')
lines.append('\n</details>\n')
with (root/'audit/AUDIT.md').open('a') as f:f.write('\n'.join(lines))
print('inventory file count',len(files))
print('top-level counts',Counter(p.relative_to(root).parts[0] for p in files))
for p in files:
    r=p.relative_to(root)
    if p.suffix in ('.ckpt','.pth','.bin','.npz','.nii') or p.name.endswith('.nii.gz') or p.name in ('wandb-summary.json','config.yaml','hparams.yaml','wandb-metadata.json'):
        print(r,p.stat().st_size)
