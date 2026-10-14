"""Read-only source history, normalization and run-record checks; no labels."""
import ast, collections, hashlib, json, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
import torch
torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
P = ROOT / 'experiments/stego_colocated'
state = torch.load(P / 'P1/stage2/checkpoints/seed0/final.pt', map_location='cpu', weights_only=True, mmap=True)
target = state['source_sha256']['train.py']
commits = subprocess.check_output(['git', 'log', '--all', '--format=%H', '--', 'experiments/stego_colocated/train.py'], cwd=ROOT, text=True).splitlines()
matches = []
for commit in commits:
    source = subprocess.check_output(['git', 'show', commit + ':experiments/stego_colocated/train.py'], cwd=ROOT)
    if hashlib.sha256(source).hexdigest() == target:
        matches.append(commit)
        if len(matches) == 1:
            (OUT / 'checkpoint_train_source.txt').write_bytes(source)
records = json.loads((ROOT / 'audit/naming/image_records.json').read_text())
splits = {split: dict(sorted(collections.Counter(r['patient'] for r in records if r['split'] == split).items(), key=lambda x: int(x[0]))) for split in ['train', 'val']}
monitors = []
for seed in range(3):
    p = P / f'P1/stage2/logs/seed{seed}_complete.json'
    run = json.loads(p.read_text())
    last = run['monitors'][-1]
    monitors.append(dict(seed=seed, source=str(p.relative_to(ROOT)), batches=run['batches'], final=last))
source = ast.parse((ROOT / 'src/eval_segmentation.py').read_text())
call = next(n for n in ast.walk(source) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'TF.normalize')
eval_mean, eval_std = [ast.literal_eval(x) for x in call.args[1:]]
cfg = json.loads((P / 'P1/stage2/configs/locked.json').read_text())['preprocessing']
pixel_examples = []
for value in [0., .5, 1.]:
    expected = [(value-m)/s for m,s in zip(cfg['mean'],cfg['std'])]
    actual = [(value-m)/s for m,s in zip(eval_mean,eval_std)]
    pixel_examples.append(dict(windowed_intensity=value, expected_BiomedCLIP=expected, current_eval=actual))
data = dict(producer=__file__, git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    checkpoint_train_source_sha256=target, matching_git_commits=matches,
    patient_slice_counts=splits, patient_overlap=sorted(set(splits['train']) & set(splits['val'])),
    final_monitors=monitors, normalization_examples=pixel_examples,
    labels_opened=False, training_steps=0)
(OUT / 'provenance_checks.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2))
