"""Report every screen monitor; no annotation/score files are read."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'experiments/stego_colocated'
read=lambda p:json.loads(p.read_text())
decision=read(P/'screen_decision.json')
lines=['\n### Amended Task 2: completed screen\n',
 'Producers: `experiments/stego_colocated/train.py` and `run_round.py screen`; raw per-batch logs, initial monitors, per-run timing, overlap snapshots, complete.json and final checkpoints are in each arm’s `stage1/`. Monitor std is the mean per-channel spatial std of L2-normalized codes; cosine averages off-diagonal dot products of per-image mean normalized patch codes. Cluster usage comes from fixed train-only K8 monitor clustering, not the Stage-A fit. All numbers below are label-free.\n',
 '| Arm | Seed | Batch | Code std | Cross-image cosine | Usage C0–C7 (%) | Healthy |',
 '| --- | ---: | ---: | ---: | ---: | --- | --- |']
timings=[]
for arm in ['C448','P1']:
    for seed in [0,1,2]:
        d=P/arm/'stage1/logs';r=read(d/f'seed{seed}_complete.json')
        rows=[dict(batch=0,**r['initial']),*r['monitors']]
        for row in rows:
            usage=', '.join(f'{100*u:.2f}' for u in row['cluster_usage'])
            lines.append(f"| {arm} | {seed} | {row['batch']} | {row['code_std']:.8f} | {row['cross_image_cosine']:.8f} | {usage} | {row['healthy']} |")
        t=read(d/f'seed{seed}_timing.json');timings.append((arm,seed,t['first10_seconds'],t['projected_run_seconds'],r['seconds']))
lines += ['\n| Arm | Seed | First 10 batches (s) | Projected 150 (s) | Actual run (s) |','| --- | ---: | ---: | ---: | ---: |']
for arm,seed,a,b,c in timings:lines.append(f'| {arm} | {seed} | {a:.2f} | {b:.2f} | {c:.2f} |')
lines += ['\n| Arm | Seed 0 screen | Seed 1 screen | Seed 2 screen | Advances |','| --- | --- | --- | --- | --- |']
for arm,d in decision.items():lines.append(f"| {arm} | {d['seeds']['0']} | {d['seeds']['1']} | {d['seeds']['2']} | {d['passed']} |")
lines += ['\nEach seed must be healthy at 50, 100 and 150 AND have batch-150 cosine below its own initialization. Arm decisions are the conjunction of all three seeds. State serialization passed an exact synthetic next-update replay; actual runs started fresh and did not resume. Overlap checks ran before each seed and the same package lock prevented overlapping trainers. Validation labels remain closed.\n']
text='\n'.join(lines);(ROOT/'audit/colocated/screen_summary.md').write_text(text)
report=ROOT/'audit/FIXES6.md';old=report.read_text();assert '### Amended Task 2: completed screen' not in old;report.write_text(old+text)
print(json.dumps(decision))
