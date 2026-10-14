"""Report every fixed arm/seed/rule; no label-driven winner selection."""
import os
os.environ['DENSE_STAGE']='evaluation'
from runtime import *
x=read(HERE/'metrics/results.json');lines=['# Dense correspondence experiment — results','', 'All configurations below are reported; none is selected using evaluation scores. Refer to ARCHITECTURE.md, TRAINING_PROTOCOL.md and LABEL_USAGE.md. Ground truth was opened only after all six prediction manifests were sealed.','', '| Arm | Seed | Rule | Cluster | Scale | Aggregation | Output | IoU mean ± SD [min–max] % | Dice mean ± SD [min–max] % |','|---|---|---|---|---|---|---|---|---|']
def fmt(s):return f"{100*s['mean']:.3f} ± {100*s['std']:.3f} [{100*s['min']:.3f}–{100*s['max']:.3f}]"
for key,r in x['results'].items():
 for rule,v in r['rules'].items():
  for kind in ['hard','F1']:
   for res in ['224','512']:
    for agg in ['pooled','mean_slice']:
     q=v[kind][res]['summary'][agg];lines.append(f"| {r['arm']} | {r['seed']} | {rule} | {r['selected'][rule]} | {res} | {agg} | {kind} | {fmt(q['iou'])} | {fmt(q['dice'])} |")
lines+=['','## Exactly matched baseline comparison','', 'Patient-pooled macro scores at224. Naming is independent per fit; abstentions are empty masks. The baseline source remains the original sealed naming StageB, with the same validation patients and F1 parameters.','', '| Arm | Seed | Rule | Baseline hard IoU/Dice | New hard IoU/Dice | Baseline F1 IoU/Dice | New F1 IoU/Dice |','|---|---|---|---|---|---|---|']
for key,r in x['results'].items():
 b=read(BASE/f'inductive_s{r["seed"]}_scores.json')
 for rule in ['N1','N2','N3']:
  cells=[]
  for kind in ['hard','F1']:
   for q in [b['rules'][rule]['results'][kind]['224']['summary']['pooled'],r['rules'][rule][kind]['224']['summary']['pooled']]:cells.append(' / '.join(f'{100*q[m]["mean"]:.3f}' for m in ['iou','dice']))
  lines.append('| '+' | '.join([r['arm'],str(r['seed']),rule,*cells])+' |')
lines+=['','## Failure diagnostics','', '| Arm | Seed | Rule | Output | Tiny-liver slices | Tiny native mean-slice IoU % | Empty slices | Empty with false positives |','|---|---|---|---|---|---|---|---|']
for r in x['results'].values():
 for rule,v in r['rules'].items():
  for kind in ['hard','F1']:
   q=v[kind]['diagnostics'];lines.append('| '+' | '.join(map(str,[r['arm'],r['seed'],rule,kind,q['tiny_slices'],100*q['tiny_mean_iou'] if q['tiny_mean_iou'] is not None else 'N/A',q['empty_slices'],q['empty_with_fp']]))+' |')
lines+=['','## Actual seed-to-seed mask agreement','', '| Arm | Rule | Output | Seeds | Prediction-to-prediction Dice % | Different native pixels |','|---|---|---|---|---|---|']
for r in x['seed_stability']:lines.append('| '+' | '.join(map(str,[r['arm'],r['rule'],r['kind'],r['seeds'],100*r['interseed_dice'],r['different_pixels']]))+' |')
lines+=['','## Density diagnostic','', 'The statistic below measures disagreement with nearest enlargement of a28×28 within-cell majority projection of the new categorical map. Nonzero demonstrates categorical detail beyond that restricted representation. It does not establish anatomically correct boundaries, nor prove that new high-frequency semantic information is available in frozen tokens.','']
for key,r in x['results'].items():lines.append(f"- {key}: {100*r['density_diagnostic']['disagreement_with_nearest_28_cell_majority']:.6f}% categorical disagreement on32 fixed uniformly spaced validation slices (no labels used).")
lines+=['','## Figures','']
for path in x['figures']:lines.append(f'- [{Path(path).name}]({path})')
lines+=['','## Interpretation constraints','', 'This is a bounded six-run experiment, not optimization for liver IoU. The geometry-matched InfoNCE objective differs from literal STEGO correlation distillation. Token-only learned upsampling can produce fine variation but does not automatically recover anatomical detail unavailable in the28×28 input. A positive conclusion requires both the real saved figures and held-out scoring; fine-looking masks alone do not establish pixel-level semantics. The fixed spatial sample for k-means is a tractability compromise shared by both new arms, whereas the old baseline used all patch features. All20 patients informed historical choices. No unseen-cohort generalization or full-convergence claim follows. A negative result must remain in this report.']
(HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('REPORT READY')
