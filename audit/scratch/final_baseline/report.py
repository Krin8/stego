"""Render all pre-declared rows, without selecting a seed or mapping."""
from common import *
verification=json.loads((OUT/'verification.json').read_text())
patients=sorted(verification['patient_counts'],key=int)
results={(k,s):json.loads((OUT/f'k{k}_s{s}_scores.json').read_text()) for k in [8,16] for s in [0,1,2]}
fits={(k,s):json.loads((OUT/f'k{k}_s{s}_fit.json').read_text()) for k,s in results}
def stats(v):
 v=np.asarray(v,dtype=float)
 return dict(mean=float(v.mean()),std=float(v.std(ddof=1)),min=float(v.min()),max=float(v.max()),median=float(np.median(v)))
summary=[]; averaged={}
for k in [8,16]:
 for mapping in ['optimal','majority']:
  av={patient:{metric:float(np.mean([results[k,s]['lopo'][patient][mapping][metric] for s in [0,1,2]]))
               for metric in ['iou','dice']} for patient in patients}
  averaged[f'k{k}_{mapping}']=av
  row=dict(k=k,mapping=mapping,seed='mean of 0/1/2',
           **{metric:stats([v[metric] for v in av.values()]) for metric in ['iou','dice']})
  summary.append(row)
dump('summary.json',dict(script=__file__,labels_used='Other 19 patients for mapping; held-out patient for scoring',
 summary=summary,seed_averaged_patients=averaged))
report=ROOT/'audit/FINAL_BASELINE.md'
protocol=report.read_text().split('## Progress')[0]
lines=[protocol,'## Final estimate\n',
 '**K-means fitting uses no labels. Mapping and scoring use labels on separate patients (leave-one-patient-out).** '
 'The primary estimate is K=8 uncentered with IoU-optimal mapping. K=16 and strict-majority mapping are declared secondary analyses. '
 'All numbers below are percentages unless explicitly identified as counts, time or inertia. '
 'The six fits completed before annotation-based scoring began.\n',
 'Summary producer: `audit/scratch/final_baseline/report.py` → `summary.json`; each input is '
 '`score.py --k K --seed S` → `kK_sS_scores.json`. Every fold pools that patient’s pixels before scoring; '
 'summary statistics give each patient equal weight. Std is sample standard deviation across patients, not standard error or a confidence interval.\n']
def table(rows):
 text=['| K | Seed | Mapping | IoU mean ± std | IoU min / median / max | Dice mean ± std | Dice min / median / max |',
       '|---|---|---|---:|---:|---:|---:|']
 for r in rows:
  i,d=r['iou'],r['dice']
  text.append(f"| {r['k']} | {r['seed']} | {r['mapping']} | {100*i['mean']:.3f} ± {100*i['std']:.3f} | {100*i['min']:.3f} / {100*i['median']:.3f} / {100*i['max']:.3f} | {100*d['mean']:.3f} ± {100*d['std']:.3f} | {100*d['min']:.3f} / {100*d['median']:.3f} / {100*d['max']:.3f} |")
 return '\n'.join(text)+'\n'
lines+=['### Seed-averaged patient estimates\n',table(summary),'### Each seed separately\n']
rows=[]
for (k,s),r in results.items():
 for mapping in ['optimal','majority']:rows.append(dict(k=k,seed=s,mapping=mapping,**r['summary'][mapping]))
lines.append(table(rows))
lines+=['## Per-patient scores and fitted mappings\n',
 'Producer: `score.py`; exact TP/FP/FN, confusion matrices, held-out pixel counts, mapping-fit patient IDs '
 'and selected liver-cluster subsets are preserved in the six `kK_sS_scores.json` files. '
 'Both mappings use only the other 19 patients; the held-out label does not select a subset. '
 'IoU-optimal mapping exhaustively evaluates every subset (256 for K=8; 65,536 for K=16); '
 'ties follow the first integer bitmask, as in FORENSICS. Strict majority maps a cluster to liver '
 'only when its other-19 liver count exceeds its background count. Empty clusters and ties map to background.\n']
for (k,s),r in results.items():
 lines+=[f'### K={k}, seed={s}\n',
 '| Patient | Slices | Optimal IoU | Optimal Dice | Liver clusters (optimal) | Majority IoU | Majority Dice | Liver clusters (majority) |',
 '|---|---:|---:|---:|---|---:|---:|---|']
 for patient in patients:
  q=r['lopo'][patient];o=q['optimal'];m=q['majority']
  lines.append(f"| {patient} | {q['slice_count']} | {100*o['iou']:.3f} | {100*o['dice']:.3f} | {o['clusters']} | {100*m['iou']:.3f} | {100*m['dice']:.3f} | {m['clusters']} |")
 lines.append('')
lines+=['## Cohort, feature provenance and verification\n',
 '`verify.py` → `verification.json`: all 2,341 sample IDs are unique; all 20 CT patients present; '
 'former train/val patient sets are disjoint. Counts below use metadata only, no annotations. '
 'Exact ordered IDs and source/file SHA256 values are saved in verification.json.\n',
 '| Former split | Patient | Slices |','|---|---|---:|']
for split in ['train','val']:
 for patient,n in sorted(verification['checks'][split]['patients'].items(),key=lambda x:int(x[0])):
  lines.append(f'| {split} | {patient} | {n} |')
lines+=['',
 'Feature producers: `audit/scratch/baseline/extract.py` (all 1,869 former-train slices) and '
 '`audit/scratch/forensics/task1.py` (all 472 former-validation slices; independently verified by BASELINE extract.py). '
 'Read-only arrays: `audit/scratch/baseline/train_features.npy` and `audit/scratch/forensics/val_features.npy`. '
 'The new `verify.py` called the project’s actual extractor on 16 images from each split: both maximum absolute differences were exactly 0. '
 'It also verified the complete dataset order via the actual loader. This is a live 32-image spot-check plus existing full-extraction provenance; '
 'the full cohort was not re-extracted.\n',
 'The project loader supplies the approved CT preprocessing and masks (`src/chaos_preprocessing.py`, `src/data.py`): '
 'HU window [-150,250], native orientation/spacing derivatives, RGB replication, BiomedCLIP mean/std once, '
 '224×224 scoring resolution and nearest-neighbour masks. Features are the extractor’s first output (`src/modules.py`), '
 'normalized final-block patch tokens with CLS excluded, 768 channels on 14×14. The learned STEGO code/head/probe '
 'does not supply these features or assignments. New scripts only perform per-patch L2 normalization, no centering, '
 'then preserve BASELINE KMeans settings and FORENSICS confusion/mapping functions.\n',
 'Fit producer: `fit.py --k K --seed S` → `kK_sS_fit.json`, `_grids.npy`, `_centers.npy`. '
 'All 458,836 patch vectors were used, ordered former-train then former-validation. Annotation reads were blocked '
 'during verification/fitting by the Python audit hook. No kNN cache was read or regenerated. CPU only; '
 'no neural training or source/data/checkpoint modifications.\n',
 '`validate.py` → `validation.json`: all six fits, 120 held-out folds and 240 mapped scores passed checks; '
 'held-out patient exclusion, consistent label counts, statistics and exhaustive mapping dimensions were verified. '
 'Feature/source input SHA256 hashes are unchanged.\n',
 '| K | Seed | Iterations | Inertia | Fit seconds | Scoring seconds | Fit warnings |',
 '|---|---|---:|---:|---:|---:|---|']
for key,f in fits.items():
 lines.append(f"| {key[0]} | {key[1]} | {f['iterations']} | {f['inertia']:.6f} | {f['elapsed_seconds']:.2f} | {results[key]['elapsed_seconds']:.2f} | {f['warnings']} |")
primary=summary[0]
lines+=['', '## Verdict and limits\n',
 f"The pre-declared primary K=8 uncentered baseline achieves **{100*primary['iou']['mean']:.3f}% mean liver IoU** "
 f"and **{100*primary['dice']['mean']:.3f}% mean liver Dice**, averaging seeds within patient then averaging all 20 patients. "
 f"Between-patient sample standard deviations are {100*primary['iou']['std']:.3f} and {100*primary['dice']['std']:.3f} percentage points. "
 'No seed, K or mapping was selected based on these results.\n',
 '**Interpretation:** this is transductive, label-free feature clustering with label-calibrated mapping. '
 'Held-out patient images enter clustering, while held-out patient labels never enter its mapping. '
 'It is not a fully label-free end-to-end segmentation estimate or an untouched external test. '
 'Fold training-label sets overlap, and the cohort has already been explored in earlier audits; LOPO does not undo that history. '
 'The primary and secondary rows were frozen before this round.\n',
 '**UNVERIFIED:** generalization beyond this single acquisition cohort; MRI performance (CT only here); '
 'performance on an untouched test set or on unseen patients whose images were excluded from clustering; '
 'native-resolution boundary quality (scoring uses the same 224×224 masks as BASELINE). No confidence interval '
 'or acquisition-generalization claim is made.\n',
 'Reproduction commands (run from project root; scripts refuse to overwrite completed fit/score outputs):\n',
 '```sh\nPYTHONDONTWRITEBYTECODE=1 python3 audit/scratch/final_baseline/verify.py\n'
 'for k in 8 16; do\n  for seed in 0 1 2; do\n'
 '    PYTHONDONTWRITEBYTECODE=1 python3 audit/scratch/final_baseline/fit.py --k "$k" --seed "$seed"\n  done\ndone\n'
 'for k in 8 16; do\n  for seed in 0 1 2; do\n'
 '    PYTHONDONTWRITEBYTECODE=1 python3 audit/scratch/final_baseline/score.py --k "$k" --seed "$seed"\n  done\ndone\n'
 'PYTHONDONTWRITEBYTECODE=1 python3 audit/scratch/final_baseline/report.py\n```\n']
report.write_text('\n'.join(lines))
print(table(summary),flush=True)
