"""Report all declared arms without hiding unsuccessful variants."""
from pipeline import *
variants=['B0','B1','B2_336','B2_448','B3','B4_soft','B4_guided','B5_2D','B5_3D','B6']
meta=metadata();patients=sorted(meta['patient_counts'],key=int)
rows=[];per_patient={};seed_rows=[];pending=[]
for variant in variants:
 paths=[OUT/f'{variant}_s{s}_scores.json' for s in [0,1,2]]
 if not all(p.exists() for p in paths):pending.append(variant);continue
 runs=[json.loads(p.read_text()) for p in paths]
 for res in [224,512]:
  for mapping in ['optimal','majority']:
   values={p:{metric:float(np.mean([r['results'][str(res)]['folds'][p][mapping][metric] for r in runs])) for metric in ['iou','dice']} for p in patients}
   key=f'{variant}_{res}_{mapping}';per_patient[key]=values
   rows.append(dict(variant=variant,resolution=res,mapping=mapping,
      **{metric:patient_stats([v[metric] for v in values.values()]) for metric in ['iou','dice']}))
   for s,r in enumerate(runs):seed_rows.append(dict(variant=variant,seed=s,resolution=res,mapping=mapping,**r['results'][str(res)]['summary'][mapping]))
primary=[r for r in rows if r['resolution']==224 and r['mapping']=='optimal']
best=max(primary,key=lambda r:r['iou']['mean'])['variant'] if not pending else None
dump('summary.json',dict(script=__file__,rows=rows,per_patient=per_patient,seed_rows=seed_rows,best=best,pending=pending,
 labels_used='LOPO mapping/scoring; best/parent selection uses cohort scores and is exploratory; feature extraction and clustering are label-free'))
ceiling=json.loads((OUT/'ceiling.json').read_text())
def fmt(s):return f"{100*s['mean']:.3f} ± {100*s['std']:.3f} [{100*s['min']:.3f}, {100*s['max']:.3f}]"
dimensions={'B0':('224','14²'),'B1':('224 crop','14²'),'B2_336':('336 crop','21²'),'B2_448':('448 crop','28²'),'B3':('448 crop / 224 tiles','28² stitched'),'B6':('224 crop','16²')}
parent=json.loads((OUT/'parent_selection.json').read_text())['parent'] if (OUT/'parent_selection.json').exists() else None
if parent:
 for v in ['B4_soft','B4_guided','B5_2D','B5_3D']:dimensions[v]=dimensions[parent]
lines=['# Frozen-feature CT improvements\n',
 'Locked implementation details: [PROTOCOL.md](PROTOCOL.md). All outputs under audit/improve; CPU only. '
 'No head training, installation, project-source changes, derivative/cache regeneration, or existing checkpoint modifications. '
 'Seeds0/1/2, K8, uncentered normalized final patch tokens, all2,341 CT slices and20 patients. '
 'All patches participate in Lloyd KMeans (n_init3, max_iter100, tol1e-4); no patch subsampling.\n',
 '## Evaluation and label use\n',
 'Primary resolution is224×224, preserving FINAL_BASELINE; supplementary native resolution is512×512. '
 'Mapping is fitted on the other19 patients, then scored on the held-out patient, independently per scoring resolution. '
 'Strict majority and exhaustive256-subset optimal mapping both reported. Outside-crop pixels are forced background '
 'and their missed liver pixels remain in the denominator when fitting/scoring mappings. '
 'Patient pixels are pooled before IoU/Dice; seed scores averaged within patient, then equally weighted patient '
 'mean/sample-SD/min/max. No best-seed selection.\n',
 '**Labels:** extraction/cropping/clustering/temperature/filter parameters use no ground-truth annotations. '
 'LOPO mapping and scoring use labels on separate patients. Grid ceilings directly use all labels. '
 'Largest-liver figure slices use validation labels; the four original fixed slices are label-blind patient midpoints. '
 'B4/B5 parent selection and final best-variant selection use cohort LOPO scores and are exploratory. '
 'This is transductive clustering: held-out images are included in fitting, held-out labels excluded from their mappings.\n',
 'For B4/B5, “optimal/majority” refers to the parent hard-cluster mapping fitted on the other19 patients. '
 'Mappings are inherited and not reoptimized after smoothing/component filtering. '
 'B4 soft-only and soft+guided are separately reported; B5 2D and3D are separate filters applied to the parent hard map.\n',
 '## Step0: annotation-derived grid approximations\n',
 'Producer: `ceiling.py` → `ceiling.json`. Hard cells use majority (>0.5), means use bilinear upsampling '
 '(align_corners=False) then threshold>=0.5. All cells use native512 pixels, with explicit nonoverlapping integer-bin '
 'boundaries. Primary results then use the same nearest224 mask transform. These are diagnostic constructions, '
 'not guaranteed mathematical upper bounds on every possible coarse predictor. These uncropped-frame grids '
 'are also not direct upper bounds for the body-cropped/tiled variants.\n',
 'Gap explained = (100 − ceiling mean IoU)/(100 − exact B0 mean IoU), for224 only. '
 'It measures the residual gap attributable to that representation/reconstruction under this diagnostic; '
 'it is not a causal decomposition of the learned model error.\n']
for res in [224,512]:
 lines += [f'### Ceiling at scoring resolution {res}\n',
 '| Grid | Reconstruction | IoU mean ± SD [min,max] % | Dice mean ± SD [min,max] % | Gap to100 (pp) | B0 gap explained % |',
 '|---|---|---:|---:|---:|---:|']
 for r in ceiling['rows']:
  if r['resolution']!=res:continue
  fraction=r['fraction_of_224_baseline_gap'];gap=f'{100*fraction:.2f}' if fraction is not None else '—'
  lines.append(f"| {r['grid']}² | {r['method']} | {fmt(r['summary']['iou'])} | {fmt(r['summary']['dice'])} | {r['gap_to_100_pp']:.3f} | {gap} |")
 lines.append('')
for res in [224,512]:
 lines += [f'## Variant scores — {res}'+(' (primary)' if res==224 else ' (native supplementary)')+'\n',
 'Producer: `score.py` and `postprocess.py` → per-variant/per-seed `_scores.json`; '
 '`report.py` → `summary.json` aggregates seeds within patient.\n',
 '| Variant | Input | Grid | Mapping | IoU mean ± SD [min,max] % | Dice mean ± SD [min,max] % |',
 '|---|---|---|---|---:|---:|']
 for r in rows:
  if r['resolution']!=res:continue
  v=r['variant'];dim=dimensions[v];name=v+(' exploratory' if v.startswith(('B4','B5')) else '')
  lines.append(f"| {name} | {dim[0]} | {dim[1]} | {r['mapping']} | {fmt(r['iou'])} | {fmt(r['dice'])} |")
 lines.append('')
if pending:lines.append('PENDING / UNVERIFIED: '+', '.join(pending)+'\n')
if primary:
 baseline=next(r for r in primary if r['variant']=='B0');threshold=baseline['iou']['mean']+baseline['iou']['std']
 beating=[r['variant'] for r in primary if r['iou']['mean']>threshold]
 lines+=['## Descriptive comparison\n',f"B0+its across-patient SD threshold is {100*threshold:.3f}% IoU. "
  f"Variants exceeding it (primary optimal mapping): {', '.join(beating) if beating else 'none'}. "
  'This is the requested descriptive criterion, not a paired statistical significance test.\n']
 if parent:lines.append(f'B4/B5 parent: **{parent}**, pre-authorized selection by mean224 optimal LOPO IoU across seeds. '
  'Selection evidence: `select_parent.py` → `parent_selection.json`. No parameter tuning followed selection.\n')
 if best:lines.append(f'Descriptive best variant: **{best}**. No independent test set was used to confirm this choice.\n')
lines+=['## Per-patient tables (seeds averaged within patient)\n']
for variant in variants:
 if variant in pending:continue
 for res in [224,512]:
  lines+=[f'### {variant}, resolution{res}\n','| Patient | Slices | Optimal IoU | Optimal Dice | Majority IoU | Majority Dice |','|---|---:|---:|---:|---:|---:|']
  for patient in patients:
   o=per_patient[f'{variant}_{res}_optimal'][patient];m=per_patient[f'{variant}_{res}_majority'][patient]
   lines.append(f"| {patient} | {meta['patient_counts'][patient]} | {100*o['iou']:.3f} | {100*o['dice']:.3f} | {100*m['iou']:.3f} | {100*m['dice']:.3f} |")
  lines.append('')
lines+=['## Per-seed summaries\n','| Variant | Seed | Resolution | Mapping | IoU mean ± SD [min,max] % | Dice mean ± SD [min,max] % |','|---|---|---|---|---:|---:|']
for r in seed_rows:lines.append(f"| {r['variant']} | {r['seed']} | {r['resolution']} | {r['mapping']} | {fmt(r['iou'])} | {fmt(r['dice'])} |")
lines+=['','## Implementation evidence and limits\n',
 'B0 refitting/scoring reproduced every original per-patient IoU/Dice exactly for all3 seeds '
 '(`B0_s*_scores.json`: baseline_max_patient_metric_difference=0). `prepare.py` → `support.json` '
 'confirms existing dynamic positional interpolation at336/448 with21²/28² feature grids. '
 'Actual installed positional embedding implementation saved as `position_embedding_source.txt`. '
 'Body crop has zero empty-body fallbacks (`metadata.json`); crop boundaries and physical positions saved per sample. '
 'Native inverse-crop maps include zero-padding and force out-of-crop background.\n',
 'B0 primary hard assignments go directly14→224 to exactly reproduce FINAL_BASELINE. '
 'Crop variants are first restored to512, then sampled with the same nearest224 mask transform; '
 'native results expose the frame-level comparison. Per-seed raw numbers and cluster mappings are preserved in JSON.\n',
 'Extraction: `extract.py`, frozen project BiomedCLIP extractor in eval/no-grad; B3 averages normalized-block '
 'tokens over9 overlapping tiles before L2. `fit.py` uses a disposable disk-backed working matrix with copy_x=False '
 'to bound RAM; statistical KMeans settings unchanged. This does not alter existing features or pipeline caches. '
 'B4 temperature is derived from unlabeled distance gaps (recorded in each fit JSON), not adjusted to labels. '
 'Guided filter is implemented with installed scipy uniform filters, radius8/epsilon0.01. '
 '3D component filtering uses6-connectivity in physically sorted consecutive slices; anisotropic physical spacing '
 'is not resampled, and anatomy fragmented across slices may be removed.\n',
 'Optional B6 uses the locally available DINOv2 ViT-L/14 checkpoint via offline local torch hub, strict weight loading, '
 'final LayerNorm patch tokens, CLS excluded,16² at224 and DINO ImageNet mean/std; no download. '
 'Its checkpoint/model support is verified by successful extraction, otherwise explicitly UNVERIFIED.\n',
 'B6 is a pre-declared224-input, body-cropped comparison; it is not a resolution-matched comparison '
 'against the448 BiomedCLIP arm and cannot establish universal backbone superiority.\n',
 '**UNVERIFIED:** external/unseen-cohort generalization; performance on an untouched test set; '
 'clinical/native-resolution boundary reliability; other crop thresholds, soft temperatures or filters (not tuned). '
 'The cohort is CT-only and already explored. B4/B5 and selected-best scores include selection optimism; '
 'LOPO mapping does not remove that outer-selection bias.\n',
 'Reproduction commands: `commands.txt`. Repository state: `repository_state.json`. '
 'Live current-vs-reused feature and synthetic crop checks: `checks.py` → `checks.json`. '
 'Final fold/statistic/pixel-count/source-hash validation: `validate.py` → `validation.json`.\n',
 '## Figures\n']
if (OUT/'figures.json').exists():
 fig=json.loads((OUT/'figures.json').read_text())
 for path in fig['paths']:lines.append(f'- {path}')
 lines.append('Figure producer `figures.py`; seed0 was fixed in advance. Per-slice IoU uses native512 labels and '
  'native-resolution other19-patient mappings. Green=prediction, red=ground truth; no visual-based tuning. '
  'Selected sample IDs and exact per-slice IoU/Dice are in `figures.json`.')
 lines.append('Visual inspection: B5_2D removes disconnected false positives and more closely follows the liver '
  'on these slices, but retains stepped boundaries, small holes and undersegmentation. The largest-liver '
  'slice of patient8 still shows a large connected spill outside the ground-truth contour; patient5’s fixed '
  'slice still misses substantial liver area. Better mean IoU does not imply clean native-resolution contours.')
else:lines.append('Pending figures.')
(OUT/'IMPROVE.md').write_text('\n'.join(lines)+'\n')
print('Report updated; pending',pending,'best',best,flush=True)
