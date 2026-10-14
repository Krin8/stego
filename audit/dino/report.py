from common import *
inv=json.loads((OUT/'inventory.json').read_text());sel=json.loads((OUT/'pair_selectivity.json').read_text());rows=[];per_seed=[]
for k in [8,2]:
 runs=[json.loads((OUT/f'D1_k{k}_s{s}_scores.json').read_text()) for s in [0,1,2]]
 for res in ['224','512']:
  pats=list(runs[0]['results'][res]['folds'])
  for mapping in (['majority','optimal'] if k==8 else ['Hungarian']):
   patient={pat:{metric:float(np.mean([r['results'][res]['folds'][pat][mapping][metric] for r in runs])) for metric in ['iou','dice']} for pat in pats}
   rows.append(dict(arm='D1',grid='28×28',K=k,mapping=mapping,resolution=res,patients=patient,**{m:patient_stats([v[m] for v in patient.values()]) for m in ['iou','dice']},auc=sel['results']['D1']['auc'],auc_source='pair_score.py'))
   for s,r in enumerate(runs):per_seed.append(dict(arm='D1',K=k,seed=s,mapping=mapping,resolution=res,**{m:patient_stats([r['results'][res]['folds'][p][mapping][m] for p in pats]) for m in ['iou','dice']}))
for ref,grid in [('B0','14×14'),('B2_448','28×28'),('B6','16×16')]:
 runs=[json.loads((OLD/f'{ref}_s{s}_scores.json').read_text()) for s in [0,1,2]]
 for res in ['224','512']:
  for mapping in ['majority','optimal']:
   pats=list(runs[0]['results'][res]['folds']);patient={pat:{m:float(np.mean([r['results'][res]['folds'][pat][mapping][m] for r in runs])) for m in ['iou','dice']} for pat in pats}
   auc=sel['results']['BiomedCLIP_B2_448']['auc'] if ref=='B2_448' else None
   source='pair_score.py same28-grid body crop' if auc else 'Not recomputed; no like-for-like affinity cached for this reference'
   if ref=='B0':auc=json.loads((SELECT/'score_intra.json').read_text())['results']['baseline']['auc'];source='historical SELECTIVITY original14-grid/full-frame geometry, not directly comparable'
   rows.append(dict(arm=ref+' (historical)',grid=grid,K=8,mapping=mapping,resolution=res,patients=patient,**{m:patient_stats([v[m] for v in patient.values()]) for m in ['iou','dice']},auc=auc,auc_source=source))
passed=[r for r in rows if r['arm']=='D1' and r['resolution']=='224' and r['K']==8 and r['iou']['mean']>=.6287 and r['auc']>=.9]
rule=dict(justified=bool(passed),qualifying_mapping_rules=[r['mapping'] for r in passed],no_head_training=True,thresholds=dict(K8_primary_lopo_iou=.6287,same_image_auc=.90),skipped=['D2','D3'])
dump('summary.json',dict(producer='report.py',rows=rows,per_seed=per_seed,selectivity=sel,decision=rule))
def fmt(x):return f"{100*x['mean']:.3f} ± {100*x['std']:.3f} [{100*x['min']:.3f}–{100*x['max']:.3f}]"
lines=['# Local frozen DINO CT liver experiment','',
'All parameters fixed before scoring in `PROTOCOL.json`; producer commands in `commands.txt`. CPU only, no gradient training, downloads, installs or edits outside audit/dino. Each number below is produced by `extract.py`, `fit_score.py`, `pair_affinities.py`, `pair_score.py`, or `report.py`, unless explicitly copied from IMPROVE/SELECTIVITY.',
'', '## Local weights','', '| Requested model | Status | Path / bytes | SHA256 |','|---|---|---|---|']
for model,v in inv.items():
 if model=='other_project_checkpoint':continue
 if not v['files']:lines.append(f'| {model} | MISSING — skipped if declared | — | — |')
 for e in v['files']:lines.append(f"| {model} | FOUND | {e['path']} / {e['bytes']} | {e['sha256']} |")
lines+=['','Search scope: project, ~/.cache/torch, ~/.cache/huggingface, ~/Downloads, ~/Documents completed. A broader hidden-home enumeration was interrupted without additional results; global absence outside these searched locations is UNVERIFIED. DINO S/8 and DINOv2 L/14 are inventoried only, not substituted for missing D2/D3. The generic `models/model_epoch_0720_iter_085000.pth` could not be parsed: `PytorchStreamReader failed reading zip archive: failed finding central directory`; its identity is UNVERIFIED.',
'', '## Preprocessing and fixed protocol','',
'Existing native512 uint16 CT derivatives: HU[-150,250], linear[0,1], quantized through round(65535*x). Native orientation unchanged; no physical resampling. Same label-free body crop as B2: intensity>0.05, largest8-connected component, holes filled, square with5% margin on each side, zero padding. Resize bilinear antialias to224 for D1; grayscale replicated3 channels; ImageNet mean(0.485,0.456,0.406), std(0.229,0.224,0.225) exactly once. D1 final-block LayerNorm tokens, CLS excluded, per-token L2, uncentered,28×28×768. D2 planned448/B16 and D3 planned392/B14 are unexecuted because weights were not found.',
'', 'KMeans uses all2341 slices/20 patients without labels, K8/K2, seeds0/1/2, k-means++, n_init3, max_iter100, tol1e-4, Lloyd, float32. Other19 patient labels choose a mapping; heldout labels score only. K8 strict majority (>50%, ties background) and exhaustive256 subsets for IoU optimum; K2 one-to-one Hungarian maximizing confusion count. Pixels outside body crop are always background and never mapped as a cluster. Native nearest upsample/inverse crop;224 primary nearest-resizes native predictions. Mapping is fitted separately at each scoring resolution, matching IMPROVE. No post-processing.',
'', 'Patient IoU/Dice pool pixels across slices. Average seed metrics within each patient, then report equally weighted20-patient mean, sample SD, min/max. Every fold/subset and per-seed confusion matrix is in `D1_k*_s*_scores.json`; per-seed aggregate rows in `summary.json`.',
'', '## Segmentation (%)','', '| Arm | Grid | K | Mapping | Scoring resolution | IoU mean ± SD [min–max] | Dice mean ± SD [min–max] | Same-image AUC |','|---|---|---:|---|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['arm']} | {r['grid']} | {r['K']} | {r['mapping']} | {r['resolution']} | {fmt(r['iou'])} | {fmt(r['dice'])} | {r['auc'] if r['auc'] is not None else 'UNVERIFIED'} |")
lines+=['','Reference rows copied (no reruns) from `audit/improve/{B0,B2_448,B6}_s{0,1,2}_scores.json`. B6 is DINOv2 ViT-L/14 at224, grid16; it is not declared D3. B0 AUC is the historical SELECTIVITY14-grid full-frame measurement, not the matched28-grid body-crop AUC. B6 AUC was never established and is UNVERIFIED.',
'', '## Pair selectivity','',
'Only472 validation slices/4 patients. Replay SELECTIVITY batch16/last8, seeds20260930,7 passes,11×11 random coordinates,2048 fixed pair positions. Project loss.forward and helper execute restored-global-mean centering; no copied affinity formula. Scores/quantile thresholds fixed before masks are read. Actual PAIRS nearest-label sampler uses common28×28 majority labels from body-cropped nearest448 masks (16×16 majority, ties background). Shared labels and coordinates for D1 and BiomedCLIP B2-448. Label use is diagnostic only. AUC/AP concern liver-anchor targets; attraction budgets concern all pairs. Includes self/same-token pairs as the original declared SELECTIVITY baseline did. This changes geometry from historical full-frame14-grid SELECTIVITY, explicitly for a matched B2 comparison.',
'', '| Source | AUC | AP | Liver-target base rate | Overall attraction | Threshold | Liver-target precision | Liver-pair recall |','|---|---:|---:|---:|---:|---:|---:|---:|']
for name,r in sel['results'].items():
 for s in r['sweep']:lines.append(f"| {name} | {r['auc']:.6f} | {r['average_precision']:.6f} | {100*r['liver_target_base_rate']:.3f}% | {100*s['attraction']:.3f}% | {s['threshold']:.6f} | {100*s['liver_target_precision']:.3f}% | {100*s['liver_pair_recall']:.3f}% |")
lines+=['','## Every D1 seed (%)','','| K | Seed | Mapping | Resolution | IoU mean ± SD [min–max] | Dice mean ± SD [min–max] |','|---:|---:|---|---:|---:|---:|']
for r in per_seed:lines.append(f"| {r['K']} | {r['seed']} | {r['mapping']} | {r['resolution']} | {fmt(r['iou'])} | {fmt(r['dice'])} |")
extract=json.loads((OUT/'D1_extract.json').read_text());fig=json.loads((OUT/'figures.json').read_text())
lines+=['','## Sanity / figures','',f"`extract.py` → `D1_extract.json`: strict load missing_keys={extract['missing_keys']}, unexpected_keys={extract['unexpected_keys']}; eval={extract['model_eval']}; all parameters frozen={extract['all_requires_grad_false']}; two-call max difference={extract['deterministic_max_abs_difference']}; all features finite. No gradients or backbone updates.",
'',f"`figures.py`: `{fig['path']}`; same8 fixed slices from IMPROVE. PCA fitted jointly on their features without labels; deterministic full SVD, shared1st/99th percentile scaling. GT is displayed only; the historical largest-liver slice selection used labels.",
'', '## Declared decision rule','',
f"A DINO-based STEGO head is justified only if at least one declared arm reaches K8 primary224 LOPO liver IoU≥62.87% AND same-image liver-anchor AUC≥0.90. Outcome: {'HOLDS' if passed else 'DOES NOT HOLD'}; qualifying mappings: {rule['qualifying_mapping_rules']}. No head training was run. This rule is a declared screening heuristic, not proof that head training improves segmentation.",
'', '## UNVERIFIED / interpretation limits','',
'D2/D3 missing-weight arms; generic unreadable checkpoint identity; exhaustive local inventory outside searched locations; B6 affinity; independent-cohort or untouched-test performance; liver-absent slice behavior; trained-head benefit. These20 patients have informed earlier pipeline choices. KMeans is label-free, but semantic cluster mapping and evaluation use labels on separate patients. Optimal-subset LOPO is a label-assisted mapping protocol, not wholly label-free liver identification. Four-patient pair-AUC is diagnostic, includes self pairs, and does not establish performance on new cohorts. All poor results and seeds are retained.']
(OUT/'DINO.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rule))
