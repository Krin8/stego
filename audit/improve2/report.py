from common import *
patients=sorted(metadata()['patient_counts'],key=int);variants=['B2_hard','B5_2D','F1','F2']
runs=[json.loads((OUT/f'seed{s}.json').read_text()) for s in [0,1,2]]
gt_counts=np.load(OLD/'scoring_masks_native.npy',mmap_mode='r').sum((1,2))
cohort=dict(slices=len(gt_counts),empty_gt_slices=int((gt_counts==0).sum()),minimum_gt_pixels=int(gt_counts.min()))
rows=[];per={}
for v in variants:
 for res in ['224','512']:
  pp={pat:{m:float(np.mean([r['variants'][v]['results'][res]['patients'][pat][m] for r in runs])) for m in ['iou','dice']} for pat in patients}
  per[f'{v}_{res}']=pp
  rows.append(dict(variant=v,resolution=res,**{m:patient_stats([x[m] for x in pp.values()]) for m in ['iou','dice']}))
diagrows=[];lossrows=[];boundary_rows=[]
for v in variants:
 seeds=[]
 for s,r in enumerate(runs):
  ds=r['variants'][v]['diagnostics'];fps=[n for d in ds.values() for n in d['empty_fp_pixels']]
  mm=[q for d in ds.values() for q in d['multicomponent']]
  counts={k:sum(q[k] for q in mm) for k in ['tp','fp','fn']}
  seeds.append(dict(seed=s,empty_slices=len(fps),empty_fp_pixels_mean=float(np.mean(fps)) if fps else None,
   empty_fp_slices=sum(n>0 for n in fps),empty_fp_pixel_counts=fps,
   multicomponent_slices=len(mm),multicomponent_pooled_iou=counts['tp']/max(1,sum(counts.values())),
   multicomponent_pooled_dice=2*counts['tp']/max(1,2*counts['tp']+counts['fp']+counts['fn'])))
 diagrows.append(dict(variant=v,seeds=seeds))
 for pat in patients:
  bd=[r['variants'][v]['boundary'][pat] for r in runs]
  rr=dict(variant=v,patient=pat)
  for metric in ['hd95','mean_surface_distance','finite_only_hd95','finite_only_mean_surface_distance']:
   values=[b[metric] for b in bd];rr[metric]=float(np.mean(values)) if all(x is not None for x in values) else None
  rr['one_empty_slices_per_seed']=[b['cases']['one_empty'] for b in bd]
  boundary_rows.append(rr)
for stage in ['hard_to_B5','guided_to_lcc']:
 for s,r in enumerate(runs):
  ds=[x for p in r['component_loss'].values() for x in p['details']]
  gt=sum(d['gt_pixels'] for d in ds);removed=sum(d[stage]['removed_true_positive_pixels'] for d in ds)
  lossrows.append(dict(stage=stage,seed=s,multicomponent_slices=len(ds),gt_pixels=gt,
   affected_slices=sum(d[stage]['removed_true_positive_pixels']>0 for d in ds),removed_tp_pixels=removed,
   removed_fraction_of_gt=removed/max(1,gt),
   previously_hit_components=sum(d[stage]['previously_hit_components'] for d in ds),
   previously_hit_components_entirely_lost=sum(d[stage]['previously_hit_components_entirely_lost'] for d in ds)))
truth_details=[x for p in runs[0]['component_loss'].values() for x in p['details']]
component_truth=dict(multicomponent_slices=len(truth_details),
 slices_with_any_component_under10_pixels=sum(min(d['gt_component_sizes'])<10 for d in truth_details))
dump('summary.json',dict(script=__file__,rows=rows,patients=per,diagnostics=diagrows,component_loss=lossrows,
 component_truth=component_truth,boundary=boundary_rows,cohort=cohort,selected_arm=None))
def fmt(d):return f"{100*d['mean']:.3f} ± {100*d['std']:.3f} [{100*d['min']:.3f}, {100*d['max']:.3f}]"
def number(x):return '∞' if x is None else f'{x:.3f}'
lines=['# Fixed F1/F2 refinement\n',
 'Protocol: [PROTOCOL.md](PROTOCOL.md). No arm selected, no parameter tuning. All outputs restricted to audit/improve2. '
 'No source/dataset/cache/checkpoint changes, installs, model loading, frozen forward passes or k-means refitting. '
 'Cached B2-448 hard grids and saved soft weights were reused; inherited mappings unchanged.\n',
 '## Main results\n',
 'All 20 patients, 2,341 slices, seeds 0/1/2. Average seeds within patient, then report equally weighted patient '
 'mean, sample SD and [minimum,maximum]. Both strict-majority and optimal inherited subsets are identical '
 'for every seed/patient and both scoring resolutions; these rows apply to both. No reoptimization after filtering. '
 'F1/F2 are exploratory because the parent was historically selected using cohort labels.\n',
 'Producer: `run.py --seed 0/1/2` → `seed0/1/2.json`; aggregation: `report.py` → `summary.json`.\n',
 '| Arm | Scoring resolution | IoU % mean ± SD [min,max] | Dice % mean ± SD [min,max] | Mapping |',
 '|---|---|---:|---:|---|']
for r in rows:lines.append(f"| {r['variant']} | {r['resolution']} | {fmt(r['iou'])} | {fmt(r['dice'])} | Both inherited |")
lines+=['','B2_hard and B5_2D exactly reproduce every IMPROVE patient IoU/Dice at both resolutions for all seeds '
 '(reference_max_metric_difference=0). B5_2D is the original largest component of hard predictions. '
 'F1/F2 instead apply largest-component removal after guided soft predictions, then holes/closing as declared.\n',
 '## (a) No-liver slices\n',
 f"**There are {cohort['empty_gt_slices']} eligible native slices: all {cohort['slices']} masks contain liver (smallest {cohort['minimum_gt_pixels']} pixels).** "
 'FP pixels per no-liver slice and the corresponding FP rate are **UNVERIFIED / not estimable**, '
 'not evidence of zero false positives. The number of observed qualifying slices with FP is 0 only because '
 'the qualifying set is empty. Labeled liver-absent CT slices would settle this.\n',
 '| Arm | Eligible slices | Mean FP pixels/slice | Slices with any FP |',
 '|---|---:|---|---|']
for v in variants:lines.append(f'| {v} | 0 | N/A | 0 of 0; rate N/A |')
lines+=['','## (b) Multi-component liver slices\n',
 f"Ground truth has >=2 eight-connected components in **{component_truth['multicomponent_slices']} slices**. "
 f"Of these, {component_truth['slices_with_any_component_under10_pixels']} contain a component smaller than 10 pixels. "
 'No component-size threshold was used; component count does not necessarily correspond to anatomical lobes. '
 'Labels are used only to diagnose what each predeclared largest-component stage removed.\n',
 '| Largest-component stage | Seed | Eligible slices | Slices losing captured liver | Captured liver pixels removed | % of eligible GT liver removed | Previously hit GT components entirely lost |',
 '|---|---|---:|---:|---:|---:|---:|']
for d in lossrows:lines.append(f"| {d['stage']} | {d['seed']} | {d['multicomponent_slices']} | {d['affected_slices']} | {d['removed_tp_pixels']} | {100*d['removed_fraction_of_gt']:.3f} | {d['previously_hit_components_entirely_lost']} / {d['previously_hit_components']} |")
lines+=['','The removal counts compare masks immediately before/after the relevant largest-component step. '
 'Thus they isolate that step’s loss of already-captured liver; they do not attribute every missed GT pixel to it. '
 'F1/F2 hole filling, closing and temporal filtering can subsequently alter these pixels.\n',
 '**Yes, largest-component filtering drops true liver on multi-component slices.** '
 'The tables quantify removed captured pixels and entirely lost previously hit components for each seed; '
 'improved overall IoU does not eliminate this failure mode.\n',
 '| Arm | Seed | Multi-component pooled IoU % | Multi-component pooled Dice % |',
 '|---|---|---:|---:|']
for rr in diagrows:
 for d in rr['seeds']:lines.append(f"| {rr['variant']} | {d['seed']} | {100*d['multicomponent_pooled_iou']:.3f} | {100*d['multicomponent_pooled_dice']:.3f} |")
lines+=['','## (c) Native surface distances, per patient\n',
 'Units: **native pixels**, symmetric 2D boundary-pixel distances pooled across each patient’s slices, then '
 'seed-average per patient. HD95 is the 95th percentile of the pooled bidirectional surface distances; '
 'MSD is their mean. Surface pixels use4-neighbour erosion. Distances are exact Euclidean cKDTree nearest '
 'distances, not 3D or millimetres. Labels provide GT surfaces for scoring only.\n',
 'An empty prediction with nonempty GT contributes infinite distances for every unmatched GT surface point. '
 'Strict MSD is therefore infinite if any such point exists. HD95 can remain finite if the infinite tail lies '
 'above its95th percentile. Conditional finite-only distances are also shown; these exclude missing-object '
 'surface points and must be interpreted alongside the missed-slice counts. No finite diagonal penalty was substituted.\n',
 '| Arm | Patient | HD95 px | MSD px | Finite-only HD95 px | Finite-only MSD px | One-empty slices seeds0/1/2 |',
 '|---|---|---:|---:|---:|---:|---|']
for d in boundary_rows:lines.append(f"| {d['variant']} | {d['patient']} | {number(d['hd95'])} | {number(d['mean_surface_distance'])} | {number(d['finite_only_hd95'])} | {number(d['finite_only_mean_surface_distance'])} | {d['one_empty_slices_per_seed']} |")
lines+=['','## Patient IoU/Dice\n']
for v in variants:
 lines+=[f'### {v}\n','| Patient |224 IoU % |224 Dice % |512 IoU % |512 Dice % |','|---|---:|---:|---:|---:|']
 for p in patients:
  x,y=per[f'{v}_224'][p],per[f'{v}_512'][p]
  lines.append(f"| {p} | {100*x['iou']:.3f} | {100*x['dice']:.3f} | {100*y['iou']:.3f} | {100*y['dice']:.3f} |")
 lines.append('')
lines+=['## Labels and limitations\n',
 'Labels enter only inherited other 19-patient mappings, held-out scoring, empty/multi-component/surface analyses '
 'and worst-slice selection; the original largest-liver figure selection also used labels. No labels tune '
 'F1/F2 or select between them. Parent selection in IMPROVE used cohort scores, so both arms remain exploratory. '
 'All cohort images entered the original clustering: transductive, single CT acquisition cohort, no untouched '
 'test set. External performance, liver-absent specificity and 3D/mm boundary metrics remain **UNVERIFIED**. '
 'The 3-slice median uses adjacent acquired slices and endpoint replication; it is not a fixed-millimetre filter.\n',
 'Verification: `validate.py` → `validation.json` checks exact reference reproduction, all patient summary '
 'statistics, the temporal median against an independent Boolean majority implementation, and native-pixel '
 'surface distances against a synthetic 5-pixel displacement. Source inputs are read-only; their digests are '
 'recorded. `repository_final.json` records repository status.\n',
 '## Figures\n']
if (OUT/'figures.json').exists():
 f=json.loads((OUT/'figures.json').read_text());lines += [f'- {x}' for x in f['paths']]
 lines.append('Same 8 earlier slices plus 3 worst B5_2D native per-slice IoUs over all 2341 slices, fixed seed 0. '
  'Worst-slice selection uses labels and is not representative sampling. Details and selected IDs: `figures.json`. '
  'Red=ground truth; green=prediction; each prediction column reports native per-slice IoU.')
 lines.append('PDF rendered with installed pdftoppm and visually inspected (`pdf_render.png`). '
  'F1/F2 smooth stepped boundaries and fill internal holes, but connected spill and missed separate liver '
  'regions remain. All three worst reference slices still have zero IoU in both arms; neither arm is selected.')
else:lines.append('Pending figure export.')
(OUT/'IMPROVE2.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rows),flush=True)
