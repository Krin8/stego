"""Aggregate seeds within patient, then report equal-patient sample SD (ddof=1)."""
from study import *
allruns=[json.loads((HERE/f'inductive_seed{s}_results.json').read_text()) for s in [0,1,2]]
pats=list(allruns[0]['results']['F1']['patients'])
def stats(v):
    a=np.asarray(v,float)
    return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)) if len(a)>1 else 0.,min=float(a.min()),max=float(a.max()))
def fmt(s):return f"{100*s['mean']:.3f} ± {100*s['sd']:.3f} [{100*s['min']:.3f}–{100*s['max']:.3f}]"
def pct(v):return f'{100*v:.3f}'
summary=[];per=[];boundaries=[];tiny=[]
for arm in ['hard','F1']:
    for res in ['224','512']:
        for aggregation in ['pooled','slice_mean']:
            pv={pat:{m:float(np.mean([r['results'][arm]['patients'][pat][aggregation][res][m] for r in allruns])) for m in ['iou','dice']} for pat in pats}
            for pat,v in pv.items():per.append(dict(arm=arm,resolution=res,aggregation=aggregation,patient=pat,slices=allruns[0]['results'][arm]['patients'][pat]['count'],**v))
            summary.append(dict(arm=arm,resolution=res,aggregation=aggregation,**{m:stats([v[m] for v in pv.values()]) for m in ['iou','dice']}))
    for pat in pats:
        for seed,r in enumerate(allruns):boundaries.append(dict(arm=arm,patient=pat,seed=seed,**r['results'][arm]['patients'][pat]['boundary']))
    for category in ['<2000','>=2000','empty']:
        for res in ['224','512']:
            rows=[]
            for seed,r in enumerate(allruns):
                ss=[s for s in r['results'][arm]['slices'] if (s['gt_pixels']<2000 if category=='<2000' else s['gt_pixels']>=2000 if category=='>=2000' else s['gt_pixels']==0)]
                c={k:sum(s['scores'][res][k] for s in ss) for k in ['tp','fp','fn']}
                rows.append(dict(seed=seed,count=len(ss),pooled_iou=c['tp']/max(1,sum(c.values())) if ss else None,
                    pooled_dice=2*c['tp']/max(1,2*c['tp']+c['fp']+c['fn']) if ss else None,
                    mean_slice_iou=float(np.mean([s['scores'][res]['iou'] for s in ss])) if ss else None,
                    mean_slice_dice=float(np.mean([s['scores'][res]['dice'] for s in ss])) if ss else None,
                    total_fp_pixels=sum(s['fp_pixels'] for s in ss),slices_with_fp=sum(s['fp_pixels']>0 for s in ss)))
            tiny.append(dict(arm=arm,category=category,resolution=res,rows=rows))
old=[json.loads((ROOT/f'audit/improve2/seed{s}.json').read_text()) for s in [0,1,2]]
comparisons=[]
for arm,oldarm in [('hard','B2_hard'),('F1','F1')]:
    for res in ['224','512']:
        for cohort in ['all20','same4']:
            ids=list(old[0]['variants'][oldarm]['results'][res]['patients']) if cohort=='all20' else pats
            pv={pat:{metric:float(np.mean([s['variants'][oldarm]['results'][res]['patients'][pat][metric] for s in old])) for metric in ['iou','dice']} for pat in ids}
            row=dict(arm=arm,resolution=res,cohort=cohort,**{m:stats([v[m] for v in pv.values()]) for m in ['iou','dice']})
            inductive=next(x for x in summary if x['arm']==arm and x['resolution']==res and x['aggregation']=='pooled')
            row['inductive_minus_lopo_pp']={m:100*(inductive[m]['mean']-row[m]['mean']) for m in ['iou','dice']};comparisons.append(row)
dump('summary.json',dict(producer='report.py',seed_aggregation='average metric over seeds per patient; then equal-patient mean, sample SD, min, max',patients=per,summary=summary,boundaries=boundaries,strata=tiny,lopo_comparisons=comparisons))
lines=['# Inductive frozen-feature F1 evaluation','',
'Producer: `study.py fit/mapping/evaluate --seed 0/1/2`; aggregation: `report.py`. Full per-seed, per-slice and surface-distance evidence: `inductive_seed{0,1,2}_results.json`, `summary.json`.',
'', 'Centres and temperature fitted on all 1,869 slices from the 16 training patients; no validation images or labels in fitting. Strict-majority mapping uses native512 training labels only (ties → background), before any validation scoring. A single mapping is used for both resolutions and hard/F1. Validation: 472 slices, patients 5/6/8/30. Labels enter mapping and scoring only; post-processing uses image/predictions.',
'', 'Pooled IoU = sum(TP)/(sum(TP)+sum(FP)+sum(FN)) within each patient; pooled Dice = 2sum(TP)/(2sum(TP)+sum(FP)+sum(FN)). Slice mean = arithmetic mean of individual slice IoUs/Dices. Average seeds within each patient, then equally weight the four patients; SD is sample SD across these four seed-averaged patient scores. Seed-specific results remain in JSON. These are different estimands, not confidence intervals.',
'', '## Per-patient results (%)','', '| Arm | Resolution | Patient (slices) | Pooled IoU / Dice | Mean slice IoU / Dice |','|---|---:|---|---:|---:|']
for arm in ['hard','F1']:
 for res in ['224','512']:
  for pat in pats:
   a=next(x for x in per if x['arm']==arm and x['resolution']==res and x['patient']==pat and x['aggregation']=='pooled');b=next(x for x in per if x['arm']==arm and x['resolution']==res and x['patient']==pat and x['aggregation']=='slice_mean')
   lines.append(f"| {arm} | {res} | {pat} ({a['slices']}) | {pct(a['iou'])} / {pct(a['dice'])} | {pct(b['iou'])} / {pct(b['dice'])} |")
lines+=['','## Across-patient summary (%)','','| Arm | Resolution | Aggregation | IoU mean ± SD [min–max] | Dice mean ± SD [min–max] |','|---|---:|---|---:|---:|']
for r in summary:lines.append(f"| {r['arm']} | {r['resolution']} | {r['aggregation']} | {fmt(r['iou'])} | {fmt(r['dice'])} |")
lines+=['','## Small-liver slices (size defined at native512)','','Slice strata pool the same 472 validation slices across patients, then average the three seed scores. Both mean-slice and pooled-pixel results shown. Labels define strata and score, never select parameters.','','| Arm | Resolution | Native liver pixels | Slices | Pooled IoU / Dice | Mean slice IoU / Dice |','|---|---:|---|---:|---:|---:|']
for r in tiny:
 if r['category']=='empty':continue
 v=r['rows'];vals=[np.mean([x[k] for x in v]) for k in ['pooled_iou','pooled_dice','mean_slice_iou','mean_slice_dice']]
 lines.append(f"| {r['arm']} | {r['resolution']} | {r['category']} | {v[0]['count']} | {pct(vals[0])} / {pct(vals[1])} | {pct(vals[2])} / {pct(vals[3])} |")
empty=next(x for x in tiny if x['arm']=='F1' and x['category']=='empty' and x['resolution']=='512')['rows']
lines+=['',f"No-liver slices: {empty[0]['count']}. FP count/rate on liver-absent slices is UNVERIFIED / not estimable when this count is zero; zero eligible slices does not establish zero false positives.",
'','## Native surface distances (pixels)','','Symmetric 2D 4-connected boundary distances are pooled across slices per patient; HD95 is the 95th percentile, MSD the mean. One-empty prediction/GT contributes infinity for its unmatched surface points. Infinity is retained, not silently dropped. Conditional finite-only metrics are separately labeled. These are pixels, not mm or 3D distances.','','| Arm | Patient | Seed | HD95 | MSD | Finite-only HD95 / MSD | One-empty slices |','|---|---:|---:|---:|---:|---:|---:|']
def num(v):return '∞' if v is None else f'{v:.3f}'
for r in boundaries:lines.append(f"| {r['arm']} | {r['patient']} | {r['seed']} | {num(r['hd95'])} | {num(r['mean_surface_distance'])} | {num(r['finite_only_hd95'])} / {num(r['finite_only_mean_surface_distance'])} | {r['cases']['one_empty']} |")
lines+=['','## Comparison with IMPROVE2 LOPO (%)','','Historical LOPO fitted centres on all20 patients, mapped on other19 labels. Current fit and mapping use train16 only. Same4 comparison controls the scored patients; all20 comparison also changes the scoring cohort. Neither is an untouched test: this cohort informed historical pipeline selection.','','| Arm | Resolution | LOPO cohort | LOPO IoU mean ± SD [min–max] | LOPO Dice mean ± SD [min–max] | Inductive minus LOPO IoU / Dice (pp) |','|---|---:|---|---:|---:|---:|']
for r in comparisons:lines.append(f"| {r['arm']} | {r['resolution']} | {r['cohort']} | {fmt(r['iou'])} | {fmt(r['dice'])} | {r['inductive_minus_lopo_pp']['iou']:+.3f} / {r['inductive_minus_lopo_pp']['dice']:+.3f} |")
lines+=['','## Deployment and limitations','','`all20_seed*.npy/.json` use existing all20 label-free centres and all20 strict-majority mapping. They are deployable artifacts, NOT the source of the inductive scores. Seed0 is the default by declaration, not selected by validation performance.',
'', 'UNVERIFIED: independent acquisition cohort, liver-absent slices, clinical utility, outside-CHAOS performance, arbitrary PNG intensity history, 3D/mm boundary accuracy. CT only, 20 patients, one acquisition cohort, only four validation patients. Tiny liver slices can be missed completely. The parent was historically selected using cohort labels, so this is an inductive recomputation on an already-used split, not an unbiased untouched-test estimate.']
(HERE/'FINAL_PIPELINE.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[lines.index('## Across-patient summary (%)'):lines.index('## Small-liver slices (size defined at native512)')]))
