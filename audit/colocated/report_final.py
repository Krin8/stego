"""Assemble Stage B tables only after the global prediction seal exists."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'experiments/stego_colocated'
read=lambda p:json.loads(p.read_text())
seal=P/'all_predictions_sealed.json';assert seal.exists()
digest=hashlib.sha256(seal.read_bytes()).hexdigest();assert digest==(P/'all_predictions_sealed.sha256').read_text().strip()
decision=read(P/'screen_decision.json');arms=[a for a in ['C448','P1'] if decision[a]['passed']]
results={a:read(P/a/'stage2/metrics/results.json') for a in arms}
for r in results.values():assert r['global_seal_sha256']==digest
lines=['\n### Amended Task 3: final-only Stage 2 and sealed Stage B\n',
 f'Global Stage-A seal SHA-256: `{digest}`. `seal_all.py` verified all advancing arms and seeds before `evaluate.py` opened validation masks. Producers: `stages.py` (label-free fitting, naming, hard/F1 predictions), `seal_all.py` (global gate), `evaluate.py` (validation scoring), and `audit/colocated/report_final.py` (these tables).\n',
 'All Stage-B scores below use labels for evaluation only. Train-only naming N1/N2/N3 and all predictions were fixed before those reads. Pooled-pixel scores sum TP/FP/FN over each patient; mean-slice scores average that patient’s per-slice scores. Across-patient summaries weight the four patients equally and use sample SD (ddof=1). The four validation patients informed earlier choices; this is not an untouched test. No score selected a checkpoint, target or arm.\n',
 '| Arm | Seed | Final batch | Std | Cosine | Cluster entropy | Usage C0–C7 (%) | Bad checks |',
 '| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |']
for arm in arms:
    for seed in [0,1,2]:
        done=read(P/arm/f'stage2/logs/seed{seed}_complete.json');m=done['monitors'][-1]
        usage=', '.join(f'{100*u:.2f}' for u in m['cluster_usage'])
        lines.append(f"| {arm} | {seed} | {done['batches']} | {m['code_std']:.8f} | {m['cross_image_cosine']:.8f} | {m['cluster_entropy']:.6f} | {usage} | {m['consecutive_bad_checks']} |")
lines += ['\n#### Naming (no labels)\n','| Arm | Seed | N1 | N2 | N3 |','| --- | ---: | --- | --- | --- |']
for arm,r in results.items():
    for seed in [0,1,2]:
        names=r['results'][str(seed)]['selected'];lines.append(f"| {arm} | {seed} | {names['N1']} | {names['N2']} | {names['N3']} |")
lines += ['\n#### Stage B: every rule, scale and aggregation\n',
 '| Arm | Seed | Rule | Prediction | Scale | Aggregation | IoU mean ± SD | Dice mean ± SD |',
 '| --- | ---: | --- | --- | ---: | --- | --- | --- |']
patient_lines=['\n#### Stage B: per patient\n',
 '| Arm | Seed | Rule | Prediction | Scale | Aggregation | Patient 5 IoU/Dice | Patient 6 IoU/Dice | Patient 8 IoU/Dice | Patient 30 IoU/Dice |',
 '| --- | ---: | --- | --- | ---: | --- | --- | --- | --- | --- |']
for arm,r in results.items():
    for seed,run in r['results'].items():
        for rule in ['N1','N2','N3']:
            for kind in ['hard','F1']:
                for scale in ['224','512']:
                    result=run['rules'][rule][kind][scale]
                    for agg in ['pooled','mean_slice']:
                        summary=result['summary'][agg]
                        fmt=lambda k:f"{100*summary[k]['mean']:.4f} ± {100*summary[k]['std']:.4f}"
                        prefix=f'| {arm} | {seed} | {rule} | {kind} | {scale} | {agg} | '
                        lines.append(prefix+fmt('iou')+' | '+fmt('dice')+' |')
                        patient_lines.append(prefix+' | '.join(f"{100*result['patients'][p][agg]['iou']:.4f}/{100*result['patients'][p][agg]['dice']:.4f}" for p in ['5','6','8','30'])+' |')
lines+=patient_lines
primary={a:dict(seed_iou=r['primary_seed_iou'],mean=r['mean_seed_iou'],worst=r['worst_seed_iou'],outcome=r['outcome']) for a,r in results.items()}
delta=None
if 'P1' in primary and 'C448' in primary:
    delta=dict(mean=primary['P1']['mean']-primary['C448']['mean'],
               per_seed=[a-b for a,b in zip(primary['P1']['seed_iou'],primary['C448']['seed_iou'])])
lines += ['\n#### Predeclared criterion and ablation\n',
 '| Arm | Seed 0 F1 IoU | Seed 1 F1 IoU | Seed 2 F1 IoU | Seed mean | Worst seed |',
 '| --- | ---: | ---: | ---: | ---: | ---: |']
for arm,r in primary.items():lines.append(f"| {arm} | "+' | '.join(f'{x:.4f}' for x in r['seed_iou']+[r['mean'],r['worst']])+' |')
lines += [f"\nP1 minus C448 (percentage points): {json.dumps(delta) if delta else 'UNVERIFIED: both arms did not advance'}. Frozen baseline supplied in the request: hard **70.83 ± 1.92**, F1 **77.96 ± 2.07**; quoted prior identical-protocol reference, not refitted here.\n",
 'Success: P1 mean over seeds of F1 pooled-pixel IoU >=80.03 AND no seed below77.96. Partial: mean >77.96. Otherwise failure.\n',
 f"**Outcome: {primary['P1']['outcome'].upper() if 'P1' in primary else 'NOT ASSESSED: P1 failed screen'}.** Primary criterion uses N3, 224, patient-pooled then patient mean then seed mean; all other rules/scales/aggregations are reported without selection.\n",
 'UNVERIFIED: generalization beyond the four repeatedly used validation patients, an untouched cohort, longer-run behavior and clinical usefulness. This bounded round does not establish those. No full training, cache regeneration, derivative generation, installs or downloads occurred. All new artifacts are in the requested output locations; existing experiment/data/checkpoint artifacts were read only.\n']
text='\n'.join(lines);(ROOT/'audit/colocated/final_summary.md').write_text(text)
(ROOT/'audit/colocated/final_summary.json').write_text(json.dumps(dict(primary=primary,delta=delta,global_seal_sha256=digest),indent=2)+'\n')
report=ROOT/'audit/FIXES6.md';old=report.read_text();assert '### Amended Task 3: final-only' not in old;report.write_text(old+text)
print(json.dumps(dict(primary=primary,delta=delta),indent=2))
