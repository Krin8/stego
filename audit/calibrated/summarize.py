"""Summarize recorded diagnostics and sealed scores; no images or labels opened."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'audit/calibrated';EXP=ROOT/'experiments/stego_calibrated'
def read(p):return json.loads(p.read_text())
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
out=[]
stage1={};rows=[];timing=[]
for seed in [0,1,2]:
    p=EXP/f'D448/stage1/logs/seed{seed}_complete.json'
    if not p.exists():continue
    c=read(p);stage1[str(seed)]=c
    initial=c['initial'];rows.append(['D448',seed,0,f"{initial['code_std']:.8f}",f"{initial['cross_image_cosine']:.8f}",', '.join(f'{100*x:.2f}' for x in initial['cluster_usage']),initial['healthy']])
    for r in c['monitors']:rows.append(['D448',seed,r['batch'],f"{r['code_std']:.8f}",f"{r['cross_image_cosine']:.8f}",', '.join(f'{100*x:.2f}' for x in r['cluster_usage']),r['healthy']])
    t=read(p.parent/f'seed{seed}_timing.json');timing.append(['D448',seed,'150',f"{t['first10_seconds']:.3f}",f"{t['projected_run_seconds']/60:.3f}",f"{c['seconds']/60:.3f}"])
out.append('## Task 3 — Stage 1 screen\n\nC448 was not run because it failed admissibility. D448 uses the same frozen head/loss implementation and seeded initialization; only the calibrated shifts differ from the sealed baseline. Production runs are sequential, use a package-wide exclusive lock, and have individual process snapshots with no other training job.\n\n'+table(['Candidate','Seed','Batch','Code std','Cosine','Usage C0…C7 (%)','Healthy'],rows))
out.append(table(['Candidate','Seed','Budget','First 10 seconds','Projected minutes','Actual minutes'],timing))
if len(stage1)==3:
    passed=all(x['screen_pass'] for x in stage1.values())
    decision={'D448':dict(passed=passed,seeds={s:x['screen_pass'] for s,x in stage1.items()}),'C448':dict(passed=False,reason='inadmissible; not run')}
    target=EXP/'screen_decision.json'
    if target.exists():assert read(target)==decision
    else:target.write_text(json.dumps(decision,indent=2)+'\n')
    out.append(f'Stage 1 gate: **{"PASS" if passed else "FAIL"}**. Every seed must be healthy at all three checks and have lower cosine at batch 150 than at initialization; the gate is applied without labels.')
stage2={};rows=[];times=[]
for seed in [0,1,2]:
    p=EXP/f'D448/stage2/logs/seed{seed}_complete.json'
    if not p.exists():continue
    c=read(p);stage2[str(seed)]=c
    for r in c['monitors']:rows.append(['D448',seed,r['batch'],f"{r['code_std']:.8f}",f"{r['cross_image_cosine']:.8f}",', '.join(f'{100*x:.2f}' for x in r['cluster_usage']),r['healthy']])
    t=read(p.parent/f'seed{seed}_timing.json');times.append([seed,f"{t['first10_seconds']:.3f}",f"{t['projected_run_seconds']/60:.3f}",f"{c['seconds']/60:.3f}"])
if stage2:
    out.append('## Task 4 — Stage 2\n\nAll runs start from scratch and use the final batch-500 state; no checkpoint selection.\n\n'+table(['Candidate','Seed','Batch','Code std','Cosine','Usage C0…C7 (%)','Healthy'],rows))
    out.append(table(['Seed','First 10 seconds','Projected minutes','Actual minutes'],times))
metric=EXP/'D448/stage2/metrics/results.json'
if metric.exists():
    x=read(metric);rows=[];per=[]
    for seed,r in x['results'].items():
        for rule,rv in r['rules'].items():
            for kind,kv in rv.items():
                for res,values in kv.items():
                    for agg in ['pooled','mean_slice']:
                        s=values['summary'][agg]
                        rows.append(['D448',seed,rule,r['selected'][rule],kind,res,agg,f"{100*s['iou']['mean']:.4f} ± {100*s['iou']['std']:.4f}",f"{100*s['dice']['mean']:.4f} ± {100*s['dice']['std']:.4f}"])
                        for patient,v in values['patients'].items():per.append([seed,rule,kind,res,agg,patient,v['slices'],f"{100*v[agg]['iou']:.4f}",f"{100*v[agg]['dice']:.4f}"])
    out.append('### Stage B — all rules and scales\n\nLabels are used only for validation scoring, after all three selected runs were globally sealed. N1 uses pretrained image/text similarities; N2 uses a fixed anatomical prior; N3 accepts their agreement and scores abstention as an empty mask. Patient-pooled IoU sums TP/FP/FN over all slices of that patient; mean-slice averages individual slice scores (empty union scores zero, inherited unchanged). Primary: N3, F1, 224, four-patient mean of pooled-pixel scores; SD is sample SD across patients.\n\n'+table(['Candidate','Seed','Rule','Named cluster','Prediction','Scale','Aggregation','IoU % mean ± SD','Dice % mean ± SD'],rows))
    out.append('### Per-patient Stage B\n\n'+table(['Seed','Rule','Prediction','Scale','Aggregation','Patient','Slices','IoU %','Dice %'],per))
    out.append(f"### Criterion outcome\n\n**{x['outcome'].upper()}** — N3/F1/224 seed IoUs: {x['primary_seed_iou']}; mean {x['mean_seed_iou']:.4f}%, worst seed {x['worst_seed_iou']:.4f}%.\n\nSuccess: the mean over the three seeds of the F1 pooled-pixel IoU is at least 80.03 AND no seed is below 77.96. Partial: mean above 77.96 but not success. Anything else: failure.\n\nGlobal prediction seal SHA256: `{x['global_seal_sha256']}`. All N1/N2 secondary rows are reported; no candidate, seed, naming rule or checkpoint was selected using scores.")
out.append('### Limitations and UNVERIFIED\n\nThe four validation patients informed earlier choices of K, resolution, window and refinement; this is not an untouched test. Historical seed-1 process/checkpoint lineage remains UNVERIFIED; current serialization equality is verified only on the same CPU/runtime, not across platforms. Label-free cluster usage is a newly fitted fixed-monitor K=8 diagnostic, so IDs do not track semantic identities across checks. No full-length run, external cohort, or improved anatomical specificity is established by passing the stability gate. No installs or downloads; sealed experiment, old caches/checkpoints and datasets were not modified.\n\nProducing scripts: `audit/calibrated/calibrate.py`, `experiments/stego_calibrated/check_state.py`, `train.py`, `stages.py`, `seal_all.py`, `evaluate.py`, and `audit/calibrated/summarize.py`. The new stage source preserves inherited naming/F1 and changes only run-root plumbing, final-checkpoint loading and gating.')
report=ROOT/'audit/FIXES5.md';text=report.read_text();marker='\n<!-- GENERATED RESULTS -->\n'
text=text.split(marker)[0].replace('`resume_seed1.py:29`','`resume_seed1.py:32`').replace('`resume_seed1.py:31`, :60–63','`resume_seed1.py:34`, :60–63')
report.write_text(text+marker+'\n\n'.join(out)+'\n')
(HERE/'summary.json').write_text(json.dumps(dict(stage1={s:x['screen_pass'] for s,x in stage1.items()},stage2_completed=list(stage2),stageB_complete=metric.exists()),indent=2)+'\n')
print('Updated',report)
