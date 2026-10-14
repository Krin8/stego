"""Predeclared six-run evaluation report; no validation-based candidate replacement."""
from pathlib import Path
import json,sys,statistics
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
def criterion(values):
 if len(values)!=3 or any(x is None for x in values):return 'FAILURE — incomplete batch500 evidence'
 mean=statistics.mean(values);worst=min(values)
 if mean>.6414:return 'SUCCESS'
 if mean>.50 and worst>.40:return 'PARTIAL RESULT'
 return 'FAILURE'
assert criterion([.65,.65,.65])=='SUCCESS'
assert criterion([.6414]*3)=='PARTIAL RESULT'
assert criterion([.51,.51,.51])=='PARTIAL RESULT'
assert criterion([.7,.5,.4])=='FAILURE'
assert criterion([.5]*3)=='FAILURE'
assert criterion([None,.7,.7]).startswith('FAILURE')
if '--self-test' in sys.argv:
 print('Strict criterion boundary tests PASS; no training or evaluation.');sys.exit(0)
def pct(x):return f'{100*x:.3f}'
def pair(r):return f'{pct(r["iou"])} / {pct(r["dice"])}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
cal=json.loads((OUT/'calibration.json').read_text());rows=[];patients=[];summaries=[];monitor_rows=[];overlays=[];allresults=[]
for candidate in ['D','E']:
 final=[]
 for seed in [0,1,2]:
  name=f'fixes4_{candidate}_seed{seed}';r=json.loads((OUT/f'{name}_evaluation.json').read_text());allresults.extend(r['results'])
  terminal=json.loads((OUT/'experiments'/name/'result.json').read_text())
  for x in r['results']:
   if x['status']!='EVALUATED':rows.append([candidate,seed,x['batch'],'MISSING','MISSING','MISSING','Early-stopped' if x['early_stopped'] else 'UNVERIFIED']);continue
   active=sum(u>=.01 for u in x['monitors']['cluster_usage'])
   status='Collapse threshold exceeded' if x['unhealthy'] else 'Degenerate probe' if active<2 else 'Non-collapsed'
   lopo={m:x['lopo_summary'][m]['mean'] for m in ['iou','dice']}
   rows.append([candidate,seed,x['batch'],pair(x['majority']),pair(x['optimal']),pair(lopo),status])
   for patient,p in x['lopo'].items():patients.append([candidate,seed,x['batch'],patient,','.join(map(str,p['heldout']['clusters'])),pair(p['heldout'])])
   if x['batch']==500:final.append(x['lopo_summary']['iou']['mean'])
  mon=terminal['monitors'];monitor_rows.append([candidate,seed,terminal['batches'],f'{mon["code_std"]:.8f}',f'{mon["cross_image_cosine"]:.8f}',f'{mon["cluster_entropy"]:.6f}',str([round(u,6) for u in mon['cluster_usage']]),mon['consecutive_bad_checks'],terminal['early_stopped']])
  overlays.append([candidate,seed,terminal['overlay']])
 result=criterion(final);summaries.append(dict(candidate=candidate,mean=statistics.mean(final) if len(final)==3 else None,worst=min(final) if len(final)==3 else None,criterion=result,seeds_completed=len(final)))
evaluation_table=table(['Candidate','Seed','Batch','Majority IoU / Dice %','Optimal IoU / Dice %','LOPO mean IoU / Dice %','Collapse status'],rows)
parts=['\n## Task 3 — all checkpoint evaluations',
'Producer: `audit/scratch/fixes4/evaluate.py --candidate D|E --seed 0|1|2`, one invocation per run; outputs `fixes4_<candidate>_seed<seed>_evaluation.json`. Imports actual extractor, preprocessing, head and cluster probe. Original BASELINE/FORENSICS scoring functions are executed unchanged. Global majority/optimal mappings use all472 validation labels for mapping AND scoring; LOPO fits the IoU-optimal subset on three patients and scores the held-out fourth. No training label is read or fitted. Scores are percentages. LOPO is an equal-patient average, not pixel pooling.',evaluation_table,
'Validation patients: 5 (95 slices), 6 (82), 8 (98), 30 (197), matching immutable BASELINE sample metadata. All18 checkpoint rows are reported; missing ones are not replaced. Cluster-label mappings, confusion matrices, per-patient scores and checkpoint existence/size are in the JSON evidence.',
'### Per-patient LOPO scores and mappings',table(['Candidate','Seed','Batch','Held-out patient','Liver cluster subset fit on other3','Held-out IoU / Dice %'],patients),
'### Terminal monitors (no labels)',table(['Candidate','Seed','Batch','Code std','Cross-image cosine','Entropy nats','Usage IDs0–7','Consecutive bad','Early stop'],monitor_rows),
'### Fixed-slice overlays (validation labels for GT/mapped display only)',table(['Candidate','Seed','Original terminal overlay'],overlays),
'## Task 4 — criterion applied verbatim',
'“the MEAN across the three seeds of leave-one-patient-out mean liver IoU at batch 500 exceeds 64.14% for a candidate. Partial result: mean above 50% with the worst seed above 40%. Anything else is a failure.”',
table(['Candidate','Mean seed LOPO IoU %','Worst seed LOPO IoU %','Result'],[[x['candidate'],pct(x['mean']) if x['mean'] is not None else 'UNVERIFIED',pct(x['worst']) if x['worst'] is not None else 'UNVERIFIED',x['criterion']] for x in summaries]),
'No winner was selected using validation labels. The fixed candidates, seeds, budgets, checkpoints and criterion were declared before training. Global optimal scores are optimistic diagnostics; LOPO reduces mapping optimism but still uses validation labels for calibration/scoring. The user-declared64.14% threshold is retained literally. For context, the existing identical-evaluation frozen K8 uncentered baseline has64.020% LOPO patient-mean IoU (BASELINE.md); this does not change the criterion.',
'### What remains UNVERIFIED',
'Long-run quality/stability and unseen-patient generalization remain UNVERIFIED. No full5000-batch run was executed. Only four validation patients are available; repeated evaluation does not create an untouched test cohort. Train-only calibration uses200 slices and55 cached kNN edges retained within that sample, exactly reproducing C; actual stochastic-training attraction rates with dropout are not guaranteed to equal those eval-mode calibration targets. Clinical/pixel-boundary performance is not established by these coarse-grid scores. Untrained linear-probe logger metrics are not a supervised reference.']
commands=[]
for summary in summaries:
 if summary['criterion']=='SUCCESS':
  candidate=summary['candidate'];sh=cal['candidates'][candidate]
  command='cd /Users/navneetbavineni/STEGO/src\nWANDB_MODE=offline HF_HUB_OFFLINE=1 caffeinate -i python3 -B train_segmentation.py "pytorch_data_dir=$PWD/pytorch_data_dir" '+f'experiment_name=fixes4_full_{candidate}_seed0 seed=0 max_steps=5000 extra_clusters=6 collapse_early_stop=True batch_size=16 num_workers=0 val_freq=50 '+ ' '.join(f'{k}={sh[k]}' for k in ['pos_intra_shift','pos_inter_shift','neg_inter_shift'])
  commands.append(command)
  parts+=['### Eligible full-run command — NOT RUN; default seed0, not validation-selected','```sh\n'+command+'\n```']
if not commands:parts.append('No candidate meets the success criterion; no full-run command is printed or executed.')
marker='\n## Task 3 — all checkpoint evaluations';path=ROOT/'audit/FIXES4.md';text=path.read_text()
split=parts.index('## Task 4 — criterion applied verbatim')
if '--phase3' in sys.argv:
 assert marker not in text
 path.write_text(text+'\n\n'.join(parts[:split])+'\n')
 print(evaluation_table);sys.exit(0)
assert '## Task 4 — criterion applied verbatim' not in text
path.write_text(text+'\n\n'.join(parts[split:] if marker in text else parts)+'\n')
(OUT/'round_summary.json').write_text(json.dumps(dict(script=__file__,summaries=summaries,overlays=overlays,full_commands=commands,results=allresults),indent=2)+'\n')
print(evaluation_table);print(json.dumps(summaries,indent=2))
