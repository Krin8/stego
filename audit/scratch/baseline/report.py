"""Assemble measured outputs only; no model fitting, image/mask/checkpoint writes."""
from common import *
from scipy.stats import spearmanr
import subprocess
rows=[]
for center in [0,1]:
 for k in [8,16]:
  for seed in [0,1,2]:
   name=f'k{k}_c{center}_s{seed}';r=json.loads((OUT/f'{name}.json').read_text());r['id']=name;rows.append(r)
heads=json.loads((OUT/'heads.json').read_text())
for key,r in heads.items():r['id']=key;rows.append(r)
extract=json.loads((OUT/'extract.json').read_text());meta=json.loads((OUT/'samples.json').read_text())
fmt=lambda x:f'{100*x:.3f}'
pair=lambda r:fmt(r['iou'])+' / '+fmt(r['dice'])
lines=['# CHAOS CT identical-sample baseline and held-out-patient mapping','',
'## Verdict','',
'VERDICT_PLACEHOLDER','',
'## Protocol and label use','',
f'All {extract["train_slices"]:,} train slices ({extract["train_patches"]:,} patches; {len(extract["train_patients"])} patients), and all {extract["val_slices"]} validation slices (four disjoint patients). Producer: `audit/scratch/baseline/extract.py`, `extract.json`, `samples.json`. Fresh features matched the FORENSICS validation features with maximum absolute error {extract["val_feature_max_abs_difference"]}; every validation mask and ordered sample ID matched exactly. The actual project extractor, preprocessing, head and probe are imported through `common.py`. Extraction uses `ImagesOnlyDataset`; a runtime guard forbids opening training annotations.',
'',
'Frozen features: final-block LayerNorm patch tokens, CLS excluded, 14×14 grid, 768 channels. No-mean variant applies per-patch L2 directly; centered variant subtracts the mean of ALL training patch tokens and then applies L2. K-means is fitted only on training images, with K=8/16 and seeds 0/1/2 declared in advance. Lloyd, three initializations per seed, maximum 100 iterations, tolerance 1e-4; no validation labels influence model fitting, mean, K, seed or parameters. K-means assignments use nearest-neighbor upsampling to 224×224, matching DIAG/FORENSICS. Trained heads use the actual bilinear-upsample-code-then-probe path. Thus sample, scoring and mapping rules are identical; the native inference/upsampling operations remain method-specific, not artificially equated.',
'',
'**Label accounting for every table:** global majority and global optimal use all 472 validation masks for BOTH mapping and scoring (optimistic diagnostics). LOPO selects the optimal subset with three validation patients’ labels, then scores on the fourth patient’s labels; the held-out labels never choose its mapping. LOPO is supervised calibration of a cluster mapping, not label-free evaluation. Patient mean/min weight each patient equally; pooled LOPO aggregates TP/FP/FN across all held-out pixels for direct comparison to global scores. Slice counts, feature dimensions, fit inertia/iterations, runtimes and label-free proxies use no labels. Confusion counts, IoU, Dice and correlations with oracle IoU use evaluation labels only.',
'',
'`common.py` AST-imports FORENSICS `scores` and `mapping_report`, and its actual confusion-matrix expression. Only K-dependent constants are generalized. Both K=8 and K=16 use exhaustive enumeration of all 2^K subsets, including the empty subset. K=8 outputs are asserted identical to the unmodified FORENSICS function. Strict-majority means liver pixels > background pixels, with ties assigned to background. Trained-head prediction artifacts are reused from `forensics/task1.py`; `heads.py` verifies that all five global confusion matrices and mappings exactly reproduce FORENSICS. No CRF is applied.',
'',
'## Main comparison: all validation pixels','',
'Every score below is percent, **IoU / Dice**. Producers: frozen rows `audit/scratch/baseline/kmeans.py --k K --center C --seed S` → `kK_cC_sS.json`; head rows `audit/scratch/baseline/heads.py` → `heads.json`, using `forensics/task1.py` predictions. Global mapping/scoring labels: all four validation patients.',
'',
'| Method | K | Seed | Majority IoU / Dice | Optimal IoU / Dice | LOPO mean IoU / Dice |',
'|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f'| {r["method"]} | {r["K"]} | {r["seed"]} | {pair(r["majority"])} | {pair(r["optimal"])} | {fmt(r["lopo_summary"]["iou"]["mean"])} / {fmt(r["lopo_summary"]["dice"]["mean"])} |')
lines += ['| DIAG supervised linear probe | — | 20260930 | UNVERIFIED | UNVERIFIED | UNVERIFIED |','',
'The original DIAG linear-probe coefficients were not saved: `audit/scratch/diag_ceiling.py` records `model_saved=False` and contains no model serialization. A targeted file-name search under `audit/scratch/diag` and `audit/run_records` found no probe/coefficient, pickle, joblib or NPZ artifact. Its 100-slice predictions cannot establish a 472-slice score or LOPO results. A recoverable original fitted estimator (and its train mean/normalization metadata) would settle this; supervised refitting is outside this request. No old 100-slice score is mixed into this table.',
'', '## Leave-one-patient-out mapping','',
'Producer: `extract.py` for IDs/counts (no labels); `kmeans.py` and `heads.py` for every per-patient and aggregate score. LOPO mapping labels: the other three patients only; scoring labels: the indicated held-out patient.', '',
'| Patient | Validation slices |','|---|---:|']
for p,n in extract['val_patients'].items():lines.append(f'| {p} | {n} |')
lines += ['', '| Method / K / seed | Patient 5 IoU / Dice | Patient 6 IoU / Dice | Patient 8 IoU / Dice | Patient 30 IoU / Dice | Mean IoU / Dice | Minimum IoU / Dice | Pooled IoU / Dice |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 vals=[pair(r['lopo'][p]['heldout']) for p in ['5','6','8','30']]
 lines.append('| '+f'{r["method"]} / {r["K"]} / {r["seed"]}'+' | '+' | '.join(vals)+f' | {fmt(r["lopo_summary"]["iou"]["mean"])} / {fmt(r["lopo_summary"]["dice"]["mean"])} | {fmt(r["lopo_summary"]["iou"]["min"])} / {fmt(r["lopo_summary"]["dice"]["min"])} | {pair(r["lopo_pooled"])} |')
lines += ['', 'Mapping subsets, including every global and held-out mapping, are recorded below. IDs belong only to that fitted model and are not shared across seeds. A LOPO score can be lower than its global optimum; macro vs pooled weighting can also change their ordering.', '',
'| ID | Global majority liver clusters | Global optimal liver clusters | Held-out 5 mapping | Held-out 6 mapping | Held-out 8 mapping | Held-out 30 mapping |','|---|---|---|---|---|---|---|']
for r in rows:lines.append('| '+r['id']+' | '+str(r['majority']['clusters'])+' | '+str(r['optimal']['clusters'])+' | '+' | '.join(str(r['lopo'][p]['heldout']['clusters']) for p in ['5','6','8','30'])+' |')
lines += ['', '## Label-free checkpoint proxies','']
proxyfiles=sorted(OUT.glob('proxies_*_*.json'));proxies=[]
for p in proxyfiles:proxies.extend(json.loads(p.read_text())['results'])
proxies.sort(key=lambda r:r['batch']);assert len({r['batch'] for r in proxies})==len(proxies)
targets={r['batch']:r for p in FORE.glob('task5_*_*.json') for r in json.loads(p.read_text())}
correlations={}
if len(proxies)==50:
 assert [r['batch'] for r in proxies]==list(range(100,5001,100))
 get={'correspondence loss':lambda r:r['losses']['weighted_correspondence'],'cross-image cosine':lambda r:r['monitors']['cross_image_cosine'],'cluster entropy':lambda r:r['monitors']['cluster_entropy'],'ARI vs frozen K8':lambda r:r['agreement']['ARI']}
 higher={'correspondence loss':False,'cross-image cosine':False,'cluster entropy':True,'ARI vs frozen K8':True}
 y=[targets[r['batch']]['optimal']['iou'] for r in proxies]
 for name,fn in get.items():
  values=[fn(r) for r in proxies];stat=spearmanr(values,y);selected=(max if higher[name] else min)(proxies,key=fn)
  correlations[name]=dict(spearman_rho=float(stat.statistic),nominal_p=float(stat.pvalue),direction='maximize' if higher[name] else 'minimize',selected_batch=selected['batch'],selected_value=fn(selected),selected_oracle_iou=targets[selected['batch']]['optimal']['iou'])
 lines += ['Producers: `audit/scratch/baseline/proxies.py --start A --stop B` → five `proxies_A_B.json` chunks; this `report.py` computes Spearman correlations against the fresh per-checkpoint oracle scores from `forensics/task5.py`. All 50 saved checkpoints, batches 100–5000. Correspondence loss uses the SAME 16 train batches and kNN partners/seeds as FORENSICS Task 4, dropout off, and the actual correspondence helper with restored-mean centering. Frozen activations are reused only after exact visual-trunk parameter equality and live feature/code equality checks. No label files are opened by the proxy scripts. Cross-image cosine/entropy and ARI use all 472 validation images without masks. ARI reference is the predeclared all-train K=8, uncentered, seed-0 baseline.', '',
 'Proxy selection directions were set before examining correlations: minimize objective/cosine; maximize entropy/ARI. These directions are diagnostic heuristics, not established semantic-quality criteria. All correlations and reported quality at the selected batch use validation labels through oracle IoU. The checkpoints share one training trajectory, so nominal Spearman p-values do not account for serial dependence and should not be interpreted as independent-test significance.', '',
 '| Proxy | Spearman rho with oracle IoU | Nominal p | Direction | Selected batch | Selected value | Selected oracle IoU % | Within batches 500–600? |','|---|---:|---:|---|---:|---:|---:|---|']
 for name,r in correlations.items():lines.append(f'| {name} | {r["spearman_rho"]:.6f} | {r["nominal_p"]:.3g} | {r["direction"]} | {r["selected_batch"]} | {r["selected_value"]:.9f} | {fmt(r["selected_oracle_iou"])} | {"Yes" if 500<=r["selected_batch"]<=600 else "No"} |')
 lines += ['', 'Interpretation: lower correspondence loss favors a late, lower-quality checkpoint despite its strong positive correlation with IoU (the desired minimization direction is reversed). Higher entropy likewise favors poorer late checkpoints. Minimum code cosine selects batch 500, near the quality peak, but its rank correlation is weak. Maximum ARI selects batch 700, one saved checkpoint after batch 600: nearby, but past the peak. Neither the weak cosine association nor the modest ARI association establishes a transferable label-free selector.', '', '| Batch | Correspondence loss | Code cosine | Cluster entropy | ARI vs frozen K8 | Oracle liver IoU % |','|---:|---:|---:|---:|---:|---:|']
 for r in proxies:lines.append(f'| {r["batch"]} | {r["losses"]["weighted_correspondence"]:.9f} | {r["monitors"]["cross_image_cosine"]:.9f} | {r["monitors"]["cluster_entropy"]:.9f} | {r["agreement"]["ARI"]:.9f} | {fmt(targets[r["batch"]]["optimal"]["iou"])} |')
else:lines += [f'UNVERIFIED/incomplete: {len(proxies)} of 50 proxy measurements are currently available.']
lines += ['', '## Fit diagnostics, execution and provenance','', '| Fit ID | Iterations (selected initialization) | Inertia | Runtime seconds | Warnings |','|---|---:|---:|---:|---|']
for r in rows[:12]:lines.append(f'| {r["id"]} | {r["iterations"]} | {r["inertia"]:.6f} | {r["elapsed_seconds"]:.3f} | {"; ".join(r["warnings"]) or "None"} |')
verification=json.loads((OUT/'verification.json').read_text())
centered8=[r for r in rows[:12] if r['K']==8 and r['centered']];uncentered8=[r for r in rows[:12] if r['K']==8 and not r['centered']]
min_inertia=min(centered8,key=lambda r:r['inertia'])
lines += ['',f'Seed sensitivity: uncentered K=8 yields {fmt(uncentered8[0]["optimal"]["iou"])}% liver IoU in all three seeds. Centered K=8 ranges from {fmt(min(r["optimal"]["iou"] for r in centered8))}% to {fmt(max(r["optimal"]["iou"] for r in centered8))}%. Its lowest-inertia seed ({min_inertia["seed"]}, inertia {min_inertia["inertia"]:.6f}) has liver IoU {fmt(min_inertia["optimal"]["iou"])}%; selecting the highest-IoU centered seed would itself use evaluation labels. Producers: `kmeans.py` fit outputs and this `report.py` aggregation. These runs do not establish that centering is reliably better.', '',
'`verify.py` validates the AST-reused confusion/mapping routines on synthetic K=8/K=16 data, all confusion totals, slice/patch counts, and pooled-LOPO ≤ global-optimal IoU. It also retains source hashes and the empty DIAG-probe artifact search in `verification.json`. Uncentered K=8 cross-seed partition ARIs (no labels): '+', '.join(f'{k}: {v:.9f}' for k,v in verification['label_free_partition_ARI'].items() if k.startswith('K8 center0'))+'.']
lines += ['', 'Iteration count reports the selected initialization; it does not establish convergence of every restart. No early stop or tuning is based on evaluation scores. K-means fitting is the only fitting performed in this investigation; no neural/supervised training, installs, derivatives, pipeline caches or checkpoints were created/changed. Scratch arrays are diagnostic intermediates, not pipeline caches.', '',
'The initial Anaconda-runtime launch failed during dependency import when the read-only guard rejected a dependency opening `/dev/null` for writing (`extract.log`). It was rerun with the project/FORENSICS Python 3.14 environment (`extract_py314.log`), without changing packages or loosening the guard. An initial report assembly ran before the last fit JSON existed (`report.log`); it was rerun after all fits completed. Neither failure affected computed fit/evaluation results. Successful scripts enforce a 570-second hard timeout. Feature extraction took '+str(extract['elapsed_seconds'])+' seconds. Each k-means fit, head rescoring and proxy chunk is separately bounded. All source imports suppress bytecode writes.', '',
'Checkpoint paths (existing files only), from `heads.py`/`heads.json`:', '']
for r in rows[12:]:lines.append(f'- {r["id"]}: `{r["checkpoint"]}`.')
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();tracked=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=ROOT,text=True).strip()
lines += ['',f'Source commit: `{commit}`. Tracked changes relative to HEAD: `{tracked or "none"}`. Existing untracked run/audit artifacts were left in place. No source edits were made.', '',
'## Limits / UNVERIFIED','',
'- DIAG supervised linear-probe rescoring/LOPO: UNVERIFIED because the fitted estimator was not saved; a recoverable fitted artifact is required. No supervised reference was refitted.',
'- Generalization beyond the four validation patients and this acquisition cohort is UNVERIFIED. LOPO separates mapping labels from scoring labels, but all four patients were already used in the broader project’s diagnostics/checkpoint discussions; it is not a new untouched test cohort. In particular, the existing batch-600 checkpoint was selected by validation cluster mIoU (`audit/run_records/fixes3_full_C/postrun_evidence.json`, callback monitor `test/cluster/mIoU`); checkpoint selection is NOT nested within these LOPO folds. Only mapping calibration is held out.',
'- Global optimal mappings are optimistic, and their subset choice is not an unsupervised prediction rule. Even LOPO needs three annotated calibration patients.',
'- A semantic-quality selector validated across independent runs/patients is UNVERIFIED. Within-trajectory correlations are descriptive, not evidence that a proxy will generalize.',
'- Dense-feature k-means and the trained-head inference use their native, different interpolation paths. This is a same-sample, same-mapping comparison, not an ablation isolating interpolation or resolution effects.',
'- No clinical boundary-quality claim is made from overlap scores alone.']
km=rows[:12];trained=rows[12:];best_head=max(trained,key=lambda r:r['optimal']['iou'])
verdict=[f'The trained head does **not** beat the frozen-feature baseline here. Across all requested K-means fits, global optimal liver IoU spans {fmt(min(r["optimal"]["iou"] for r in km))}–{fmt(max(r["optimal"]["iou"] for r in km))}%, versus at most {fmt(best_head["optimal"]["iou"])}% for the five heads ({best_head["method"]}). Same held-out slices and mapping definitions; fitting labels are never used.',
f'LOPO head mean IoU spans {fmt(min(r["lopo_summary"]["iou"]["mean"] for r in trained))}–{fmt(max(r["lopo_summary"]["iou"]["mean"] for r in trained))}%; K-means spans {fmt(min(r["lopo_summary"]["iou"]["mean"] for r in km))}–{fmt(max(r["lopo_summary"]["iou"]["mean"] for r in km))}%. The per-patient/pooled tables quantify transfer without fitting a mapping on its held-out patient.']
verdict.append('Mapping-transfer loss measured with the SAME pixel pooling: '+ '; '.join(f'{r["id"]} {fmt(r["optimal"]["iou"])} → {fmt(r["lopo_pooled"]["iou"])}% IoU' for r in trained)+'. Thus the head deficit is not explained mainly by choosing its mapping on the scored patient. This does not eliminate checkpoint-selection optimism.')
if correlations:verdict.append('Label-free proxy correlations with oracle liver IoU: '+ '; '.join(f'{name} rho={r["spearman_rho"]:.3f}, extremum at batch {r["selected_batch"]}' for name,r in correlations.items())+'. This is one trajectory and an evaluation-label-based analysis; it does not establish a reliable selection rule.')
verdict.append('UNVERIFIED: the unsaved DIAG linear probe on all 472 slices; generalization beyond four validation patients; label-free checkpoint selection on independent runs.')
lines[lines.index('VERDICT_PLACEHOLDER')]='\n\n'.join(verdict)
(ROOT/'audit/BASELINE.md').write_text('\n'.join(lines)+'\n')
summary=dict(script=__file__,rows=rows,proxy_correlations=correlations,proxy_count=len(proxies),verdict=verdict,commit=commit,tracked_changes=tracked,elapsed_seconds=elapsed())
dump('summary.json',summary)
print('\n'.join(verdict),flush=True)
