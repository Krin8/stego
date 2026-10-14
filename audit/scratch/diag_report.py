"""Assemble only diagnostic results; every displayed number is sourced from a named producer."""
from diag_common import *
from PIL import Image
P=json.loads((OUT/'plan.json').read_text());C={};A={}
for b in [12,11,10,9,8]:
 for c in [0,1]:
    C[b,c]=json.loads((OUT/f'ceiling_b{b}_c{c}.json').read_text())
    A[b,c]=json.loads((OUT/f'affinity_b{b}_c{c}.json').read_text())
D=json.loads((OUT/'collapse.json').read_text())
def table(header,rows):return '\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def pct(x):return f'{100*x:.2f}'
def fl(x):return f'{x:.4f}'
def src(b):return 'Final (12)' if b==12 else f'Block {b}'
def yn(c):return 'Yes' if c else 'No'
ceiling_rows=[];candidate_rows=[]
for (b,c),d in C.items():
 ceiling_rows.append([src(b),yn(c),*[pct(d['kmeans'][str(k)]['majority_oracle']['liver_iou']) for k in [2,4,8]],pct(d['linear']['liver_iou'])])
 e=A[b,c];cc=e['candidates'];candidate_rows.append([src(b),yn(c),' / '.join(fl(v) for v in cc[0]['shifts_intra_inter_neg'][:2]),' / '.join(fl(x['shifts_intra_inter_neg'][2]) for x in cc),' / '.join(f'{100*x["attraction"]["knn"][2]:.1f}' for x in cc),' / '.join(f'{x["constant_code_correspondence_loss"]:+.4f}' for x in cc)])
t1=table(['Source','Train mean subtracted?','K=2 IoU %','K=4 IoU %','K=8 IoU %','Linear IoU %'],ceiling_rows)
t2=table(['Source','Mean subtracted?','Intra / kNN shifts','Negative shifts for 20% / 15% / 10% attraction','kNN attraction at those SAME negative shifts %','Constant-code loss at those sets'],candidate_rows)
text=['# CHAOS v2 CT: feature ceiling, affinity calibration and collapse diagnosis',
'''## Verdict

The frozen features contain liver information; the trained head's collapse does not establish that BiomedCLIP is unusable. The fixed diagnostic linear fits are all near61–62% liver IoU, with the highest observed value in block9 without mean subtraction. Majority-oracle K=8 reaches its highest observed value with mean-subtracted final-block features; raw block11 is close. These are descriptive validation results, not automatic choices of an unsupervised model or evidence of precise organ boundaries. K=2/4 majority-oracle scores are zero in every variant: no cluster has majority liver pixels. That does NOT mean the grid contains no liver signal. Hungarian K=2 scores below show the effect of requiring one cluster to represent liver.

The step3600 head has direction collapse on all tested validation patients. No non-finite values were found. Raw amplitudes vary, so literal identical raw vectors would be inaccurate; near-identical normalized directions and single-cluster assignment are the supported claim.

For calibration, keep the approved restored-global-mean helper. The user's attraction-fraction heuristic is NOT from Paper A. Candidate positive shifts use their pair-type25th percentiles (75% attraction), and negative shifts use random-pair80th/85th/90th percentiles (20/15/10% attraction). The20% and15% candidates still give negative constant-code correspondence loss in every source; satisfying that heuristic alone can continue to reward constant alignment. The10% candidates without mean subtraction give slightly positive constant-code loss and are better starting controls, not verified fixes. A useful current-source control is final-block/no-mean: (intra,inter,negative)=(0.4584,0.4525,0.7792). Block9/no-mean's analogous set is (0.2044,0.1970,0.5749). No configs were changed, and no claim that these settings will learn a good segmentation is verified.

At a SINGLE shared threshold, no source meets both random attraction<=20% and majority kNN attraction: even the least restrictive negative candidate leaves fewer than half the kNN patch pairs attractive. The separate positive/negative shifts satisfy the requested role-specific fractions by construction, not because the affinity distributions cleanly separate.
''',
'''## Protocol and provenance

Producers are executable scripts in `audit/scratch/`, importing the actual project modules. Run with `python3 -B audit/scratch/<script>.py`; per-source scripts require `--block 12|11|10|9|8 --center 0|1`. Detailed JSON is in `audit/scratch/diag/`. The shared guard `diag_common.py` enforces CPU-only execution, offline access, writes only under audit/scratch or audit/DIAG.md, and a570-second alarm per process. No project source, dataset, existing kNN cache or checkpoint was changed by these diagnostics. No training loop, optimizer step on the backbone/head, package install or new pipeline cache was run. Scratch .npy arrays are diagnostic intermediates only.

`diag_plan.py` fixes seed20260930 before reading labels: 200 train slices over16 patients (12 or13 per patient) and100 validation slices over4 patients (25 each), with one seeded draw per physical-order slice stratum. `diag/plan.json` preserves every sample ID/index. The four displayed validation slices are fixed middle selections, not selected by organ size or model score. Twenty checkpoint-test slices are a fixed evenly spaced subset of the100.

`diag_extract.py --split train|val` imports `modules.BiomedCLIPFeaturizer` and the actual `data.CHAOS`/`chaos_preprocessing` transform. A runtime wrapper around the extractor's own `get_intermediate_layers` call captures timm block indices7..11 (one-based8..12), with reshape=True and norm=True, then returns the original final-block output. Final output equality,768 channels,14×14 grid, frozen/eval state and finite values are asserted. CLS is excluded by the actual timm reshape path. No extractor/affinity formula is reimplemented. Normalized block tokens mean the trunk's LayerNorm; ceiling comparisons additionally use the user-approved per-patch L2 normalization after optional train-feature-mean subtraction.

`diag_ceiling.py`: training mean uses all39200 train patch vectors; validation contributes no fitting statistics. Full Euclidean KMeans fits normalized vectors using random_state20260930, n_init3, max_iter100, tol1e-4 for K2/4/8. Validation patch assignments are nearest-neighbour upsampled from14×14 to224×224. Mapping is computed ONLY for scoring from validation counts: each cluster becomes liver iff liver pixels strictly outnumber background pixels; ties map to background. This majority rule maximizes classification accuracy, not necessarily IoU. It is an optimistic validation-label oracle, not a deployable unsupervised class naming rule. K2 Hungarian uses the same scoring-only confusion counts and maximizes correct pixels.

The user explicitly authorized supervised diagnostic linear fitting as the sole labels-for-fitting exception: LogisticRegression(C=1, class_weight=balanced, solver=lbfgs, max_iter200, tol1e-5), with nearest-neighbour14×14 training labels; no hyperparameter/threshold tuning. Validation logits are bilinearly resized to224×224, align_corners=False, then thresholded at0. All fits converged without recorded warnings. No probe model is saved. This particular regularized probe is an empirical recoverability diagnostic, NOT a mathematical upper bound on all possible decoders. All IoU/Dice scores pool pixels across the100 validation slices; they are not per-volume scores.

PCA is fitted exclusively to normalized training vectors (3 randomized components, fixed seed); RGB display clipping uses the training component1st/99th percentiles. No mask enters PCA or color scaling. Color axes can differ between sources. Each saved PNG contains the input, ground-truth overlay and native-grid PCA enlarged by nearest-neighbour. Top3 PCA components are not all768 dimensions: weak liver isolation in PCA alone does not negate the classification results.
''',
'## Table 1 — frozen-feature scoring\n\nAll scores below: `diag_ceiling.py`, outputs `diag/ceiling_b*_c*.json`. Units are percentages.\n\n'+t1,
'### Dice and Hungarian details\n\n'+table(['Source','Mean?','K2 majority Dice','K4 majority Dice','K8 majority Dice','K2 Hungarian IoU / Dice','Linear Dice'],[[src(b),yn(c),*[pct(d['kmeans'][str(k)]['majority_oracle']['liver_dice']) for k in [2,4,8]],pct(d['kmeans']['2']['hungarian_oracle']['liver_iou'])+' / '+pct(d['kmeans']['2']['hungarian_oracle']['liver_dice']),pct(d['linear']['liver_dice'])] for (b,c),d in C.items()]),
'''## Affinity measurement

`diag_affinity.py` imports and calls the actual `modules.sample` and `ContrastiveCorrelationLoss.helper`. Sampling uses the configured11×11 random coordinates. It supplies one-channel constant-one codes and shift0; actual helper output has cd==1 (asserted), so minus its returned loss gives the exact centered feature affinity. This retains project channel normalization, row centering and global-mean restoration. No reimplementation of those operations is used. Earlier-layer candidates pass their captured normalized tokens to this same helper; mean-subtracted variants subtract the train-only mean before sampling. Unlike k-means preprocessing, channel L2 occurs AFTER bilinear sampling, inside the real helper, matching training. Dropout is disabled for this frozen-feature measurement; actual training-time stochastic-dropout affinity quantiles remain UNVERIFIED.

Intra pairs use each sampled feature grid against itself, as in the real loss. Random pairs exclude the same image; cross-patient pairs also exclude the same patient. Each of these has200 image pairs and2928200 patch pairs. Existing-cache pairs use all55 directed top-seven edges retained inside the fixed200-slice sample (805255 patch pairs,54 anchors). No extra images or new neighbours were added. Every source uses the SAME existing final-block cache; source-specific neighbour ranking is UNVERIFIED. Cache identity was checked using the actual `knn_cache.expected_identity/read_cache` against current source/weights/data. Global mean is restored within actual helper batches of at most16 image pairs. Affinities after centering need not stay inside[-1,1].

Current shift order is intra/inter/negative=(0.18,0.12,0.46). Quantiles/current fractions below come from `diag_affinity.py`. The candidate constant-code loss is the configured weighted mean of actual helper-derived affinities after the additive shift; weights remain intra0.67/inter0.25/negative0.63. Zero code-correlation gives correspondence loss0. This scalar test only compares degenerate behaviors on these sampled pair pools; it does not predict optimizer dynamics or prove non-collapse.
''',
'## Table 2 — candidate shift sets\n\n`diag_affinity.py`, outputs `diag/affinity_b*_c*.json`. For EVERY set, intra attraction is75%, kNN attraction at its own positive shift is75%, and random attraction at the negative shift is20%/15%/10% respectively. The same-negative-threshold kNN column exposes overlap rather than conflating the two different thresholds. Lower loss is preferred.\n\n'+t2,
'### Exact affinity quantiles\n\n'+table(['Source','Mean?','Pair type','q5','q25','q50','q75','q90','q95','q99'],[[src(b),yn(c),k,*[fl(v['quantiles'][str(q)]) for q in [5,25,50,75,90,95,99]]] for (b,c),d in A.items() for k,v in d['quantiles'].items()]),
'### Attraction under all three current shifts (%)\n\n'+table(['Source','Mean?','Pair type','intra0.18','inter0.12','negative0.46'],[[src(b),yn(c),k,*[pct(z) for z in v['attraction_current_intra_inter_neg']]] for (b,c),d in A.items() for k,v in d['quantiles'].items()]),
'### Movement across candidate sets (%)\n\n'+table(['Source','Mean?','Random target','Intra at own shift','kNN at own shift','Random at neg','Cross-patient at neg','kNN at neg'],[[src(b),yn(c),pct(x['target_random_attraction']),pct(x['attraction']['intra'][0]),pct(x['attraction']['knn'][1]),pct(x['attraction']['random'][2]),pct(x['attraction']['cross_patient'][2]),pct(x['attraction']['knn'][2])] for (b,c),d in A.items() for x in d['candidates']]),
'## Step3600 collapse — `diag_collapse.py`\n\n'+json.dumps({k:v for k,v in D.items() if k!='per_slice'},indent=2)+'''\n
Codes are measured before L2 normalization and after per-pixel L2 normalization. Reported std is the population std over pixels, averaged over channels (also within-image mean and min/max per-channel values in JSON). Cross-image cosine averages all patch pairs between distinct images; cross-patient restricts to distinct patients. The script imports the real extractor and probe, strictly loads the saved net/probe state, and performs actual CPU forwards. It does not use the previously diagnosed four-slice shortcut.

**Coverage requirement not met:** ten CT validation patients do not exist. All four available patients are covered. Changing the split or relabelling training patients as validation would invalidate this diagnosis; wider patient generalization remains UNVERIFIED.
''',
'## Image paths — `diag_ceiling.py`\n\n']
for (b,c),d in C.items():
 text.append(f'### {src(b)}, train mean subtraction: {yn(c)}\n\n'+'\n'.join(f'- [{Path(p).name}]({ROOT/p})' for p in d['pca']['paths']))
# Compact overview is just a montage of the already-generated scientific images.
contact=Image.new('RGB',(1344,1280),'white')
for row,((b,c),d) in enumerate(C.items()):
 for col,p in enumerate(d['pca']['paths']):
    with Image.open(ROOT/p) as im:contact.paste(im.resize((336,128)),(col*336,row*128))
contact.save(OUT/'pca_overview.png')
text.append(f'\n[All PCA/overlay panels]({OUT/"pca_overview.png"}) — montage produced by `diag_report.py`; full-size panels above.\n')
runtimes=[json.loads((OUT/f'extract_{s}.json').read_text())['elapsed_seconds'] for s in ['train','val']]+[d['elapsed_seconds'] for d in C.values()]+[d['elapsed_seconds'] for d in A.values()]+[P['elapsed_seconds'],D['elapsed_seconds']]
text.append('## Verification limits and next decisions\n\n'+f'Completed producer runtimes range{min(runtimes):.2f}–{max(runtimes):.2f}s, all below570s (each producer JSON records elapsed time). '+'''Seed/sample are fixed, with no validation-based fitting or threshold selection; source score comparisons are descriptive. Small between-layer differences are not established as statistically meaningful. No extra seeds, patient confidence intervals, actual boundary metric, source-specific kNN cache, full-val scoring, full-resolution decoder, training-time dropout calibration or retraining was run. The55-edge kNN subset is a material limitation; do not treat its millions of correlated patch pairs as millions of independent observations.

Recommendations only: first establish non-collapse with a train-only calibrated negative shift and unchanged restored-mean helper; log code-direction variability and cluster usage. Treat the10% candidates without mean subtraction as controlled starting hypotheses. Test actual training/dropout distributions before a full run. The fixed K2 probe partitions broad appearance groups, while K8 oracle scoring exposes recoverable liver information; investigate granularity without using validation labels to fit or select the unsupervised model. Do not promise precise liver contours from this14×14 grid or infer that a linear diagnostic is an unsupervised result. Hyperparameters, source choice and whether a candidate produces coherent anatomy after optimization all remain UNVERIFIED until separately authorized experiments.\n''')
(ROOT/'audit/DIAG.md').write_text('\n\n'.join(text))
dump('summary.json',dict(feature_table=ceiling_rows,shift_table=candidate_rows,producer_runtime_range_seconds=[min(runtimes),max(runtimes)],report=str(ROOT/'audit/DIAG.md')))
print(t1+'\n\n'+t2)
