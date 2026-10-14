# CHAOS v2 CT: feature ceiling, affinity calibration and collapse diagnosis

## Verdict

The frozen features contain liver information; the trained head's collapse does not establish that BiomedCLIP is unusable. The fixed diagnostic linear fits are all near61–62% liver IoU, with the highest observed value in block9 without mean subtraction. Majority-oracle K=8 reaches its highest observed value with mean-subtracted final-block features; raw block11 is close. These are descriptive validation results, not automatic choices of an unsupervised model or evidence of precise organ boundaries. K=2/4 majority-oracle scores are zero in every variant: no cluster has majority liver pixels. That does NOT mean the grid contains no liver signal. Hungarian K=2 scores below show the effect of requiring one cluster to represent liver.

The step3600 head has direction collapse on all tested validation patients. No non-finite values were found. Raw amplitudes vary, so literal identical raw vectors would be inaccurate; near-identical normalized directions and single-cluster assignment are the supported claim.

For calibration, keep the approved restored-global-mean helper. The user's attraction-fraction heuristic is NOT from Paper A. Candidate positive shifts use their pair-type25th percentiles (75% attraction), and negative shifts use random-pair80th/85th/90th percentiles (20/15/10% attraction). The20% and15% candidates still give negative constant-code correspondence loss in every source; satisfying that heuristic alone can continue to reward constant alignment. The10% candidates without mean subtraction give slightly positive constant-code loss and are better starting controls, not verified fixes. A useful current-source control is final-block/no-mean: (intra,inter,negative)=(0.4584,0.4525,0.7792). Block9/no-mean's analogous set is (0.2044,0.1970,0.5749). No configs were changed, and no claim that these settings will learn a good segmentation is verified.

At a SINGLE shared threshold, no source meets both random attraction<=20% and majority kNN attraction: even the least restrictive negative candidate leaves fewer than half the kNN patch pairs attractive. The separate positive/negative shifts satisfy the requested role-specific fractions by construction, not because the affinity distributions cleanly separate.


## Protocol and provenance

Producers are executable scripts in `audit/scratch/`, importing the actual project modules. Run with `python3 -B audit/scratch/<script>.py`; per-source scripts require `--block 12|11|10|9|8 --center 0|1`. Detailed JSON is in `audit/scratch/diag/`. The shared guard `diag_common.py` enforces CPU-only execution, offline access, writes only under audit/scratch or audit/DIAG.md, and a570-second alarm per process. No project source, dataset, existing kNN cache or checkpoint was changed by these diagnostics. No training loop, optimizer step on the backbone/head, package install or new pipeline cache was run. Scratch .npy arrays are diagnostic intermediates only.

`diag_plan.py` fixes seed20260930 before reading labels: 200 train slices over16 patients (12 or13 per patient) and100 validation slices over4 patients (25 each), with one seeded draw per physical-order slice stratum. `diag/plan.json` preserves every sample ID/index. The four displayed validation slices are fixed middle selections, not selected by organ size or model score. Twenty checkpoint-test slices are a fixed evenly spaced subset of the100.

`diag_extract.py --split train|val` imports `modules.BiomedCLIPFeaturizer` and the actual `data.CHAOS`/`chaos_preprocessing` transform. A runtime wrapper around the extractor's own `get_intermediate_layers` call captures timm block indices7..11 (one-based8..12), with reshape=True and norm=True, then returns the original final-block output. Final output equality,768 channels,14×14 grid, frozen/eval state and finite values are asserted. CLS is excluded by the actual timm reshape path. No extractor/affinity formula is reimplemented. Normalized block tokens mean the trunk's LayerNorm; ceiling comparisons additionally use the user-approved per-patch L2 normalization after optional train-feature-mean subtraction.

`diag_ceiling.py`: training mean uses all39200 train patch vectors; validation contributes no fitting statistics. Full Euclidean KMeans fits normalized vectors using random_state20260930, n_init3, max_iter100, tol1e-4 for K2/4/8. Validation patch assignments are nearest-neighbour upsampled from14×14 to224×224. Mapping is computed ONLY for scoring from validation counts: each cluster becomes liver iff liver pixels strictly outnumber background pixels; ties map to background. This majority rule maximizes classification accuracy, not necessarily IoU. It is an optimistic validation-label oracle, not a deployable unsupervised class naming rule. K2 Hungarian uses the same scoring-only confusion counts and maximizes correct pixels.

The user explicitly authorized supervised diagnostic linear fitting as the sole labels-for-fitting exception: LogisticRegression(C=1, class_weight=balanced, solver=lbfgs, max_iter200, tol1e-5), with nearest-neighbour14×14 training labels; no hyperparameter/threshold tuning. Validation logits are bilinearly resized to224×224, align_corners=False, then thresholded at0. All fits converged without recorded warnings. No probe model is saved. This particular regularized probe is an empirical recoverability diagnostic, NOT a mathematical upper bound on all possible decoders. All IoU/Dice scores pool pixels across the100 validation slices; they are not per-volume scores.

PCA is fitted exclusively to normalized training vectors (3 randomized components, fixed seed); RGB display clipping uses the training component1st/99th percentiles. No mask enters PCA or color scaling. Color axes can differ between sources. Each saved PNG contains the input, ground-truth overlay and native-grid PCA enlarged by nearest-neighbour. Top3 PCA components are not all768 dimensions: weak liver isolation in PCA alone does not negate the classification results.


## Table 1 — frozen-feature scoring

All scores below: `diag_ceiling.py`, outputs `diag/ceiling_b*_c*.json`. Units are percentages.

| Source | Train mean subtracted? | K=2 IoU % | K=4 IoU % | K=8 IoU % | Linear IoU % |
| --- | --- | --- | --- | --- | --- |
| Final (12) | No | 0.00 | 0.00 | 64.16 | 60.89 |
| Final (12) | Yes | 0.00 | 0.00 | 65.51 | 60.84 |
| Block 11 | No | 0.00 | 0.00 | 64.99 | 61.78 |
| Block 11 | Yes | 0.00 | 0.00 | 48.32 | 61.42 |
| Block 10 | No | 0.00 | 0.00 | 47.66 | 61.64 |
| Block 10 | Yes | 0.00 | 0.00 | 47.43 | 61.41 |
| Block 9 | No | 0.00 | 0.00 | 64.50 | 61.99 |
| Block 9 | Yes | 0.00 | 0.00 | 0.00 | 61.67 |
| Block 8 | No | 0.00 | 0.00 | 61.73 | 61.24 |
| Block 8 | Yes | 0.00 | 0.00 | 0.00 | 61.00 |

### Dice and Hungarian details

| Source | Mean? | K2 majority Dice | K4 majority Dice | K8 majority Dice | K2 Hungarian IoU / Dice | Linear Dice |
| --- | --- | --- | --- | --- | --- | --- |
| Final (12) | No | 0.00 | 0.00 | 78.17 | 17.50 / 29.79 | 75.69 |
| Final (12) | Yes | 0.00 | 0.00 | 79.16 | 15.84 / 27.35 | 75.65 |
| Block 11 | No | 0.00 | 0.00 | 78.78 | 18.33 / 30.99 | 76.37 |
| Block 11 | Yes | 0.00 | 0.00 | 65.16 | 16.02 / 27.62 | 76.10 |
| Block 10 | No | 0.00 | 0.00 | 64.55 | 17.70 / 30.08 | 76.27 |
| Block 10 | Yes | 0.00 | 0.00 | 64.35 | 15.89 / 27.42 | 76.09 |
| Block 9 | No | 0.00 | 0.00 | 78.42 | 17.31 / 29.51 | 76.53 |
| Block 9 | Yes | 0.00 | 0.00 | 0.00 | 16.04 / 27.65 | 76.29 |
| Block 8 | No | 0.00 | 0.00 | 76.34 | 16.82 / 28.80 | 75.96 |
| Block 8 | Yes | 0.00 | 0.00 | 0.00 | 0.00 / 0.00 | 75.78 |

## Affinity measurement

`diag_affinity.py` imports and calls the actual `modules.sample` and `ContrastiveCorrelationLoss.helper`. Sampling uses the configured11×11 random coordinates. It supplies one-channel constant-one codes and shift0; actual helper output has cd==1 (asserted), so minus its returned loss gives the exact centered feature affinity. This retains project channel normalization, row centering and global-mean restoration. No reimplementation of those operations is used. Earlier-layer candidates pass their captured normalized tokens to this same helper; mean-subtracted variants subtract the train-only mean before sampling. Unlike k-means preprocessing, channel L2 occurs AFTER bilinear sampling, inside the real helper, matching training. Dropout is disabled for this frozen-feature measurement; actual training-time stochastic-dropout affinity quantiles remain UNVERIFIED.

Intra pairs use each sampled feature grid against itself, as in the real loss. Random pairs exclude the same image; cross-patient pairs also exclude the same patient. Each of these has200 image pairs and2928200 patch pairs. Existing-cache pairs use all55 directed top-seven edges retained inside the fixed200-slice sample (805255 patch pairs,54 anchors). No extra images or new neighbours were added. Every source uses the SAME existing final-block cache; source-specific neighbour ranking is UNVERIFIED. Cache identity was checked using the actual `knn_cache.expected_identity/read_cache` against current source/weights/data. Global mean is restored within actual helper batches of at most16 image pairs. Affinities after centering need not stay inside[-1,1].

Current shift order is intra/inter/negative=(0.18,0.12,0.46). Quantiles/current fractions below come from `diag_affinity.py`. The candidate constant-code loss is the configured weighted mean of actual helper-derived affinities after the additive shift; weights remain intra0.67/inter0.25/negative0.63. Zero code-correlation gives correspondence loss0. This scalar test only compares degenerate behaviors on these sampled pair pools; it does not predict optimizer dynamics or prove non-collapse.


## Table 2 — candidate shift sets

`diag_affinity.py`, outputs `diag/affinity_b*_c*.json`. For EVERY set, intra attraction is75%, kNN attraction at its own positive shift is75%, and random attraction at the negative shift is20%/15%/10% respectively. The same-negative-threshold kNN column exposes overlap rather than conflating the two different thresholds. Lower loss is preferred.

| Source | Mean subtracted? | Intra / kNN shifts | Negative shifts for 20% / 15% / 10% attraction | kNN attraction at those SAME negative shifts % | Constant-code loss at those sets |
| --- | --- | --- | --- | --- | --- |
| Final (12) | No | 0.4584 / 0.4525 | 0.7164 / 0.7460 / 0.7792 | 31.3 / 26.2 / 20.5 | -0.0294 / -0.0107 / +0.0102 |
| Final (12) | Yes | -0.1937 / -0.2150 | 0.3400 / 0.4140 / 0.5010 | 30.5 / 25.6 / 19.9 | -0.0993 / -0.0527 / +0.0021 |
| Block 11 | No | 0.3270 / 0.3229 | 0.6740 / 0.7109 / 0.7509 | 27.5 / 22.5 / 17.1 | -0.0366 / -0.0133 / +0.0119 |
| Block 11 | Yes | -0.2678 / -0.2799 | 0.3506 / 0.4285 / 0.5211 | 27.1 / 22.1 / 16.6 | -0.1089 / -0.0598 / -0.0015 |
| Block 10 | No | 0.2580 / 0.2492 | 0.5654 / 0.6063 / 0.6561 | 28.6 / 23.6 / 17.8 | -0.0526 / -0.0268 / +0.0046 |
| Block 10 | Yes | -0.1956 / -0.2136 | 0.2533 / 0.3251 / 0.4192 | 28.3 / 23.3 / 17.6 | -0.1047 / -0.0595 / -0.0002 |
| Block 9 | No | 0.2044 / 0.1970 | 0.4797 / 0.5212 / 0.5749 | 28.5 / 23.4 / 17.6 | -0.0539 / -0.0278 / +0.0060 |
| Block 9 | Yes | -0.1727 / -0.1891 | 0.2136 / 0.2752 / 0.3586 | 28.1 / 22.9 / 17.3 | -0.0993 / -0.0605 / -0.0079 |
| Block 8 | No | 0.1626 / 0.1586 | 0.4430 / 0.4873 / 0.5456 | 25.5 / 20.6 / 15.3 | -0.0516 / -0.0237 / +0.0130 |
| Block 8 | Yes | -0.1825 / -0.1936 | 0.2043 / 0.2668 / 0.3510 | 25.2 / 20.2 / 15.2 | -0.0902 / -0.0509 / +0.0022 |

### Exact affinity quantiles

| Source | Mean? | Pair type | q5 | q25 | q50 | q75 | q90 | q95 | q99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Final (12) | No | intra | 0.2671 | 0.4584 | 0.6132 | 0.7673 | 0.8635 | 0.9143 | 1.0299 |
| Final (12) | No | knn | 0.2572 | 0.4525 | 0.6008 | 0.7530 | 0.8471 | 0.8941 | 0.9937 |
| Final (12) | No | random | 0.2272 | 0.4217 | 0.5567 | 0.6884 | 0.7792 | 0.8235 | 0.9072 |
| Final (12) | No | cross_patient | 0.2319 | 0.4202 | 0.5570 | 0.6898 | 0.7809 | 0.8259 | 0.9101 |
| Final (12) | Yes | intra | -0.4261 | -0.1937 | 0.1077 | 0.4617 | 0.7153 | 0.8381 | 1.0176 |
| Final (12) | Yes | knn | -0.4341 | -0.2150 | 0.0685 | 0.4238 | 0.6747 | 0.7910 | 0.9594 |
| Final (12) | Yes | random | -0.5011 | -0.2818 | -0.0275 | 0.2728 | 0.5010 | 0.6168 | 0.7899 |
| Final (12) | Yes | cross_patient | -0.5017 | -0.2856 | -0.0319 | 0.2720 | 0.5012 | 0.6183 | 0.7921 |
| Block 11 | No | intra | 0.1029 | 0.3270 | 0.5092 | 0.7070 | 0.8249 | 0.8876 | 1.0349 |
| Block 11 | No | knn | 0.1009 | 0.3229 | 0.4995 | 0.6928 | 0.8086 | 0.8665 | 0.9947 |
| Block 11 | No | random | 0.0772 | 0.3144 | 0.4710 | 0.6376 | 0.7509 | 0.8031 | 0.9076 |
| Block 11 | No | cross_patient | 0.0806 | 0.3103 | 0.4710 | 0.6397 | 0.7527 | 0.8058 | 0.9113 |
| Block 11 | Yes | intra | -0.4924 | -0.2678 | 0.0488 | 0.4129 | 0.6841 | 0.8158 | 1.0035 |
| Block 11 | Yes | knn | -0.4937 | -0.2799 | 0.0198 | 0.3833 | 0.6505 | 0.7738 | 0.9428 |
| Block 11 | Yes | random | -0.5136 | -0.2990 | -0.0263 | 0.2813 | 0.5211 | 0.6441 | 0.8233 |
| Block 11 | Yes | cross_patient | -0.5145 | -0.3024 | -0.0288 | 0.2822 | 0.5236 | 0.6474 | 0.8263 |
| Block 10 | No | intra | 0.0941 | 0.2580 | 0.4172 | 0.6112 | 0.7635 | 0.8542 | 1.0106 |
| Block 10 | No | knn | 0.0850 | 0.2492 | 0.4046 | 0.5947 | 0.7424 | 0.8243 | 0.9672 |
| Block 10 | No | random | 0.0680 | 0.2350 | 0.3724 | 0.5277 | 0.6561 | 0.7309 | 0.8745 |
| Block 10 | No | cross_patient | 0.0693 | 0.2326 | 0.3731 | 0.5317 | 0.6609 | 0.7353 | 0.8796 |
| Block 10 | Yes | intra | -0.3701 | -0.1956 | 0.0307 | 0.3299 | 0.6127 | 0.7729 | 0.9940 |
| Block 10 | Yes | knn | -0.3780 | -0.2136 | 0.0024 | 0.2993 | 0.5784 | 0.7238 | 0.9248 |
| Block 10 | Yes | random | -0.3935 | -0.2340 | -0.0439 | 0.1937 | 0.4192 | 0.5609 | 0.7912 |
| Block 10 | Yes | cross_patient | -0.3969 | -0.2368 | -0.0440 | 0.1970 | 0.4257 | 0.5680 | 0.7962 |
| Block 9 | No | intra | 0.0447 | 0.2044 | 0.3429 | 0.5258 | 0.6937 | 0.8028 | 0.9785 |
| Block 9 | No | knn | 0.0361 | 0.1970 | 0.3309 | 0.5083 | 0.6697 | 0.7667 | 0.9162 |
| Block 9 | No | random | 0.0180 | 0.1876 | 0.3013 | 0.4436 | 0.5749 | 0.6585 | 0.8118 |
| Block 9 | No | cross_patient | 0.0169 | 0.1846 | 0.3017 | 0.4472 | 0.5798 | 0.6637 | 0.8166 |
| Block 9 | Yes | intra | -0.3249 | -0.1727 | 0.0265 | 0.2778 | 0.5446 | 0.7158 | 0.9770 |
| Block 9 | Yes | knn | -0.3306 | -0.1891 | 0.0005 | 0.2493 | 0.5081 | 0.6600 | 0.8881 |
| Block 9 | Yes | random | -0.3456 | -0.2062 | -0.0378 | 0.1636 | 0.3586 | 0.4903 | 0.7397 |
| Block 9 | Yes | cross_patient | -0.3484 | -0.2092 | -0.0375 | 0.1674 | 0.3652 | 0.4981 | 0.7471 |
| Block 8 | No | intra | 0.0065 | 0.1626 | 0.2878 | 0.4630 | 0.6482 | 0.7717 | 0.9679 |
| Block 8 | No | knn | 0.0010 | 0.1586 | 0.2787 | 0.4477 | 0.6244 | 0.7341 | 0.8987 |
| Block 8 | No | random | -0.0043 | 0.1572 | 0.2665 | 0.4060 | 0.5456 | 0.6371 | 0.8020 |
| Block 8 | No | cross_patient | -0.0079 | 0.1536 | 0.2655 | 0.4078 | 0.5495 | 0.6414 | 0.8066 |
| Block 8 | Yes | intra | -0.3243 | -0.1825 | -0.0058 | 0.2279 | 0.5008 | 0.6849 | 0.9698 |
| Block 8 | Yes | knn | -0.3259 | -0.1936 | -0.0242 | 0.2066 | 0.4665 | 0.6282 | 0.8733 |
| Block 8 | Yes | random | -0.3322 | -0.1970 | -0.0396 | 0.1545 | 0.3510 | 0.4837 | 0.7417 |
| Block 8 | Yes | cross_patient | -0.3344 | -0.1998 | -0.0400 | 0.1576 | 0.3563 | 0.4898 | 0.7486 |

### Attraction under all three current shifts (%)

| Source | Mean? | Pair type | intra0.18 | inter0.12 | negative0.46 |
| --- | --- | --- | --- | --- | --- |
| Final (12) | No | intra | 97.93 | 98.88 | 74.75 |
| Final (12) | No | knn | 97.68 | 98.77 | 73.76 |
| Final (12) | No | random | 96.90 | 98.37 | 68.31 |
| Final (12) | No | cross_patient | 97.10 | 98.47 | 68.08 |
| Final (12) | Yes | intra | 44.68 | 49.10 | 25.11 |
| Final (12) | Yes | knn | 41.72 | 46.09 | 22.61 |
| Final (12) | Yes | random | 32.43 | 37.46 | 12.23 |
| Final (12) | Yes | cross_patient | 32.35 | 37.30 | 12.19 |
| Block 11 | No | intra | 90.57 | 94.20 | 56.63 |
| Block 11 | No | knn | 90.28 | 94.08 | 55.30 |
| Block 11 | No | random | 89.17 | 92.94 | 51.86 |
| Block 11 | No | cross_patient | 89.06 | 93.01 | 51.78 |
| Block 11 | Yes | intra | 40.79 | 45.03 | 22.04 |
| Block 11 | Yes | knn | 38.70 | 42.93 | 20.23 |
| Block 11 | Yes | random | 33.06 | 38.07 | 13.19 |
| Block 11 | Yes | cross_patient | 33.06 | 37.99 | 13.27 |
| Block 10 | No | intra | 86.05 | 92.83 | 44.05 |
| Block 10 | No | knn | 84.82 | 91.93 | 42.19 |
| Block 10 | No | random | 83.17 | 90.48 | 34.84 |
| Block 10 | No | cross_patient | 82.86 | 90.42 | 35.28 |
| Block 10 | Yes | intra | 36.46 | 41.72 | 17.05 |
| Block 10 | Yes | knn | 33.94 | 39.03 | 15.40 |
| Block 10 | Yes | random | 26.26 | 32.13 | 8.28 |
| Block 10 | Yes | cross_patient | 26.54 | 32.35 | 8.56 |
| Block 9 | No | intra | 79.34 | 87.99 | 33.27 |
| Block 9 | No | knn | 78.07 | 87.09 | 31.10 |
| Block 9 | No | random | 76.48 | 85.69 | 22.65 |
| Block 9 | No | cross_patient | 75.89 | 85.10 | 23.20 |
| Block 9 | Yes | intra | 33.86 | 39.97 | 13.53 |
| Block 9 | Yes | knn | 31.22 | 37.18 | 12.01 |
| Block 9 | Yes | random | 23.28 | 29.94 | 5.92 |
| Block 9 | Yes | cross_patient | 23.68 | 30.29 | 6.16 |
| Block 8 | No | intra | 71.49 | 82.74 | 25.35 |
| Block 8 | No | knn | 70.56 | 82.12 | 23.55 |
| Block 8 | No | random | 69.95 | 82.05 | 17.95 |
| Block 8 | No | cross_patient | 69.27 | 81.29 | 18.28 |
| Block 8 | Yes | intra | 29.37 | 35.59 | 11.52 |
| Block 8 | Yes | knn | 27.41 | 33.42 | 10.25 |
| Block 8 | Yes | random | 22.33 | 28.93 | 5.70 |
| Block 8 | Yes | cross_patient | 22.66 | 29.24 | 5.88 |

### Movement across candidate sets (%)

| Source | Mean? | Random target | Intra at own shift | kNN at own shift | Random at neg | Cross-patient at neg | kNN at neg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Final (12) | No | 20.00 | 75.00 | 75.00 | 20.00 | 20.27 | 31.26 |
| Final (12) | No | 15.00 | 75.00 | 75.00 | 15.00 | 15.22 | 26.23 |
| Final (12) | No | 10.00 | 75.00 | 75.00 | 10.00 | 10.23 | 20.49 |
| Final (12) | Yes | 20.00 | 75.00 | 75.00 | 20.00 | 19.90 | 30.54 |
| Final (12) | Yes | 15.00 | 75.00 | 75.00 | 15.00 | 14.94 | 25.65 |
| Final (12) | Yes | 10.00 | 75.00 | 75.00 | 10.00 | 10.01 | 19.94 |
| Block 11 | No | 20.00 | 75.00 | 75.00 | 20.00 | 20.29 | 27.54 |
| Block 11 | No | 15.00 | 75.00 | 75.00 | 15.00 | 15.25 | 22.52 |
| Block 11 | No | 10.00 | 75.00 | 75.00 | 10.00 | 10.21 | 17.07 |
| Block 11 | Yes | 20.00 | 75.00 | 75.00 | 20.00 | 20.06 | 27.10 |
| Block 11 | Yes | 15.00 | 75.00 | 75.00 | 15.00 | 15.09 | 22.15 |
| Block 11 | Yes | 10.00 | 75.00 | 75.00 | 10.00 | 10.12 | 16.65 |
| Block 10 | No | 20.00 | 75.00 | 75.00 | 20.00 | 20.53 | 28.61 |
| Block 10 | No | 15.00 | 75.00 | 75.00 | 15.00 | 15.50 | 23.58 |
| Block 10 | No | 10.00 | 75.00 | 75.00 | 10.00 | 10.41 | 17.84 |
| Block 10 | Yes | 20.00 | 75.00 | 75.00 | 20.00 | 20.35 | 28.25 |
| Block 10 | Yes | 15.00 | 75.00 | 75.00 | 15.00 | 15.32 | 23.27 |
| Block 10 | Yes | 10.00 | 75.00 | 75.00 | 10.00 | 10.29 | 17.59 |
| Block 9 | No | 20.00 | 75.00 | 75.00 | 20.00 | 20.53 | 28.55 |
| Block 9 | No | 15.00 | 75.00 | 75.00 | 15.00 | 15.50 | 23.44 |
| Block 9 | No | 10.00 | 75.00 | 75.00 | 10.00 | 10.39 | 17.56 |
| Block 9 | Yes | 20.00 | 75.00 | 75.00 | 20.00 | 20.40 | 28.10 |
| Block 9 | Yes | 15.00 | 75.00 | 75.00 | 15.00 | 15.36 | 22.95 |
| Block 9 | Yes | 10.00 | 75.00 | 75.00 | 10.00 | 10.32 | 17.26 |
| Block 8 | No | 20.00 | 75.00 | 75.00 | 20.00 | 20.30 | 25.55 |
| Block 8 | No | 15.00 | 75.00 | 75.00 | 15.00 | 15.32 | 20.58 |
| Block 8 | No | 10.00 | 75.00 | 75.00 | 10.00 | 10.27 | 15.28 |
| Block 8 | Yes | 20.00 | 75.00 | 75.00 | 20.00 | 20.33 | 25.20 |
| Block 8 | Yes | 15.00 | 75.00 | 75.00 | 15.00 | 15.29 | 20.23 |
| Block 8 | Yes | 10.00 | 75.00 | 75.00 | 10.00 | 10.26 | 15.16 |

## Step3600 collapse — `diag_collapse.py`

{
  "script": "/Users/navneetbavineni/STEGO/audit/scratch/diag_collapse.py",
  "checkpoint": "checkpoints/chaos/chaos_chaos_v2_ct_date_Sep30_15-52-07/checkpoint-epoch=10-step=3600-mIoU=test/cluster/mIoU=46.1792.ckpt",
  "checkpoint_global_step": 3600,
  "slice_count": 20,
  "patient_count": 4,
  "patients": [
    "5",
    "6",
    "8",
    "30"
  ],
  "requirement_ten_patients": "UNVERIFIED / impossible in current CT validation split: only four exist",
  "cluster_usage": [
    1003520,
    0
  ],
  "cluster_fractions": [
    1.0,
    0.0
  ],
  "cross_image_code_cosine": 0.9999127984046936,
  "cross_patient_code_cosine": 0.999912440776825,
  "raw_code_pixel_std": {
    "global_per_channel_mean": 9.118496894836426,
    "global_per_channel_min": 3.4154646396636963,
    "global_per_channel_max": 13.025422096252441,
    "within_image_per_channel_mean": 9.010177612304688
  },
  "unit_code_pixel_std": {
    "global_per_channel_mean": 0.0011053604539483786,
    "global_per_channel_min": 0.0009224582463502884,
    "global_per_channel_max": 0.0013089915737509727,
    "within_image_per_channel_mean": 0.0009753894992172718
  },
  "code_norm_mean": 439.62506103515625,
  "finite": true,
  "elapsed_seconds": 5.68
}

Codes are measured before L2 normalization and after per-pixel L2 normalization. Reported std is the population std over pixels, averaged over channels (also within-image mean and min/max per-channel values in JSON). Cross-image cosine averages all patch pairs between distinct images; cross-patient restricts to distinct patients. The script imports the real extractor and probe, strictly loads the saved net/probe state, and performs actual CPU forwards. It does not use the previously diagnosed four-slice shortcut.

**Coverage requirement not met:** ten CT validation patients do not exist. All four available patients are covered. Changing the split or relabelling training patients as validation would invalidate this diagnosis; wider patient generalization remains UNVERIFIED.


## Image paths — `diag_ceiling.py`



### Final (12), train mean subtraction: No

- [pca_b12_c0_CT_5_0046.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b12_c0_CT_5_0046.png)
- [pca_b12_c0_CT_6_0043.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b12_c0_CT_6_0043.png)
- [pca_b12_c0_CT_8_0058.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b12_c0_CT_8_0058.png)
- [pca_b12_c0_CT_30_0134.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b12_c0_CT_30_0134.png)

### Final (12), train mean subtraction: Yes

- [pca_b12_c1_CT_5_0046.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b12_c1_CT_5_0046.png)
- [pca_b12_c1_CT_6_0043.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b12_c1_CT_6_0043.png)
- [pca_b12_c1_CT_8_0058.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b12_c1_CT_8_0058.png)
- [pca_b12_c1_CT_30_0134.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b12_c1_CT_30_0134.png)

### Block 11, train mean subtraction: No

- [pca_b11_c0_CT_5_0046.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b11_c0_CT_5_0046.png)
- [pca_b11_c0_CT_6_0043.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b11_c0_CT_6_0043.png)
- [pca_b11_c0_CT_8_0058.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b11_c0_CT_8_0058.png)
- [pca_b11_c0_CT_30_0134.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b11_c0_CT_30_0134.png)

### Block 11, train mean subtraction: Yes

- [pca_b11_c1_CT_5_0046.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b11_c1_CT_5_0046.png)
- [pca_b11_c1_CT_6_0043.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b11_c1_CT_6_0043.png)
- [pca_b11_c1_CT_8_0058.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b11_c1_CT_8_0058.png)
- [pca_b11_c1_CT_30_0134.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b11_c1_CT_30_0134.png)

### Block 10, train mean subtraction: No

- [pca_b10_c0_CT_5_0046.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b10_c0_CT_5_0046.png)
- [pca_b10_c0_CT_6_0043.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b10_c0_CT_6_0043.png)
- [pca_b10_c0_CT_8_0058.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b10_c0_CT_8_0058.png)
- [pca_b10_c0_CT_30_0134.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b10_c0_CT_30_0134.png)

### Block 10, train mean subtraction: Yes

- [pca_b10_c1_CT_5_0046.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b10_c1_CT_5_0046.png)
- [pca_b10_c1_CT_6_0043.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b10_c1_CT_6_0043.png)
- [pca_b10_c1_CT_8_0058.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b10_c1_CT_8_0058.png)
- [pca_b10_c1_CT_30_0134.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b10_c1_CT_30_0134.png)

### Block 9, train mean subtraction: No

- [pca_b9_c0_CT_5_0046.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b9_c0_CT_5_0046.png)
- [pca_b9_c0_CT_6_0043.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b9_c0_CT_6_0043.png)
- [pca_b9_c0_CT_8_0058.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b9_c0_CT_8_0058.png)
- [pca_b9_c0_CT_30_0134.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b9_c0_CT_30_0134.png)

### Block 9, train mean subtraction: Yes

- [pca_b9_c1_CT_5_0046.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b9_c1_CT_5_0046.png)
- [pca_b9_c1_CT_6_0043.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b9_c1_CT_6_0043.png)
- [pca_b9_c1_CT_8_0058.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b9_c1_CT_8_0058.png)
- [pca_b9_c1_CT_30_0134.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b9_c1_CT_30_0134.png)

### Block 8, train mean subtraction: No

- [pca_b8_c0_CT_5_0046.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b8_c0_CT_5_0046.png)
- [pca_b8_c0_CT_6_0043.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b8_c0_CT_6_0043.png)
- [pca_b8_c0_CT_8_0058.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b8_c0_CT_8_0058.png)
- [pca_b8_c0_CT_30_0134.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b8_c0_CT_30_0134.png)

### Block 8, train mean subtraction: Yes

- [pca_b8_c1_CT_5_0046.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b8_c1_CT_5_0046.png)
- [pca_b8_c1_CT_6_0043.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b8_c1_CT_6_0043.png)
- [pca_b8_c1_CT_8_0058.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b8_c1_CT_8_0058.png)
- [pca_b8_c1_CT_30_0134.png](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_b8_c1_CT_30_0134.png)


[All PCA/overlay panels](/Users/navneetbavineni/STEGO/audit/scratch/diag/pca_overview.png) — montage produced by `diag_report.py`; full-size panels above.


## Verification limits and next decisions

Completed producer runtimes range2.29–21.70s, all below570s (each producer JSON records elapsed time). Seed/sample are fixed, with no validation-based fitting or threshold selection; source score comparisons are descriptive. Small between-layer differences are not established as statistically meaningful. No extra seeds, patient confidence intervals, actual boundary metric, source-specific kNN cache, full-val scoring, full-resolution decoder, training-time dropout calibration or retraining was run. The55-edge kNN subset is a material limitation; do not treat its millions of correlated patch pairs as millions of independent observations.

Recommendations only: first establish non-collapse with a train-only calibrated negative shift and unchanged restored-mean helper; log code-direction variability and cluster usage. Treat the10% candidates without mean subtraction as controlled starting hypotheses. Test actual training/dropout distributions before a full run. The fixed K2 probe partitions broad appearance groups, while K8 oracle scoring exposes recoverable liver information; investigate granularity without using validation labels to fit or select the unsupervised model. Do not promise precise liver contours from this14×14 grid or infer that a linear diagnostic is an unsupervised result. Hyperparameters, source choice and whether a candidate produces coherent anatomy after optimization all remain UNVERIFIED until separately authorized experiments.
