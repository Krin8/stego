# Cross-patient positive neighbours: CHAOS CT diagnostic

## Verdict

val: attracting liver-anchor target share changes 16.297% → 15.943% (-0.354 pp); attracted liver/liver fraction 1.424% → 1.395% (-0.029 pp). Baseline-normalized LL enrichment changes 1.330953× → 1.332077× (+0.001124×). These are diagnostic-label measurements, not selection criteria.

train: attracting liver-anchor target share changes 16.508% → 15.904% (-0.604 pp); attracted liver/liver fraction 1.638% → 1.577% (-0.061 pp). Baseline-normalized LL enrichment changes 1.328639× → 1.328982× (+0.000343×). These are diagnostic-label measurements, not selection criteria.

Cross-patient positive-shift changes are val: 0.474392 → 0.461848 (-0.012543); train: 0.451209 → 0.447286 (-0.003923). Positive attraction budgets remain approximately 75%; random negatives approximately 15%. The saved C configuration is unchanged.

Both cohorts show a small decrease in liver-target share under cross-patient exclusion. The tiny increase in baseline-normalized LL enrichment does not translate into a larger absolute LL fraction or more liver targets. There is no improvement in liver specificity demonstrated by this comparison. This is descriptive evidence, not a statistical-significance or training-benefit claim.

Positive-shift changes are limited in absolute affinity units (about 0.0125 validation and 0.0039 train), rather than a large change of scale. Cross-patient exclusion alone does not establish organ-specific correspondences. Training benefits, statistical/generalization uncertainty and anatomical registration across patients remain UNVERIFIED.

## Protocol, scope and labels

No project-source edits, neural/supervised training, installs, or dataset/checkpoint/pipeline-cache writes. Only new scratch diagnostics under `audit/scratch/crosspatient/` and this report are written. Every script has a 570-second timeout and CPU/offline execution. `neighbours.py` calls the actual project extractor and AST-imports the exact `precompute_knns.py:get_feats` pooling and ranking statements; only cross-patient exclusion is added for the alternate diagnostic set. It does NOT call the cache writer. Diagnostic `.npy` neighbour arrays are scratch results, not replacement pipeline caches.

`calibrate.py` uses actual project loss forward/helper and PAIRS coordinate sampling. Positive cross-patient comparisons use the same anchors and coordinates as unrestricted positives. C’s heuristic is reused: intra/positive 25th-percentile affinity, random cross-image negative 85th percentile. Thresholds are exact quantiles of a deterministic, label-free Monte Carlo subsample (512 uniformly selected entries per 121×121 affinity matrix; first negative roll per batch). Calibration and purity use different coordinate seeds. Observed attraction rates are measured on all sampled pairs in the independent purity pass. The heuristic is not prescribed by Paper A. Threshold changes are diagnostic values only; saved model/configuration values remain unchanged.

`purity.py` AST-imports the PAIRS `relations`, `spy_sample`, `hard_labels`, `capture`, and `label_summary` functions unchanged, and uses the real `ContrastiveCorrelationLoss.forward/helper`. Patch labels remain majority occupancy of each 16×16 validation/train-mask block, with ties assigned to background, nearest-sampled at actual feature coordinates. Pointwise restored-global-mean centering is performed by the project helper, per actual batch. Background includes all non-liver tissue and air.

**Label accounting:** all neighbour construction/ranking, sample selection, similarity/position statistics, thresholds and calibration attraction rates use NO labels. Validation purity uses the 472 validation masks. The train check uses TRAIN masks ONLY for the explicitly authorized Task 3 diagnostic, after sample/neighbours/calibration have been fixed. No masks affect neighbour selection, anchor sampling, shifts, model parameters or any unsupervised choice. Same-label, liver/liver, mixed, conditional liver-target rates and enrichment/deltas use the corresponding diagnostic cohort’s masks. A runtime guard blocks all annotations during calibration; its file hash is verified unchanged after scoring.

Train sampling: uniform without replacement, seed 20260930, 200 anchors covering {'14': 12, '10': 11, '25': 13, '22': 11, '23': 22, '21': 22, '16': 8, '26': 12, '19': 14, '18': 8, '1': 5, '28': 10, '29': 21, '27': 16, '24': 9, '2': 6}. This covers all 16 train patients, exceeding the required 12, without label-based selection. Neighbour candidates come from all 1869 train images. Scoring those edges requires 1841 train masks: 200 anchor masks plus 1641 distinct neighbour-only masks. This is still a 200-anchor check, not an all-train-anchor purity estimate. Producer: `neighbours.py` and `purity.py --split train`. Validation uses 472 masks, from four patients.

**k convention:** source cache k=30 includes self, leaving 29 non-self entries. Task 1 and primary validation purity use the requested 30 non-self neighbours, computed by the same method. Train primary uses exactly the stored cache’s 29 non-self neighbours, versus the top 29 cross-patient neighbours from the full train candidate pool. Validation prefix-29 sensitivity is reported below; its thresholds remain the independently calibrated 30-neighbour values. Top-seven values are supplementary, since the actual training loader previously sampled only seven non-self ranks.

## Task 1: label-free neighbour comparison

Producer: `neighbours.py` → `neighbours.json`. Similarity is cosine of spatial-mean dense features after L2, exactly the cache method. A normalized position is native physical-position rank divided by stack length minus one. Its cross-patient difference is descriptive, not anatomical registration. Native slice offsets are meaningful only within the same patient and series.

| Cohort / neighbours | Non-self k | Anchors | Same-patient % | Mean image cosine | Median absolute normalized-position difference | P95 normalized-position difference | Same-series native offset median / P95 |
|---|---:|---:|---:|---:|---:|---:|---|
| val / current | 30 | 472 | 83.545 | 0.987482786 | 0.051020 | 0.159574 | 7.0 / 18.0 |
| val / cross | 30 | 472 | 0.000 | 0.973060608 | 0.067157 | 0.276972 | N/A (different patients) |
| train / current | 29 | 200 | 79.638 | 0.988919020 | 0.049724 | 0.141304 | 6.0 / 16.0 |
| train / cross | 29 | 200 | 0.000 | 0.979582787 | 0.062341 | 0.214461 | N/A (different patients) |

Position-difference distributions, from `neighbours.py` (no labels):

| Cohort / variant | Difference definition | P5 | P25 | P50 | P75 | P95 | Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| val / current | absolute normalized stack position | 0.010204 | 0.025510 | 0.051020 | 0.085106 | 0.159574 | 0.063914 |
| val / current | native slice ranks, same series only | 1.000000 | 4.000000 | 7.000000 | 11.000000 | 18.000000 | 7.894167 |
| val / cross | absolute normalized stack position | 0.005981 | 0.030718 | 0.067157 | 0.125706 | 0.276972 | 0.092619 |
| train / current | absolute normalized stack position | 0.007908 | 0.025000 | 0.049724 | 0.080000 | 0.141304 | 0.058071 |
| train / current | native slice ranks, same series only | 1.000000 | 3.000000 | 6.000000 | 10.000000 | 16.000000 | 7.291622 |
| train / cross | absolute normalized stack position | 0.005420 | 0.027108 | 0.062341 | 0.117864 | 0.214461 | 0.080542 |

Full position and image-similarity distributions (quantiles 0/5/25/50/75/95/100), including same-series offsets, normalized positions for all edges and cross-patient edges, and top-seven/prefix-29 sets, are retained in `neighbours.json`. No labels are used for any of those distributions.

### Cache reproduction / numerical checks

The initial batch-16 extractor run stopped on an exact-rank assertion (`neighbours.log`); no cache was written. The successful run uses the cache writer’s batch size 256 (`neighbours_batch256.log`). All 29-neighbour SETS reproduce exactly in both splits; tiny within-set ordering differences remain. The train comparison uses stored order directly. Producer: `rank_check.py` → `rank_check.json`, no labels.

| Split | Rows with reordered ranks | Changed rank entries | Maximum cosine gap at swapped ranks | All neighbour sets identical? | Max dense-feature difference, batch256 vs prior batch16 |
|---|---:|---:|---:|---|---:|
| val | 1 | 2 | 1.192092896e-07 | True | 0.002285004 |
| train | 5 | 10 | 1.788139343e-07 | True | 0.001745224 |

The numerical differences do not change neighbour membership. Their exact backend/kernel cause is UNVERIFIED; do not infer a stale cache from the rank swaps. Image-neighbour construction uses live batch-256 features; loss-affinity diagnostics reuse the PAIRS/BASELINE batch-16 dense features from the same frozen weights, layer, normalization and preprocessing. Batch-level floating-point differences are recorded rather than claimed zero.

## Task 2 and 3: shifts and purity

Producer: `calibrate.py --split val|train` → `calibration_SPLIT.json`; NO labels. Both positive shifts are recalibrated within each cohort, rather than retaining the old C shift only for one variant. Intra and negative thresholds are shared between variants.

| Cohort | Intra shift (75% target) | Current positive shift (75%) | Cross-patient positive shift (75%) | Cross − current | Common negative shift (15%) | Retained affinity samples per role |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.479792714 | 0.474391788 | 0.461848468 | -0.012543321 | 0.747171819 | 7,249,920 |
| train | 0.456120253 | 0.451208979 | 0.447286189 | -0.003922790 | 0.749162257 | 2,969,600 |

Saved C reference shifts (unchanged): {'pos_intra_shift': 0.4584466218948364, 'pos_inter_shift': 0.45247218012809753, 'neg_inter_shift': 0.7459585666656494}. These were calibrated previously on a different training sample/rank distribution; the present recalibrations are not proposed source/config edits.

Producer: `purity.py --split val|train` → `purity_SPLIT.json`. All percentages below use that cohort’s diagnostic masks, except attraction rate itself (affinity-only). Parentheses give the unfiltered/random spatial-pair baseline for the identical image-pair type. Intra/negative rows are shared between variants by construction, not independently resampled.

| Cohort / pair type | Attraction % | Same-label % (random) | Liver/liver % (random) | Mixed % (random) | Liver-target share given liver anchor, attracted % (random) | Liver→background pairs attracting % | LL enrichment factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| val / intra (shared) | 75.142 | 88.341 (85.253) | 1.562 (1.176) | 11.659 (14.747) | 18.036 (13.752) | 72.353 | 1.328865 |
| val / current positive | 75.006 | 88.034 (84.993) | 1.424 (1.070) | 11.966 (15.007) | 16.297 (12.511) | 73.320 | 1.330953 |
| val / cross-patient positive | 74.940 | 87.605 (84.546) | 1.395 (1.047) | 12.395 (15.454) | 15.943 (12.245) | 73.444 | 1.332077 |
| val / negative (shared) | 15.028 | 95.113 (84.316) | 3.465 (0.732) | 4.887 (15.684) | 55.549 (8.562) | 5.331 | 4.734177 |
| train / intra (shared) | 75.265 | 86.359 (83.437) | 1.797 (1.356) | 13.641 (16.563) | 18.225 (14.071) | 73.268 | 1.324857 |
| train / current positive | 75.052 | 86.196 (83.336) | 1.638 (1.233) | 13.804 (16.664) | 16.508 (12.794) | 73.991 | 1.328639 |
| train / cross-patient positive | 75.046 | 86.211 (83.246) | 1.577 (1.187) | 13.789 (16.754) | 15.904 (12.315) | 74.063 | 1.328982 |
| train / negative (shared) | 15.164 | 94.560 (82.581) | 3.943 (0.924) | 5.440 (17.419) | 56.213 (9.582) | 5.342 | 4.267675 |

LL enrichment factor = P(liver/liver | attracted) / P(liver/liver | unfiltered). It distinguishes enrichment relative to a variant’s baseline from its absolute liver/liver fraction. A factor near 1.33 at 75% attraction can arise simply because nearly all liver/liver pairs attract; it does not mean most attracting targets of liver anchors are liver.

### Cross-patient minus current: effect sizes

Producer: `report.py` aggregates the preceding `purity.py` counts; corresponding validation/train diagnostic labels are used. pp = percentage points.

| Cohort | Attracting liver-target share change, pp | Attracting LL fraction change, pp | LL enrichment current → cross | Enrichment-factor change | Liver→background attraction change, pp |
|---|---:|---:|---|---:|---:|
| val | -0.3541 | -0.0290 | 1.330953 → 1.332077 | +0.001124 | +0.1237 |
| train | -0.6036 | -0.0610 | 1.328639 → 1.328982 | +0.000343 | +0.0722 |

### Neighbour-count sensitivity

Producer: `purity.py` prefix counts; same calibrated full-neighbour thresholds and cohort masks. This isolates adding ranks, without choosing new thresholds or neighbours using labels.

| Cohort | Prefix non-self k | Variant | Attraction % | Attracted liver-target share % | Attracted LL % | LL enrichment |
|---|---:|---|---:|---:|---:|---:|
| val | 7 | current | 75.500 | 16.941 | 1.465 | 1.322413 |
| val | 7 | cross | 75.007 | 15.884 | 1.377 | 1.331235 |
| val | 29 | current | 75.020 | 16.327 | 1.426 | 1.330719 |
| val | 29 | cross | 74.940 | 15.922 | 1.393 | 1.332110 |
| val | 30 | current | 75.006 | 16.297 | 1.424 | 1.330953 |
| val | 30 | cross | 74.940 | 15.943 | 1.395 | 1.332077 |
| train | 7 | current | 75.591 | 16.858 | 1.653 | 1.318868 |
| train | 7 | cross | 75.403 | 15.972 | 1.569 | 1.322828 |
| train | 29 | current | 75.052 | 16.508 | 1.638 | 1.328639 |
| train | 29 | cross | 75.046 | 15.904 | 1.577 | 1.328982 |

### PAIRS patient/slice strata

Producer: `purity.py`, using unchanged PAIRS strata. Percentages use the corresponding diagnostic masks. Strata overlap; do not sum them. Native adjacency/within-three are defined only in the same patient/series and therefore have no cross-patient positive entries.

| Cohort / variant / stratum | Image edges | Patch pairs | Attraction % | Same-label attracted % (random) | LL attracted % (random) | Mixed attracted % (random) | Attracted liver-target share % |
|---|---:|---:|---:|---:|---:|---:|---:|
| val / current / same_patient | 11830 | 173203030 | 75.102 | 88.328 (85.365) | 1.365 (1.027) | 11.672 (14.635) | 16.051 |
| val / current / cross_patient | 2330 | 34113530 | 74.520 | 86.530 (83.107) | 1.721 (1.284) | 13.470 (16.893) | 17.376 |
| val / current / same_slice | 0 | 0 | — | — (—) | — (—) | — (—) | — |
| val / current / adjacent_slice | 936 | 13703976 | 75.612 | 88.145 (85.107) | 1.461 (1.107) | 11.855 (14.893) | 17.083 |
| val / current / same_or_adjacent | 936 | 13703976 | 75.612 | 88.145 (85.107) | 1.461 (1.107) | 11.855 (14.893) | 17.083 |
| val / current / within_3_slices | 2784 | 40760544 | 75.498 | 87.962 (84.903) | 1.491 (1.127) | 12.038 (15.097) | 16.977 |
| val / current / same_patient_beyond_3 | 9046 | 132442486 | 74.980 | 88.441 (85.507) | 1.327 (0.997) | 11.559 (14.493) | 15.752 |
| val / cross / same_patient | 0 | 0 | — | — (—) | — (—) | — (—) | — |
| val / cross / cross_patient | 14160 | 207316560 | 74.940 | 87.605 (84.546) | 1.395 (1.047) | 12.395 (15.454) | 15.943 |
| val / cross / same_slice | 0 | 0 | — | — (—) | — (—) | — (—) | — |
| val / cross / adjacent_slice | 0 | 0 | — | — (—) | — (—) | — (—) | — |
| val / cross / same_or_adjacent | 0 | 0 | — | — (—) | — (—) | — (—) | — |
| val / cross / within_3_slices | 0 | 0 | — | — (—) | — (—) | — (—) | — |
| val / cross / same_patient_beyond_3 | 0 | 0 | — | — (—) | — (—) | — (—) | — |
| train / current / same_patient | 4619 | 67626779 | 75.006 | 86.176 (83.479) | 1.615 (1.216) | 13.824 (16.521) | 16.353 |
| train / current / cross_patient | 1181 | 17291021 | 75.232 | 86.274 (82.776) | 1.727 (1.301) | 13.726 (17.224) | 17.100 |
| train / current / same_slice | 0 | 0 | — | — (—) | — (—) | — (—) | — |
| train / current / adjacent_slice | 398 | 5827118 | 75.682 | 86.160 (83.319) | 1.678 (1.275) | 13.840 (16.681) | 17.229 |
| train / current / same_or_adjacent | 398 | 5827118 | 75.682 | 86.160 (83.319) | 1.678 (1.275) | 13.840 (16.681) | 17.229 |
| train / current / within_3_slices | 1190 | 17422790 | 75.549 | 86.155 (83.332) | 1.669 (1.265) | 13.845 (16.668) | 16.869 |
| train / current / same_patient_beyond_3 | 3429 | 50203989 | 74.818 | 86.183 (83.531) | 1.597 (1.198) | 13.817 (16.469) | 16.171 |
| train / cross / same_patient | 0 | 0 | — | — (—) | — (—) | — (—) | — |
| train / cross / cross_patient | 5800 | 84917800 | 75.046 | 86.211 (83.246) | 1.577 (1.187) | 13.789 (16.754) | 15.904 |
| train / cross / same_slice | 0 | 0 | — | — (—) | — (—) | — (—) | — |
| train / cross / adjacent_slice | 0 | 0 | — | — (—) | — (—) | — (—) | — |
| train / cross / same_or_adjacent | 0 | 0 | — | — (—) | — (—) | — (—) | — |
| train / cross / within_3_slices | 0 | 0 | — | — (—) | — (—) | — (—) | — |
| train / cross / same_patient_beyond_3 | 0 | 0 | — | — (—) | — (—) | — (—) | — |

## Limits / UNVERIFIED

- No training intervention was performed. Whether changing positives improves trained segmentation is UNVERIFIED; purity diagnostics do not establish a causal gain.
- Cross-patient slice-position offsets are not anatomical registration. Physical slice numbers/origins differ; anatomical correspondence at matched normalized positions is UNVERIFIED.
- Calibration and purity use deterministic Monte Carlo patch sampling, not exhaustive continuous coordinates. Repeat-sampling uncertainty and patient-level confidence intervals are UNVERIFIED. Patch pairs are highly dependent; their large count is not an independent sample size.
- Validation has four patients; the train diagnostic has 200 uniformly sampled anchors from 16 patients. Generalization beyond these cohorts is UNVERIFIED. Target masks outside the anchor sample are used only because neighbours come from the full train pool.
- PAIRS hard patch-majority labels approximate bilinearly mixed features at boundaries. Alternative boundary labeling is UNVERIFIED.
- The cache contains image neighbours, not patch matches. No new patch-matching rule is introduced. No shift or neighbour choice uses labels.
- Exact numerical origin of within-set cache rank swaps is UNVERIFIED; all stored/recomputed 29-neighbour memberships are identical.

## Producers and execution

Commands (already run for this diagnostic; none train models or write pipeline caches):

```text
python3 -B audit/scratch/crosspatient/neighbours.py
python3 -B audit/scratch/crosspatient/rank_check.py
python3 -B audit/scratch/crosspatient/calibrate.py --split val
python3 -B audit/scratch/crosspatient/calibrate.py --split train
python3 -B audit/scratch/crosspatient/purity.py --split val
python3 -B audit/scratch/crosspatient/purity.py --split train
python3 -B audit/scratch/crosspatient/report.py
```

Neighbour construction runtime 69.923s (both cohorts); rank check 4.022s. Calibration runtimes: validation 91.730s, train 34.727s. Purity runtimes: validation 109.728s, train 53.099s. Every script has a 570-second hard timeout. Counts/runtimes are generated by the corresponding named scripts; no labels are used for runtime measurements.

Source commit `e3c3ab0e396c194a7314d64546c81e9ba4973b72`; tracked source changes: none. Scratch data and report only.

Final checks: `verify.py` → `verification.json`; {"script": "/Users/navneetbavineni/STEGO/audit/scratch/crosspatient/verify.py", "status": "PASS", "cohorts": {"val": {"anchors": 472, "patients": 4, "mask_count": 472, "calibration_hash_unchanged": true, "current_positive_attraction": 0.7500626578021553, "cross_positive_attraction": 0.7494001395739925, "negative_attraction": 0.1502829869107574, "prefix29_cross_minus_current_liver_target_pp": -0.40562891318696725, "calibration_seconds": 91.72963262499252, "purity_seconds": 109.7280095419992}, "train": {"anchors": 200, "patients": 16, "mask_count": 1841, "calibration_hash_unchanged": true, "current_positive_attraction": 0.7505219400408395, "cross_positive_attraction": 0.7504644727018364, "negative_attraction": 0.15164386192005802, "prefix29_cross_minus_current_liver_target_pp": null, "calibration_seconds": 34.72707779200573, "purity_seconds": 53.099034915998345}}, "no_tracked_source_changes": true, "elapsed_seconds": 2.8052427919901675}
