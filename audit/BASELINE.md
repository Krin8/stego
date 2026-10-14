# CHAOS CT identical-sample baseline and held-out-patient mapping

## Verdict

The trained head does **not** beat the frozen-feature baseline here. Across all requested K-means fits, global optimal liver IoU spans 47.666–66.662%, versus at most 44.400% for the five heads (C full batch600). Same held-out slices and mapping definitions; fitting labels are never used.

LOPO head mean IoU spans 30.074–44.180%; K-means spans 47.901–66.617%. The per-patient/pooled tables quantify transfer without fitting a mapping on its held-out patient.

Mapping-transfer loss measured with the SAME pixel pooling: seed0 43.749 → 43.749% IoU; seed1 32.853 → 32.853% IoU; seed2 31.948 → 31.025% IoU; best600 44.400 → 44.400% IoU; final 29.807 → 29.807% IoU. Thus the head deficit is not explained mainly by choosing its mapping on the scored patient. This does not eliminate checkpoint-selection optimism.

Label-free proxy correlations with oracle liver IoU: correspondence loss rho=0.772, extremum at batch 4600; cross-image cosine rho=-0.173, extremum at batch 500; cluster entropy rho=-0.677, extremum at batch 5000; ARI vs frozen K8 rho=0.302, extremum at batch 700. This is one trajectory and an evaluation-label-based analysis; it does not establish a reliable selection rule.

UNVERIFIED: the unsaved DIAG linear probe on all 472 slices; generalization beyond four validation patients; label-free checkpoint selection on independent runs.

## Protocol and label use

All 1,869 train slices (366,324 patches; 16 patients), and all 472 validation slices (four disjoint patients). Producer: `audit/scratch/baseline/extract.py`, `extract.json`, `samples.json`. Fresh features matched the FORENSICS validation features with maximum absolute error 0.0; every validation mask and ordered sample ID matched exactly. The actual project extractor, preprocessing, head and probe are imported through `common.py`. Extraction uses `ImagesOnlyDataset`; a runtime guard forbids opening training annotations.

Frozen features: final-block LayerNorm patch tokens, CLS excluded, 14×14 grid, 768 channels. No-mean variant applies per-patch L2 directly; centered variant subtracts the mean of ALL training patch tokens and then applies L2. K-means is fitted only on training images, with K=8/16 and seeds 0/1/2 declared in advance. Lloyd, three initializations per seed, maximum 100 iterations, tolerance 1e-4; no validation labels influence model fitting, mean, K, seed or parameters. K-means assignments use nearest-neighbor upsampling to 224×224, matching DIAG/FORENSICS. Trained heads use the actual bilinear-upsample-code-then-probe path. Thus sample, scoring and mapping rules are identical; the native inference/upsampling operations remain method-specific, not artificially equated.

**Label accounting for every table:** global majority and global optimal use all 472 validation masks for BOTH mapping and scoring (optimistic diagnostics). LOPO selects the optimal subset with three validation patients’ labels, then scores on the fourth patient’s labels; the held-out labels never choose its mapping. LOPO is supervised calibration of a cluster mapping, not label-free evaluation. Patient mean/min weight each patient equally; pooled LOPO aggregates TP/FP/FN across all held-out pixels for direct comparison to global scores. Slice counts, feature dimensions, fit inertia/iterations, runtimes and label-free proxies use no labels. Confusion counts, IoU, Dice and correlations with oracle IoU use evaluation labels only.

`common.py` AST-imports FORENSICS `scores` and `mapping_report`, and its actual confusion-matrix expression. Only K-dependent constants are generalized. Both K=8 and K=16 use exhaustive enumeration of all 2^K subsets, including the empty subset. K=8 outputs are asserted identical to the unmodified FORENSICS function. Strict-majority means liver pixels > background pixels, with ties assigned to background. Trained-head prediction artifacts are reused from `forensics/task1.py`; `heads.py` verifies that all five global confusion matrices and mappings exactly reproduce FORENSICS. No CRF is applied.

## Main comparison: all validation pixels

Every score below is percent, **IoU / Dice**. Producers: frozen rows `audit/scratch/baseline/kmeans.py --k K --center C --seed S` → `kK_cC_sS.json`; head rows `audit/scratch/baseline/heads.py` → `heads.json`, using `forensics/task1.py` predictions. Global mapping/scoring labels: all four validation patients.

| Method | K | Seed | Majority IoU / Dice | Optimal IoU / Dice | LOPO mean IoU / Dice |
|---|---:|---:|---:|---:|---:|
| Frozen K-means | 8 | 0 | 64.135 / 78.149 | 64.135 / 78.149 | 64.020 / 78.045 |
| Frozen K-means | 8 | 1 | 64.135 / 78.149 | 64.135 / 78.149 | 64.020 / 78.045 |
| Frozen K-means | 8 | 2 | 64.135 / 78.149 | 64.135 / 78.149 | 64.020 / 78.045 |
| Frozen K-means | 16 | 0 | 59.466 / 74.581 | 59.466 / 74.581 | 55.815 / 71.630 |
| Frozen K-means | 16 | 1 | 59.440 / 74.561 | 59.440 / 74.561 | 55.694 / 71.530 |
| Frozen K-means | 16 | 2 | 59.429 / 74.552 | 59.429 / 74.552 | 55.675 / 71.515 |
| Frozen K-means + train mean subtraction | 8 | 0 | 66.662 / 79.997 | 66.662 / 79.997 | 66.617 / 79.963 |
| Frozen K-means + train mean subtraction | 8 | 1 | 48.744 / 65.541 | 48.744 / 65.541 | 48.990 / 65.678 |
| Frozen K-means + train mean subtraction | 8 | 2 | 47.666 / 64.560 | 47.666 / 64.560 | 47.901 / 64.688 |
| Frozen K-means + train mean subtraction | 16 | 0 | 60.221 / 75.172 | 60.221 / 75.172 | 60.067 / 75.008 |
| Frozen K-means + train mean subtraction | 16 | 1 | 60.169 / 75.132 | 60.169 / 75.132 | 60.015 / 74.970 |
| Frozen K-means + train mean subtraction | 16 | 2 | 59.706 / 74.770 | 59.706 / 74.770 | 59.562 / 74.611 |
| C seed0 batch500 | 8 | 0 | 43.749 / 60.869 | 43.749 / 60.869 | 43.842 / 60.620 |
| C seed1 batch500 | 8 | 1 | 0.000 / 0.000 | 32.853 / 49.457 | 32.549 / 48.837 |
| C seed2 batch500 | 8 | 2 | 0.000 / 0.000 | 31.948 / 48.425 | 30.722 / 46.841 |
| C full batch600 | 8 | 0 | 44.400 / 61.496 | 44.400 / 61.496 | 44.180 / 61.220 |
| C full batch5000 | 8 | 0 | 0.000 / 0.000 | 29.807 / 45.925 | 30.074 / 46.135 |
| DIAG supervised linear probe | — | 20260930 | UNVERIFIED | UNVERIFIED | UNVERIFIED |

The original DIAG linear-probe coefficients were not saved: `audit/scratch/diag_ceiling.py` records `model_saved=False` and contains no model serialization. A targeted file-name search under `audit/scratch/diag` and `audit/run_records` found no probe/coefficient, pickle, joblib or NPZ artifact. Its 100-slice predictions cannot establish a 472-slice score or LOPO results. A recoverable original fitted estimator (and its train mean/normalization metadata) would settle this; supervised refitting is outside this request. No old 100-slice score is mixed into this table.

## Leave-one-patient-out mapping

Producer: `extract.py` for IDs/counts (no labels); `kmeans.py` and `heads.py` for every per-patient and aggregate score. LOPO mapping labels: the other three patients only; scoring labels: the indicated held-out patient.

| Patient | Validation slices |
|---|---:|
| 5 | 95 |
| 6 | 82 |
| 8 | 98 |
| 30 | 197 |

| Method / K / seed | Patient 5 IoU / Dice | Patient 6 IoU / Dice | Patient 8 IoU / Dice | Patient 30 IoU / Dice | Mean IoU / Dice | Minimum IoU / Dice | Pooled IoU / Dice |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen K-means / 8 / 0 | 66.505 / 79.884 | 64.289 / 78.263 | 60.903 / 75.701 | 64.383 / 78.333 | 64.020 / 78.045 | 60.903 / 75.701 | 64.135 / 78.149 |
| Frozen K-means / 8 / 1 | 66.505 / 79.884 | 64.289 / 78.263 | 60.903 / 75.701 | 64.383 / 78.333 | 64.020 / 78.045 | 60.903 / 75.701 | 64.135 / 78.149 |
| Frozen K-means / 8 / 2 | 66.505 / 79.884 | 64.289 / 78.263 | 60.903 / 75.701 | 64.383 / 78.333 | 64.020 / 78.045 | 60.903 / 75.701 | 64.135 / 78.149 |
| Frozen K-means / 16 / 0 | 55.930 / 71.737 | 58.278 / 73.640 | 54.727 / 70.740 | 54.323 / 70.402 | 55.815 / 71.630 | 54.323 / 70.402 | 55.697 / 71.546 |
| Frozen K-means / 16 / 1 | 55.878 / 71.694 | 58.166 / 73.550 | 54.696 / 70.714 | 54.039 / 70.163 | 55.694 / 71.530 | 54.039 / 70.163 | 55.544 / 71.419 |
| Frozen K-means / 16 / 2 | 55.842 / 71.665 | 58.123 / 73.516 | 54.696 / 70.714 | 54.039 / 70.163 | 55.675 / 71.515 | 54.039 / 70.163 | 55.526 / 71.404 |
| Frozen K-means + train mean subtraction / 8 / 0 | 66.655 / 79.992 | 67.343 / 80.485 | 65.907 / 79.451 | 66.564 / 79.926 | 66.617 / 79.963 | 65.907 / 79.451 | 66.662 / 79.997 |
| Frozen K-means + train mean subtraction / 8 / 1 | 47.153 / 64.087 | 55.369 / 71.274 | 47.854 / 64.731 | 45.583 / 62.621 | 48.990 / 65.678 | 45.583 / 62.621 | 48.744 / 65.541 |
| Frozen K-means + train mean subtraction / 8 / 2 | 46.436 / 63.421 | 54.279 / 70.364 | 46.419 / 63.406 | 44.469 / 61.562 | 47.901 / 64.688 | 44.469 / 61.562 | 47.666 / 64.560 |
| Frozen K-means + train mean subtraction / 16 / 0 | 64.071 / 78.102 | 59.441 / 74.562 | 55.711 / 71.557 | 61.045 / 75.811 | 60.067 / 75.008 | 55.711 / 71.557 | 60.221 / 75.172 |
| Frozen K-means + train mean subtraction / 16 / 1 | 63.870 / 77.952 | 59.334 / 74.478 | 55.795 / 71.626 | 61.060 / 75.823 | 60.015 / 74.970 | 55.795 / 71.626 | 60.169 / 75.132 |
| Frozen K-means + train mean subtraction / 16 / 2 | 63.731 / 77.849 | 58.638 / 73.927 | 55.298 / 71.216 | 60.580 / 75.452 | 59.562 / 74.611 | 55.298 / 71.216 | 59.706 / 74.770 |
| C seed0 batch500 / 8 / 0 | 47.933 / 64.804 | 51.697 / 68.158 | 42.661 / 59.808 | 33.076 / 49.709 | 43.842 / 60.620 | 33.076 / 49.709 | 43.749 / 60.869 |
| C seed1 batch500 / 8 / 1 | 32.291 / 48.818 | 41.417 / 58.575 | 30.961 / 47.283 | 25.526 / 40.670 | 32.549 / 48.837 | 25.526 / 40.670 | 32.853 / 49.457 |
| C seed2 batch500 / 8 / 2 | 32.515 / 49.074 | 36.567 / 53.552 | 28.610 / 44.491 | 25.194 / 40.248 | 30.722 / 46.841 | 25.194 / 40.248 | 31.025 / 47.358 |
| C full batch600 / 8 / 0 | 43.513 / 60.640 | 49.469 / 66.193 | 42.050 / 59.205 | 41.686 / 58.842 | 44.180 / 61.220 | 41.686 / 58.842 | 44.400 / 61.496 |
| C full batch5000 / 8 / 0 | 30.039 / 46.201 | 35.545 / 52.447 | 28.550 / 44.419 | 26.161 / 41.472 | 30.074 / 46.135 | 26.161 / 41.472 | 29.807 / 45.925 |

Mapping subsets, including every global and held-out mapping, are recorded below. IDs belong only to that fitted model and are not shared across seeds. A LOPO score can be lower than its global optimum; macro vs pooled weighting can also change their ordering.

| ID | Global majority liver clusters | Global optimal liver clusters | Held-out 5 mapping | Held-out 6 mapping | Held-out 8 mapping | Held-out 30 mapping |
|---|---|---|---|---|---|---|
| k8_c0_s0 | [5] | [5] | [5] | [5] | [5] | [5] |
| k8_c0_s1 | [4] | [4] | [4] | [4] | [4] | [4] |
| k8_c0_s2 | [1] | [1] | [1] | [1] | [1] | [1] |
| k16_c0_s0 | [3] | [3] | [3, 4] | [3] | [3] | [3, 4] |
| k16_c0_s1 | [8] | [8] | [8, 13] | [8] | [8] | [8, 13] |
| k16_c0_s2 | [10] | [10] | [2, 10] | [10] | [10] | [2, 10] |
| k8_c1_s0 | [5] | [5] | [5] | [5] | [5] | [5] |
| k8_c1_s1 | [5] | [5] | [5] | [5] | [5] | [5] |
| k8_c1_s2 | [7] | [7] | [7] | [7] | [7] | [7] |
| k16_c1_s0 | [5] | [5] | [5] | [5] | [5] | [5] |
| k16_c1_s1 | [12] | [12] | [12] | [12] | [12] | [12] |
| k16_c1_s2 | [12] | [12] | [12] | [12] | [12] | [12] |
| seed0 | [2] | [2] | [2] | [2] | [2] | [2] |
| seed1 | [] | [4] | [4] | [4] | [4] | [4] |
| seed2 | [] | [0, 2] | [0, 2] | [0, 2] | [0, 2] | [0] |
| best600 | [2] | [2] | [2] | [2] | [2] | [2] |
| final | [] | [2, 5] | [2, 5] | [2, 5] | [2, 5] | [2, 5] |

## Label-free checkpoint proxies

Producers: `audit/scratch/baseline/proxies.py --start A --stop B` → five `proxies_A_B.json` chunks; this `report.py` computes Spearman correlations against the fresh per-checkpoint oracle scores from `forensics/task5.py`. All 50 saved checkpoints, batches 100–5000. Correspondence loss uses the SAME 16 train batches and kNN partners/seeds as FORENSICS Task 4, dropout off, and the actual correspondence helper with restored-mean centering. Frozen activations are reused only after exact visual-trunk parameter equality and live feature/code equality checks. No label files are opened by the proxy scripts. Cross-image cosine/entropy and ARI use all 472 validation images without masks. ARI reference is the predeclared all-train K=8, uncentered, seed-0 baseline.

Proxy selection directions were set before examining correlations: minimize objective/cosine; maximize entropy/ARI. These directions are diagnostic heuristics, not established semantic-quality criteria. All correlations and reported quality at the selected batch use validation labels through oracle IoU. The checkpoints share one training trajectory, so nominal Spearman p-values do not account for serial dependence and should not be interpreted as independent-test significance.

| Proxy | Spearman rho with oracle IoU | Nominal p | Direction | Selected batch | Selected value | Selected oracle IoU % | Within batches 500–600? |
|---|---:|---:|---|---:|---:|---:|---|
| correspondence loss | 0.772293 | 5.1e-11 | minimize | 4600 | -0.136094814 | 29.503 | No |
| cross-image cosine | -0.173301 | 0.229 | minimize | 500 | 0.046338171 | 43.758 | Yes |
| cluster entropy | -0.676927 | 6.77e-08 | maximize | 5000 | 2.024406558 | 29.807 | No |
| ARI vs frozen K8 | 0.302185 | 0.0329 | maximize | 700 | 0.226663478 | 39.495 | No |

Interpretation: lower correspondence loss favors a late, lower-quality checkpoint despite its strong positive correlation with IoU (the desired minimization direction is reversed). Higher entropy likewise favors poorer late checkpoints. Minimum code cosine selects batch 500, near the quality peak, but its rank correlation is weak. Maximum ARI selects batch 700, one saved checkpoint after batch 600: nearby, but past the peak. Neither the weak cosine association nor the modest ARI association establishes a transferable label-free selector.

| Batch | Correspondence loss | Code cosine | Cluster entropy | ARI vs frozen K8 | Oracle liver IoU % |
|---:|---:|---:|---:|---:|---:|
| 100 | -0.126379074 | 0.072296039 | 1.573105015 | 0.189715916 | 30.644 |
| 200 | -0.128407075 | 0.048639841 | 1.512371327 | 0.191686615 | 29.865 |
| 300 | -0.129809039 | 0.047012172 | 1.649042554 | 0.165923857 | 37.118 |
| 400 | -0.130607376 | 0.054796037 | 1.766565332 | 0.154710178 | 38.734 |
| 500 | -0.130791979 | 0.046338171 | 1.833061478 | 0.194238482 | 43.758 |
| 600 | -0.131566992 | 0.070932743 | 1.817874516 | 0.223315275 | 44.400 |
| 700 | -0.132032748 | 0.095667589 | 1.845440149 | 0.226663478 | 39.495 |
| 800 | -0.132502198 | 0.046988286 | 1.949290334 | 0.210406516 | 35.368 |
| 900 | -0.133169341 | 0.082024881 | 1.904375440 | 0.195716250 | 36.075 |
| 1000 | -0.133417252 | 0.122219657 | 1.887405455 | 0.190466206 | 35.606 |
| 1100 | -0.133987178 | 0.061054377 | 1.921114172 | 0.200257830 | 33.756 |
| 1200 | -0.133913284 | 0.059550458 | 1.942358086 | 0.196938058 | 34.220 |
| 1300 | -0.134313571 | 0.097159888 | 1.910019271 | 0.204367729 | 32.979 |
| 1400 | -0.134655415 | 0.091891925 | 1.911434714 | 0.201592372 | 32.634 |
| 1500 | -0.134743415 | 0.078035859 | 1.883626878 | 0.205848170 | 30.720 |
| 1600 | -0.135073500 | 0.091011519 | 1.892768977 | 0.200969529 | 31.497 |
| 1700 | -0.135091419 | 0.069184088 | 1.902721391 | 0.201849138 | 30.681 |
| 1800 | -0.135229536 | 0.087765852 | 1.891650791 | 0.202808467 | 30.998 |
| 1900 | -0.135371260 | 0.088221661 | 1.896332399 | 0.201323030 | 30.422 |
| 2000 | -0.135236283 | 0.075700154 | 1.898020268 | 0.204786841 | 30.194 |
| 2100 | -0.135354318 | 0.072443223 | 1.897209238 | 0.201611122 | 30.076 |
| 2200 | -0.135510258 | 0.081777766 | 1.912556019 | 0.201345005 | 30.122 |
| 2300 | -0.135432793 | 0.108436372 | 1.896128754 | 0.203626378 | 29.585 |
| 2400 | -0.135543523 | 0.073076268 | 1.905960263 | 0.202099125 | 29.483 |
| 2500 | -0.135687520 | 0.084029950 | 1.922730047 | 0.202303631 | 29.350 |
| 2600 | -0.135693713 | 0.078468680 | 1.918068743 | 0.202800202 | 29.234 |
| 2700 | -0.135748391 | 0.085673504 | 1.918946605 | 0.202579902 | 29.285 |
| 2800 | -0.135791289 | 0.095893211 | 1.941496172 | 0.201126826 | 29.560 |
| 2900 | -0.135865083 | 0.091701756 | 1.942723152 | 0.200603192 | 29.259 |
| 3000 | -0.135803193 | 0.074196012 | 1.955215825 | 0.200304839 | 29.341 |
| 3100 | -0.135824967 | 0.097959872 | 1.949708877 | 0.200712539 | 29.256 |
| 3200 | -0.136001557 | 0.078645931 | 1.977741291 | 0.194562392 | 29.511 |
| 3300 | -0.135934621 | 0.070070809 | 1.975794642 | 0.195650850 | 29.438 |
| 3400 | -0.135933377 | 0.088891857 | 1.986558423 | 0.194049642 | 29.344 |
| 3500 | -0.135977205 | 0.091553292 | 1.999286876 | 0.191088912 | 29.497 |
| 3600 | -0.135953919 | 0.074570217 | 2.000775619 | 0.190814482 | 29.240 |
| 3700 | -0.135946695 | 0.077427374 | 2.004954027 | 0.189419883 | 29.162 |
| 3800 | -0.135999023 | 0.077653699 | 2.012672249 | 0.188206605 | 29.026 |
| 3900 | -0.135987227 | 0.088662133 | 2.013628372 | 0.187779986 | 28.856 |
| 4000 | -0.136037705 | 0.081831231 | 2.015898725 | 0.186628539 | 28.983 |
| 4100 | -0.136084367 | 0.073504253 | 2.015990846 | 0.186303557 | 29.125 |
| 4200 | -0.136076828 | 0.078274486 | 2.019356158 | 0.185759277 | 29.232 |
| 4300 | -0.136067268 | 0.078875951 | 2.019527003 | 0.185454761 | 29.308 |
| 4400 | -0.136044235 | 0.079927898 | 2.020980599 | 0.185112985 | 29.171 |
| 4500 | -0.136074124 | 0.082760428 | 2.020920061 | 0.184610080 | 29.444 |
| 4600 | -0.136094814 | 0.081876747 | 2.022199494 | 0.184165814 | 29.503 |
| 4700 | -0.136080278 | 0.079680792 | 2.022718482 | 0.183992128 | 29.546 |
| 4800 | -0.136080277 | 0.081328262 | 2.023406689 | 0.183710685 | 29.665 |
| 4900 | -0.136079765 | 0.082815149 | 2.024295068 | 0.183379245 | 29.721 |
| 5000 | -0.136077624 | 0.081609507 | 2.024406558 | 0.183213523 | 29.807 |

## Fit diagnostics, execution and provenance

| Fit ID | Iterations (selected initialization) | Inertia | Runtime seconds | Warnings |
|---|---:|---:|---:|---|
| k8_c0_s0 | 29 | 83279.328125 | 10.514 | None |
| k8_c0_s1 | 33 | 83279.320312 | 11.537 | None |
| k8_c0_s2 | 35 | 83279.320312 | 12.027 | None |
| k16_c0_s0 | 61 | 68071.468750 | 19.351 | None |
| k16_c0_s1 | 64 | 67601.648438 | 19.260 | None |
| k16_c0_s2 | 22 | 67601.640625 | 16.806 | None |
| k8_c1_s0 | 39 | 171541.843750 | 12.888 | None |
| k8_c1_s1 | 33 | 171286.000000 | 12.365 | None |
| k8_c1_s2 | 29 | 170065.468750 | 11.778 | None |
| k16_c1_s0 | 98 | 138715.265625 | 22.837 | None |
| k16_c1_s1 | 54 | 139035.093750 | 19.423 | None |
| k16_c1_s2 | 34 | 138536.750000 | 19.238 | None |

Seed sensitivity: uncentered K=8 yields 64.135% liver IoU in all three seeds. Centered K=8 ranges from 47.666% to 66.662%. Its lowest-inertia seed (2, inertia 170065.468750) has liver IoU 47.666%; selecting the highest-IoU centered seed would itself use evaluation labels. Producers: `kmeans.py` fit outputs and this `report.py` aggregation. These runs do not establish that centering is reliably better.

`verify.py` validates the AST-reused confusion/mapping routines on synthetic K=8/K=16 data, all confusion totals, slice/patch counts, and pooled-LOPO ≤ global-optimal IoU. It also retains source hashes and the empty DIAG-probe artifact search in `verification.json`. Uncentered K=8 cross-seed partition ARIs (no labels): K8 center0 seed0 vs seed1: 0.999948838, K8 center0 seed0 vs seed2: 0.999948838, K8 center0 seed1 vs seed2: 1.000000000.

Iteration count reports the selected initialization; it does not establish convergence of every restart. No early stop or tuning is based on evaluation scores. K-means fitting is the only fitting performed in this investigation; no neural/supervised training, installs, derivatives, pipeline caches or checkpoints were created/changed. Scratch arrays are diagnostic intermediates, not pipeline caches.

The initial Anaconda-runtime launch failed during dependency import when the read-only guard rejected a dependency opening `/dev/null` for writing (`extract.log`). It was rerun with the project/FORENSICS Python 3.14 environment (`extract_py314.log`), without changing packages or loosening the guard. An initial report assembly ran before the last fit JSON existed (`report.log`); it was rerun after all fits completed. Neither failure affected computed fit/evaluation results. Successful scripts enforce a 570-second hard timeout. Feature extraction took 80.30977087499923 seconds. Each k-means fit, head rescoring and proxy chunk is separately bounded. All source imports suppress bytecode writes.

Checkpoint paths (existing files only), from `heads.py`/`heads.json`:

- seed0: `/Users/navneetbavineni/STEGO/audit/scratch/fixes3/experiments/fixes3_C_seed0/checkpoints/last.ckpt`.
- seed1: `/Users/navneetbavineni/STEGO/audit/scratch/fixes3/experiments/fixes3_C_seed1/checkpoints/last.ckpt`.
- seed2: `/Users/navneetbavineni/STEGO/audit/scratch/fixes3/experiments/fixes3_C_seed2/checkpoints/last.ckpt`.
- best600: `/Users/navneetbavineni/STEGO/checkpoints/chaos/chaos_fixes3_full_C_date_Sep30_19-47-58/checkpoint-epoch=05-step=1800-mIoU=test/cluster/mIoU=69.3704.ckpt`.
- final: `/Users/navneetbavineni/STEGO/checkpoints/chaos/chaos_fixes3_full_C_date_Sep30_19-47-58/checkpoint-epoch=42-step=15000-mIoU=test/cluster/mIoU=46.1792.ckpt`.

Source commit: `e3c3ab0e396c194a7314d64546c81e9ba4973b72`. Tracked changes relative to HEAD: `none`. Existing untracked run/audit artifacts were left in place. No source edits were made.

## Limits / UNVERIFIED

- DIAG supervised linear-probe rescoring/LOPO: UNVERIFIED because the fitted estimator was not saved; a recoverable fitted artifact is required. No supervised reference was refitted.
- Generalization beyond the four validation patients and this acquisition cohort is UNVERIFIED. LOPO separates mapping labels from scoring labels, but all four patients were already used in the broader project’s diagnostics/checkpoint discussions; it is not a new untouched test cohort. In particular, the existing batch-600 checkpoint was selected by validation cluster mIoU (`audit/run_records/fixes3_full_C/postrun_evidence.json`, callback monitor `test/cluster/mIoU`); checkpoint selection is NOT nested within these LOPO folds. Only mapping calibration is held out.
- Global optimal mappings are optimistic, and their subset choice is not an unsupervised prediction rule. Even LOPO needs three annotated calibration patients.
- A semantic-quality selector validated across independent runs/patients is UNVERIFIED. Within-trajectory correlations are descriptive, not evidence that a proxy will generalize.
- Dense-feature k-means and the trained-head inference use their native, different interpolation paths. This is a same-sample, same-mapping comparison, not an ablation isolating interpolation or resolution effects.
- No clinical boundary-quality claim is made from overlap scores alone.

Final verification: `audit/scratch/baseline/final_checks.py` → `final_checks.json`; {"script": "/Users/navneetbavineni/STEGO/audit/scratch/baseline/final_checks.py", "status": "PASS", "fit_rows": 12, "head_rows": 5, "proxy_checkpoints": 50, "proxy_chunk_runtimes": [134.013781792004, 121.65445183299016, 122.27462341700448, 122.31309133300965, 122.60829212499084], "max_feature_error": 0.0, "max_code_error": 0.0, "best600_final_loss_reproduction_errors": {"600": 0.0, "5000": 0.0}, "tracked_source_changes": false, "elapsed_seconds": 2.6224197080009617}
